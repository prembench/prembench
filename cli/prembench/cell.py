"""격자 셀 1개 측정: probe → 프롬프트 생성 → 부하 → 지표 → Run 조립.

모르는 값은 null, 필수인데 알아낼 수 없는 값은 CellError (추측해서 채우지 않는다).
grid.py 는 이 함수를 셀마다 부른다.
"""

from __future__ import annotations

import ipaddress
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

from prembench import __version__
from prembench.hf import hf_config, hf_param_count
from prembench.load import run_load
from prembench.metrics import build_result
from prembench.probe import (
    GpuSampler,
    apple_hardware,
    ollama_loaded,
    probe_engine,
    probe_host,
    probe_ollama_model,
)
from prembench.prompts import (
    build_prompts,
    content_tokens_for,
    load_hf_tokenizer,
    measure_template_overhead,
    random_text,
    token_pools,
)
from prembench.schema import (
    SCHEMA_VERSION,
    Conditions,
    Engine,
    Hardware,
    Load,
    Model,
    Run,
    Sampling,
)

# ignore_eos 를 실제로 적용하는 엔진. 나머지는 받아도 무시하므로 레코드에 null
_IGNORE_EOS_ENGINES = {"vllm", "sglang"}
_DTYPES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
_MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")


class CellError(RuntimeError):
    """측정을 시작할 수 없거나, 결과를 레코드로 만들 수 없음."""


@dataclass
class CellSpec:
    url: str
    model: str
    hf_id: str
    concurrency: int
    input_tokens: int
    output_tokens: int
    submitter: str
    provenance: str
    environment: str
    provider: str | None
    seed: int = 0
    warmup_count: int | None = None  # None → concurrency
    request_count: int | None = None  # None → concurrency × 5
    ignore_eos: bool = False
    batch_id: str | None = None
    tensor_parallel: int | None = None
    pipeline_parallel: int | None = None
    quantization: str | None = None
    weight_dtype: str | None = None
    model_load_state: str | None = None


def is_local_url(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return True
    local_names = {socket.gethostname(), socket.getfqdn()}
    if host in local_names:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(host))
        except (OSError, ValueError):
            return False
    if ip.is_loopback or ip.is_unspecified:
        return True
    try:
        own = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        own = []
    return str(ip) in own


def _hardware() -> Hardware:
    host = probe_host()
    if host.gpus:
        return host.to_hardware()
    apple = apple_hardware()
    if apple is None:
        raise CellError(f"GPU 미검출 ({host.no_gpu_reason}) — 측정할 하드웨어가 없음")
    return apple


def _model_ollama(spec: CellSpec) -> tuple[Model, str | None]:
    info = probe_ollama_model(spec.url, spec.model)
    if info is None or info.parameter_count is None:
        raise CellError(f"Ollama /api/show 에서 {spec.model} 의 파라미터 수를 못 얻음")
    params_b = info.parameter_count / 1e9
    moe = bool(info.expert_count and info.expert_count > 1)
    quant = spec.quantization or (
        f"gguf-{info.quantization_level.lower()}" if info.quantization_level else None
    )
    if quant is None:
        raise CellError("양자화 방식을 알 수 없음 — --quantization 필요")
    model = Model(
        hf_id=spec.hf_id,
        params_b=params_b,
        quantization=quant,
        weight_dtype=spec.weight_dtype,
        weight_bytes_gb=info.size_bytes / 1024**3 if info.size_bytes else None,
        max_position_embeddings=info.context_length_max,
        moe=moe,
        active_params_b=None if moe else params_b,
        chat_template_applied=True,
    )
    template = f"sha256:{info.template_sha256}" if info.template_sha256 else None
    return model, template


def _model_hf(spec: CellSpec) -> Model:
    try:
        cfg = hf_config(spec.hf_id)
        params = hf_param_count(spec.hf_id)
    except Exception as e:  # 네트워크·권한 오류를 사용자에게 그대로
        raise CellError(f"HF 메타데이터를 못 받음 ({spec.hf_id}): {e}") from e
    if params is None:
        raise CellError(f"HF API 에 {spec.hf_id} 의 파라미터 수가 없음")
    params_b = params / 1e9
    experts = next((cfg[k] for k in _MOE_KEYS if cfg.get(k)), None)
    moe = bool(experts and experts > 1)
    qcfg = cfg.get("quantization_config") or {}
    quant = spec.quantization
    if quant is None:
        if qcfg.get("quant_method"):
            bits = qcfg.get("bits")
            quant = f"{qcfg['quant_method']}-int{bits}" if bits else str(qcfg["quant_method"])
        else:
            quant = "none"
    dtype = spec.weight_dtype or (_DTYPES.get(cfg.get("torch_dtype")) if quant == "none" else None)
    return Model(
        hf_id=spec.hf_id,
        params_b=params_b,
        quantization=quant,
        weight_dtype=dtype,
        weight_bytes_gb=None,
        max_position_embeddings=cfg.get("max_position_embeddings"),
        moe=moe,
        active_params_b=None if moe else params_b,
        chat_template_applied=True,
    )


def _parallel(spec: CellSpec, gpu_count: int) -> tuple[int, int]:
    tp, pp = spec.tensor_parallel, spec.pipeline_parallel
    if gpu_count == 1:
        # GPU 1장이면 TP×PP ≤ 1 이므로 둘 다 1 로 확정된다 (추정 아님)
        if (tp or 1) != 1 or (pp or 1) != 1:
            raise CellError("GPU 가 1장인데 --tp/--pp 가 1 이 아님")
        return 1, 1
    missing = [n for n, v in (("--tp", tp), ("--pp", pp)) if v is None]
    if missing:
        raise CellError(f"GPU {gpu_count}장 — 병렬 설정을 엔진 API 로 알 수 없음: {' '.join(missing)} 필요")
    return tp, pp


async def run_cell(spec: CellSpec) -> Run:
    if not is_local_url(spec.url):
        raise CellError(
            "원격 URL 은 아직 지원하지 않음 — GPU 호스트의 하드웨어를 감지할 수 없다. GPU 서버에서 직접 실행할 것"
        )
    engine_probe = probe_engine(spec.url)
    if not engine_probe.reachable:
        raise CellError(f"엔진에 연결 못 함: {engine_probe.error}")
    if engine_probe.name is None or engine_probe.version is None:
        raise CellError("엔진 종류·버전을 식별하지 못함")
    if spec.model not in engine_probe.models:
        raise CellError(f"엔진에 {spec.model!r} 없음. 제공 모델: {engine_probe.models}")
    is_ollama = engine_probe.name == "ollama"

    hardware = _hardware()
    tp, pp = _parallel(spec, hardware.gpu_count)

    # 모델 상태는 우리 요청을 보내기 전에 본다
    load_state = spec.model_load_state
    if load_state is None and is_ollama:
        load_state = "warm" if ollama_loaded(spec.url, spec.model) else "cold"
    if load_state is None:
        raise CellError("모델 적재 상태를 엔진 API 로 알 수 없음 — --model-load-state cold|warm 필요")

    if is_ollama:
        model, template_variant = _model_ollama(spec)
    else:
        model, template_variant = _model_hf(spec), None

    tok = load_hf_tokenizer(spec.hf_id)
    pools = token_pools(tok)
    overhead = await measure_template_overhead(spec.url, spec.model, tok, spec.seed, pools)
    content_n = content_tokens_for(spec.input_tokens, overhead)

    warmup = spec.concurrency if spec.warmup_count is None else spec.warmup_count
    requests = spec.concurrency * 5 if spec.request_count is None else spec.request_count
    prompts = build_prompts(tok, warmup + requests, content_n, spec.seed, pools=pools)
    ready_prompt = random_text(tok, content_n, f"{spec.seed}:ready", pools)

    send_ignore_eos = spec.ignore_eos and engine_probe.name in _IGNORE_EOS_ENGINES
    sampling = Sampling(
        temperature=0.0, top_p=1.0, repetition_penalty=1.0,
        max_tokens=spec.output_tokens, stop=[], seed=spec.seed,
    )
    run = await run_load(
        spec.url, spec.model, prompts,
        sampling=sampling, concurrency=spec.concurrency, request_count=requests,
        warmup_count=warmup, ignore_eos=send_ignore_eos,
        gpu_sampler=GpuSampler.create(), ready_prompt=ready_prompt,
    )

    # input_tokens 는 엔진이 센 프롬프트 토큰이어야 한다 — 다르면 레코드가 거짓이 된다
    wrong = sorted({r.prompt_tokens for r in run.records if r.ok and r.prompt_tokens != spec.input_tokens},
                   key=lambda x: (x is None, x))
    if wrong:
        raise CellError(
            f"엔진이 센 프롬프트 토큰 {wrong} ≠ --in {spec.input_tokens} "
            "(엔진 context 한도로 잘렸거나 토크나이저 불일치)"
        )

    max_model_len = engine_probe.max_model_len
    if is_ollama:
        loaded = ollama_loaded(spec.url, spec.model) or {}
        max_model_len = loaded.get("context_length")

    engine = Engine(
        name=engine_probe.name, version=engine_probe.version,
        tensor_parallel=tp, pipeline_parallel=pp, data_parallel=None, expert_parallel=None,
        max_model_len=max_model_len, gpu_memory_utilization=None, kv_cache_dtype=None,
        enable_prefix_caching=None, server_command=None, extra_args=None,
        attention_backend=None, cuda_graphs=None, chunked_prefill=None,
        max_num_seqs=None, max_num_batched_tokens=None, speculative_decoding=None,
        engine_commit_or_image_digest=None,
    )
    load = Load(
        concurrency=spec.concurrency, input_tokens=spec.input_tokens, output_tokens=spec.output_tokens,
        request_count=requests, warmup_count=warmup, mode="closed_loop",
        arrival_rate_rps=None, arrival_process=None, random_seed=spec.seed, sampling=sampling,
        reasoning_enabled=None, reasoning_budget_tokens=None,
        ignore_eos=spec.ignore_eos if engine_probe.name in _IGNORE_EOS_ENGINES else None,
    )
    conditions = Conditions(
        measured_from="same_host", model_load_state=load_state, colocated_workloads=None,
        shared_tenancy_note=None, gpu_temp_start_c=run.gpu_temp_start_c, gpu_temp_end_c=run.gpu_temp_end_c,
        chat_template_variant=template_variant,
    )
    now = datetime.now(timezone.utc)
    return Run(
        schema_version=SCHEMA_VERSION,
        prembench_version=__version__,
        run_id=f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}",
        batch_id=spec.batch_id,
        submitted_at=now,
        submitter=spec.submitter,
        provenance=spec.provenance,
        environment=spec.environment,
        provider=spec.provider,
        hardware=hardware,
        model=model,
        engine=engine,
        load=load,
        conditions=conditions,
        result=build_result(run, spec.concurrency),
    )
