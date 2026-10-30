"""격자 셀 측정: probe → 프롬프트 → 부하 → 지표 → Run 조립.

Session 이 엔진·하드웨어·모델 메타데이터·토크나이저·템플릿 오버헤드를 한 번만 준비하고,
measure() 가 셀(동시성, 입력, 출력) 1회를 재서 Run 을 만든다. grid.py 는 measure() 를 셀·회차마다 부른다.

모르는 값은 null, 필수인데 알아낼 수 없는 값은 CellError (추측해서 채우지 않는다).
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
    EngineProbe,
    GpuSampler,
    apple_hardware,
    ollama_loaded,
    probe_engine,
    probe_host,
    probe_ollama_model,
)
from prembench.prompts import (
    Tokenizer,
    TokenPools,
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
    Repetition,
    Run,
    Sampling,
)

# ignore_eos 를 실제로 적용하는 엔진. 나머지는 받아도 무시하므로 레코드에 null
_IGNORE_EOS_ENGINES = {"vllm", "sglang"}
_DTYPES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
_MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")
SINGLE = Repetition(index=1, count=1, representative=True)


class CellError(RuntimeError):
    """측정을 시작할 수 없거나, 결과를 레코드로 만들 수 없음."""


class PromptTokenMismatch(CellError):
    """엔진이 센 프롬프트 토큰이 input_tokens 와 다름. 레코드를 만들지 않는다."""


@dataclass
class Target:
    """측정 대상 (엔진·모델) + 제출자 정보 + 측정 규칙. 셀과 무관하게 고정."""

    url: str
    model: str
    hf_id: str
    submitter: str
    provenance: str
    environment: str
    provider: str | None
    batch_id: str | None = None
    ignore_eos: bool = False
    warmup_count: int | None = None  # None → concurrency
    request_count: int | None = None  # None → concurrency × 5
    min_duration_s: float = 60.0  # ROADMAP 1.4: 요청 수와 60초 중 긴 쪽
    tensor_parallel: int | None = None
    pipeline_parallel: int | None = None
    quantization: str | None = None
    weight_dtype: str | None = None
    model_load_state: str | None = None


@dataclass(frozen=True)
class Cell:
    concurrency: int
    input_tokens: int
    output_tokens: int

    def __str__(self) -> str:
        return f"c{self.concurrency} in{self.input_tokens} out{self.output_tokens}"


def is_local_url(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return True
    if host in {socket.gethostname(), socket.getfqdn()}:
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


def _model_ollama(t: Target) -> tuple[Model, str | None]:
    info = probe_ollama_model(t.url, t.model)
    if info is None or info.parameter_count is None:
        raise CellError(f"Ollama /api/show 에서 {t.model} 의 파라미터 수를 못 얻음")
    params_b = info.parameter_count / 1e9
    moe = bool(info.expert_count and info.expert_count > 1)
    quant = t.quantization or (
        f"gguf-{info.quantization_level.lower()}" if info.quantization_level else None
    )
    if quant is None:
        raise CellError("양자화 방식을 알 수 없음 — --quantization 필요")
    model = Model(
        hf_id=t.hf_id,
        params_b=params_b,
        quantization=quant,
        weight_dtype=t.weight_dtype,
        weight_bytes_gb=info.size_bytes / 1024**3 if info.size_bytes else None,
        max_position_embeddings=info.context_length_max,
        moe=moe,
        active_params_b=None if moe else params_b,
        chat_template_applied=True,
    )
    template = f"sha256:{info.template_sha256}" if info.template_sha256 else None
    return model, template


def _model_hf(t: Target) -> Model:
    try:
        cfg = hf_config(t.hf_id)
        params = hf_param_count(t.hf_id)
    except Exception as e:  # 네트워크·권한 오류를 사용자에게 그대로
        raise CellError(f"HF 메타데이터를 못 받음 ({t.hf_id}): {e}") from e
    if params is None:
        raise CellError(f"HF API 에 {t.hf_id} 의 파라미터 수가 없음")
    params_b = params / 1e9
    experts = next((cfg[k] for k in _MOE_KEYS if cfg.get(k)), None)
    moe = bool(experts and experts > 1)
    qcfg = cfg.get("quantization_config") or {}
    quant = t.quantization
    if quant is None:
        if qcfg.get("quant_method"):
            bits = qcfg.get("bits")
            quant = f"{qcfg['quant_method']}-int{bits}" if bits else str(qcfg["quant_method"])
        else:
            quant = "none"
    dtype = t.weight_dtype or (_DTYPES.get(cfg.get("torch_dtype")) if quant == "none" else None)
    return Model(
        hf_id=t.hf_id,
        params_b=params_b,
        quantization=quant,
        weight_dtype=dtype,
        weight_bytes_gb=None,
        max_position_embeddings=cfg.get("max_position_embeddings"),
        moe=moe,
        active_params_b=None if moe else params_b,
        chat_template_applied=True,
    )


def _parallel(t: Target, gpu_count: int) -> tuple[int, int]:
    tp, pp = t.tensor_parallel, t.pipeline_parallel
    if gpu_count == 1:
        # GPU 1장이면 TP×PP ≤ 1 이므로 둘 다 1 로 확정된다 (추정 아님)
        if (tp or 1) != 1 or (pp or 1) != 1:
            raise CellError("GPU 가 1장인데 --tp/--pp 가 1 이 아님")
        return 1, 1
    missing = [n for n, v in (("--tp", tp), ("--pp", pp)) if v is None]
    if missing:
        raise CellError(f"GPU {gpu_count}장 — 병렬 설정을 엔진 API 로 알 수 없음: {' '.join(missing)} 필요")
    return tp, pp


class Session:
    """엔진·모델 1개에 대한 측정 세션. open() 으로 만든다."""

    def __init__(
        self,
        target: Target,
        engine: EngineProbe,
        hardware: Hardware,
        parallel: tuple[int, int],
        model: Model,
        template_variant: str | None,
        tok: Tokenizer,
        pools: TokenPools,
        overhead: int,
        load_state: str,
    ) -> None:
        self.target = target
        self.engine = engine
        self.hardware = hardware
        self.parallel = parallel
        self.model = model
        self.template_variant = template_variant
        self.tok = tok
        self.pools = pools
        self.overhead = overhead
        self.load_state = load_state
        self.max_model_len = self._max_model_len()
        self.last_pool_exhausted = False

    @property
    def is_ollama(self) -> bool:
        return self.engine.name == "ollama"

    @classmethod
    async def open(cls, t: Target, calibration_seed: int = 0) -> Session:
        if not is_local_url(t.url):
            raise CellError(
                "원격 URL 은 아직 지원하지 않음 — GPU 호스트의 하드웨어를 감지할 수 없다. GPU 서버에서 직접 실행할 것"
            )
        engine = probe_engine(t.url)
        if not engine.reachable:
            raise CellError(f"엔진에 연결 못 함: {engine.error}")
        if engine.name is None or engine.version is None:
            raise CellError("엔진 종류·버전을 식별하지 못함")
        if t.model not in engine.models:
            raise CellError(f"엔진에 {t.model!r} 없음. 제공 모델: {engine.models}")
        is_ollama = engine.name == "ollama"

        hardware = _hardware()
        parallel = _parallel(t, hardware.gpu_count)

        # 모델 상태는 우리 요청을 보내기 전에 본다
        load_state = t.model_load_state
        if load_state is None and is_ollama:
            load_state = "warm" if ollama_loaded(t.url, t.model) else "cold"
        if load_state is None:
            raise CellError("모델 적재 상태를 엔진 API 로 알 수 없음 — --model-load-state cold|warm 필요")

        if is_ollama:
            model, template_variant = _model_ollama(t)
        else:
            model, template_variant = _model_hf(t), None

        tok = load_hf_tokenizer(t.hf_id)
        pools = token_pools(tok)
        overhead = await measure_template_overhead(t.url, t.model, tok, calibration_seed, pools)
        return cls(t, engine, hardware, parallel, model, template_variant, tok, pools, overhead, load_state)

    def _max_model_len(self) -> int | None:
        if self.is_ollama:
            # 런너가 실제로 잡은 context (모델 최대가 아니라 설정값)
            return (ollama_loaded(self.target.url, self.target.model) or {}).get("context_length")
        return self.engine.max_model_len

    def skip_reason(self, cell: Cell) -> str | None:
        """측정 전에 확실히 안 되는 셀이면 이유. 아니면 None."""
        if cell.input_tokens <= self.overhead:
            return f"--in {cell.input_tokens} 가 chat template 오버헤드({self.overhead}) 이하"
        if self.max_model_len is not None and cell.input_tokens + cell.output_tokens > self.max_model_len:
            return f"입력+출력 {cell.input_tokens + cell.output_tokens} > 엔진 max_model_len {self.max_model_len}"
        return None

    async def measure(self, cell: Cell, seed: int, repetition: Repetition = SINGLE) -> Run:
        t = self.target
        content_n = content_tokens_for(cell.input_tokens, self.overhead)
        warmup = cell.concurrency if t.warmup_count is None else t.warmup_count
        requests = cell.concurrency * 5 if t.request_count is None else t.request_count

        def prompt(i: int) -> str:
            return random_text(self.tok, content_n, f"{seed}:{i}", self.pools)

        engine_name = self.engine.name
        send_ignore_eos = t.ignore_eos and engine_name in _IGNORE_EOS_ENGINES
        sampling = Sampling(
            temperature=0.0, top_p=1.0, repetition_penalty=1.0,
            max_tokens=cell.output_tokens, stop=[], seed=seed,
        )
        load_state = self.load_state
        run = await run_load(
            t.url, t.model, prompt,
            sampling=sampling, concurrency=cell.concurrency, request_count=requests,
            warmup_count=warmup, min_duration_s=t.min_duration_s, ignore_eos=send_ignore_eos,
            gpu_sampler=GpuSampler.create(),
        )
        self.load_state = "warm"  # 한 번 잰 뒤로는 적재된 상태

        # input_tokens 는 엔진이 센 프롬프트 토큰이어야 한다 — 다르면 레코드가 거짓이 된다
        wrong = [(r.index, r.prompt_tokens) for r in run.records if r.ok and r.prompt_tokens != cell.input_tokens]
        if wrong:
            shown = ", ".join(f"#{i}={n}" for i, n in wrong[:5])
            raise PromptTokenMismatch(
                f"엔진이 센 프롬프트 토큰 ≠ --in {cell.input_tokens}: {len(wrong)}/{len(run.records)}건 "
                f"[{shown}] (seed {seed}; 엔진 context 한도로 잘렸거나 토크나이저 불일치)"
            )
        self.max_model_len = self._max_model_len() if self.is_ollama else self.max_model_len

        tp, pp = self.parallel
        engine = Engine(
            name=engine_name, version=self.engine.version,
            tensor_parallel=tp, pipeline_parallel=pp, data_parallel=None, expert_parallel=None,
            max_model_len=self.max_model_len, gpu_memory_utilization=None, kv_cache_dtype=None,
            enable_prefix_caching=None, server_command=None, extra_args=None,
            attention_backend=None, cuda_graphs=None, chunked_prefill=None,
            max_num_seqs=None, max_num_batched_tokens=None, speculative_decoding=None,
            engine_commit_or_image_digest=None,
        )
        load = Load(
            concurrency=cell.concurrency, input_tokens=cell.input_tokens, output_tokens=cell.output_tokens,
            request_count=len(run.records), warmup_count=warmup, mode="closed_loop",
            arrival_rate_rps=None, arrival_process=None, random_seed=seed, sampling=sampling,
            reasoning_enabled=None, reasoning_budget_tokens=None,
            ignore_eos=t.ignore_eos if engine_name in _IGNORE_EOS_ENGINES else None,
        )
        conditions = Conditions(
            measured_from="same_host", model_load_state=load_state, colocated_workloads=None,
            shared_tenancy_note=None, gpu_temp_start_c=run.gpu_temp_start_c, gpu_temp_end_c=run.gpu_temp_end_c,
            chat_template_variant=self.template_variant,
        )
        now = datetime.now(timezone.utc)
        record = Run(
            schema_version=SCHEMA_VERSION,
            prembench_version=__version__,
            run_id=f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}",
            batch_id=t.batch_id,
            submitted_at=now,
            submitter=t.submitter,
            provenance=t.provenance,
            environment=t.environment,
            provider=t.provider,
            hardware=self.hardware,
            model=self.model,
            engine=engine,
            load=load,
            conditions=conditions,
            result=build_result(run, cell.concurrency),
            repetition=repetition,
        )
        # 스키마 밖 진단: 미리 만든 프롬프트가 떨어져 min_duration_s 를 못 채웠는가
        self.last_pool_exhausted = run.pool_exhausted
        return record


async def run_cell(target: Target, cell: Cell, seed: int) -> Run:
    """셀 1개 1회 (prembench run)."""
    session = await Session.open(target, calibration_seed=seed)
    if reason := session.skip_reason(cell):
        raise CellError(f"측정 불가: {reason}")
    return await session.measure(cell, seed)
