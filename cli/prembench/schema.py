"""측정 레코드 스키마 — ROADMAP.md 1.1 과 1:1 로 일치해야 한다. 바꾸면 둘 다.

규칙:
- 모든 키는 필수(레코드에 반드시 존재). 실측하지 못한 값은 생략이 아니라 null.
- 추정·보간 값은 넣지 않는다.
- 거부까지는 아니지만 사이트에서 구분 표시할 레코드는 SchemaWarning 으로 경고한다.
"""

from __future__ import annotations

import warnings
from collections import Counter
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Provenance = Literal["self", "community", "scraped"]
Environment = Literal["cloud_rented", "on_prem"]
Provider = Literal["RunPod", "Vast", "Lambda"]
Interconnect = Literal["PCIe", "NVLink", "none"]
Quantization = Literal["bf16", "fp8", "awq-int4", "gptq-int4", "gguf-q4"]
EngineName = Literal["vllm", "sglang", "ollama", "tgi"]
LoadMode = Literal["closed_loop", "open_loop"]
ArrivalProcess = Literal["poisson", "constant"]
TokenCountSource = Literal["engine_usage", "client_tokenizer"]
FinishReason = Literal["stop", "length", "error", "other"]
ModelLoadState = Literal["cold", "warm"]


class SchemaWarning(UserWarning):
    """유효한 레코드지만 사이트에서 구분 표시가 필요한 경우."""


class _Strict(BaseModel):
    # 오타 필드가 조용히 사라지지 않도록 모르는 키는 거부
    model_config = ConfigDict(extra="forbid")


class Hardware(_Strict):
    gpu_model: str = Field(description="GPU 모델명 (nvidia-smi name). 예: 'NVIDIA L40S'")
    gpu_count: int = Field(ge=1, description="사용한 GPU 개수")
    gpu_vram_gb: float = Field(gt=0, description="GPU 1장당 VRAM, 단위 GB")
    interconnect: Interconnect = Field(description="GPU 간 연결. 단일 GPU 면 'none'")
    cpu_model: str | None = Field(description="CPU 모델명")
    ram_gb: float | None = Field(ge=0, description="시스템 RAM, 단위 GB")
    driver_version: str | None = Field(description="NVIDIA 드라이버 버전")
    cuda_version: str | None = Field(description="CUDA 버전")
    gpu_power_limit_w: float | None = Field(gt=0, description="GPU 1장당 전력 제한, 단위 W (nvidia-smi power.limit)")
    gpu_clock_locked: bool | None = Field(description="GPU 클럭 고정(nvidia-smi -lgc 등) 여부")
    pcie_gen: int | None = Field(ge=1, description="GPU PCIe 링크 세대 (현재값)")
    pcie_lanes: int | None = Field(ge=1, description="GPU PCIe 링크 폭 (현재값), 단위 레인")
    numa_nodes: int | None = Field(ge=1, description="호스트 NUMA 노드 수")


class Model(_Strict):
    hf_id: str = Field(description="Hugging Face 리포 ID. 예: 'Qwen/Qwen2.5-7B-Instruct'")
    params_b: float = Field(gt=0, description="파라미터 수, 단위 십억(B)")
    quantization: Quantization = Field(description="가중치 양자화 방식")
    weight_bytes_gb: float | None = Field(
        gt=0, description="safetensors 파일 합계 실측 크기, 단위 GB"
    )
    max_position_embeddings: int | None = Field(
        ge=1, description="모델 config 의 max_position_embeddings, 단위 토큰"
    )
    moe: bool | None = Field(description="Mixture-of-Experts 모델 여부")
    active_params_b: float | None = Field(
        gt=0, description="토큰당 활성 파라미터 수, 단위 십억(B). dense 모델이면 params_b 와 같다"
    )
    chat_template_applied: bool | None = Field(
        description="요청에 모델 chat template 이 적용됐는지 (chat/completions 면 엔진이 적용)"
    )

    @model_validator(mode="after")
    def _check_active_params(self) -> Model:
        if self.active_params_b is not None and self.active_params_b > self.params_b:
            raise ValueError(f"active_params_b({self.active_params_b}) > params_b({self.params_b})")
        return self


class Engine(_Strict):
    name: EngineName = Field(description="서빙 엔진")
    version: str = Field(description="엔진 버전 문자열")
    tensor_parallel: int = Field(ge=1, description="텐서 병렬 크기")
    pipeline_parallel: int = Field(ge=1, description="파이프라인 병렬 크기")
    max_model_len: int | None = Field(ge=1, description="엔진에 설정한 최대 시퀀스 길이, 단위 토큰")
    gpu_memory_utilization: float | None = Field(
        gt=0, le=1, description="엔진에 설정한 GPU 메모리 사용 비율 (0~1)"
    )
    kv_cache_dtype: str | None = Field(description="KV 캐시 dtype. 예: 'auto', 'fp8'")
    enable_prefix_caching: bool | None = Field(description="prefix 캐싱 활성화 여부")
    extra_args: str | None = Field(description="그 밖의 엔진 실행 인자 원문")
    attention_backend: str | None = Field(description="어텐션 백엔드. 예: 'FLASH_ATTN', 'FLASHINFER'")
    cuda_graphs: bool | None = Field(description="CUDA graph 사용 여부 (vLLM --enforce-eager 면 false)")
    chunked_prefill: bool | None = Field(description="chunked prefill 활성화 여부")
    max_num_seqs: int | None = Field(ge=1, description="엔진의 동시 처리 시퀀스 상한")
    max_num_batched_tokens: int | None = Field(ge=1, description="스텝당 배치 토큰 상한, 단위 토큰")
    speculative_decoding: str | None = Field(
        description="speculative decoding 방식. 예: 'eagle', 'ngram'. 사용 안 하면 'none'"
    )
    engine_commit_or_image_digest: str | None = Field(
        description="엔진 git 커밋 해시 또는 컨테이너 이미지 digest (sha256:...)"
    )


class Sampling(_Strict):
    temperature: float = Field(ge=0, description="샘플링 온도")
    top_p: float = Field(gt=0, le=1, description="nucleus sampling 확률 질량 (0~1]")
    repetition_penalty: float = Field(gt=0, description="반복 패널티. 1.0 이면 없음")
    max_tokens: int = Field(ge=1, description="요청당 최대 출력 토큰, 단위 토큰")
    stop: list[str] = Field(description="stop 문자열 목록. 없으면 빈 배열")


class Load(_Strict):
    concurrency: int = Field(ge=1, description="동시 요청 수")
    input_tokens: int = Field(
        ge=1, description="요청당 입력 길이, 단위 토큰. 엔진이 본 전체 프롬프트(chat template 포함) = usage.prompt_tokens"
    )
    output_tokens: int = Field(ge=1, description="요청당 출력 길이, 단위 토큰")
    request_count: int = Field(ge=1, description="측정 구간의 총 요청 수 (워밍업 제외)")
    warmup_count: int = Field(ge=0, description="측정 전 버린 워밍업 요청 수")
    mode: LoadMode = Field(
        description="closed_loop: 동시성 고정, 끝나면 즉시 다음 요청 / open_loop: 도착률 고정"
    )
    arrival_rate_rps: float | None = Field(
        gt=0, description="open_loop 요청 도착률, 단위 req/s. closed_loop 면 null"
    )
    arrival_process: ArrivalProcess | None = Field(
        description="open_loop 도착 간격 분포. closed_loop 면 null"
    )
    random_seed: int | None = Field(description="프롬프트·도착 간격 생성 난수 시드")
    sampling: Sampling = Field(description="요청에 실어 보낸 샘플링 파라미터. 측정 시 반드시 기록")
    reasoning_enabled: bool | None = Field(description="추론(thinking) 모드 켜짐 여부")
    reasoning_budget_tokens: int | None = Field(
        ge=0, description="추론 토큰 예산, 단위 토큰. 예산을 걸지 않았으면 null"
    )

    @model_validator(mode="after")
    def _check_mode(self) -> Load:
        open_fields = (self.arrival_rate_rps, self.arrival_process)
        if self.mode == "open_loop" and None in open_fields:
            raise ValueError("mode='open_loop' 이면 arrival_rate_rps, arrival_process 가 필요함")
        if self.mode == "closed_loop" and open_fields != (None, None):
            raise ValueError("mode='closed_loop' 이면 arrival_rate_rps, arrival_process 는 null 이어야 함")
        if self.reasoning_budget_tokens is not None and self.reasoning_enabled is not True:
            raise ValueError("reasoning_budget_tokens 는 reasoning_enabled=true 일 때만 기록")
        return self


class PerRequest(_Strict):
    """측정 구간 요청 1개의 원시값. 실패 요청도 포함하며 재지 못한 값은 null."""

    ttft_ms: float | None = Field(ge=0, description="첫 토큰까지 시간, 단위 ms")
    e2e_ms: float | None = Field(ge=0, description="요청 송신~응답 종료, 단위 ms")
    tpot_ms: float | None = Field(
        ge=0, description="(e2e - ttft) / (output_tokens - 1), 단위 ms. output_tokens < 2 면 null"
    )
    output_tokens: int | None = Field(ge=0, description="출력 토큰 수 (token_count_source 기준)")
    finish_reason: FinishReason = Field(
        description="종료 이유. 실패 요청은 'error', 엔진이 stop/length 외 값을 주면 'other'"
    )
    reasoning_tokens: int | None = Field(
        ge=0, description="출력 토큰 중 추론 토큰 수. 엔진이 분리해 주지 않으면 null"
    )


class FinishReasonCounts(_Strict):
    stop: int = Field(ge=0, description="자연 종료(EOS·stop 문자열) 요청 수")
    length: int = Field(ge=0, description="max_tokens 도달로 끊긴 요청 수")
    error: int = Field(ge=0, description="실패 요청 수")
    other: int = Field(ge=0, description="그 밖의 종료 이유 요청 수")

    @property
    def total(self) -> int:
        return self.stop + self.length + self.error + self.other


class Result(_Strict):
    ttft_p50_ms: float | None = Field(ge=0, description="첫 토큰까지 시간 p50, 단위 ms")
    ttft_p95_ms: float | None = Field(ge=0, description="첫 토큰까지 시간 p95, 단위 ms")
    ttft_p99_ms: float | None = Field(ge=0, description="첫 토큰까지 시간 p99, 단위 ms")
    tpot_p50_ms: float | None = Field(ge=0, description="출력 토큰당 시간 p50, 단위 ms")
    tpot_p95_ms: float | None = Field(ge=0, description="출력 토큰당 시간 p95, 단위 ms")
    e2e_p50_ms: float | None = Field(ge=0, description="요청 전체 지연 p50, 단위 ms")
    e2e_p95_ms: float | None = Field(ge=0, description="요청 전체 지연 p95, 단위 ms")
    e2e_p99_ms: float | None = Field(ge=0, description="요청 전체 지연 p99, 단위 ms")
    itl_p99_ms: float | None = Field(ge=0, description="연속 토큰(청크) 도착 간격 p99, 단위 ms")
    output_tps_total: float | None = Field(ge=0, description="전체 출력 처리량, 단위 tokens/s")
    output_tps_per_user: float | None = Field(
        ge=0, description="요청(사용자) 1개당 출력 처리량, 단위 tokens/s"
    )
    requests_per_sec: float | None = Field(ge=0, description="완료 요청 처리량, 단위 req/s")
    error_count: int = Field(ge=0, description="실패한 요청 수")
    oom: bool = Field(description="OOM 발생 여부. OOM 셀도 기록한다")
    gpu_mem_peak_gb: float | None = Field(ge=0, description="GPU 메모리 최대 사용량, 단위 GB (nvidia-smi)")
    gpu_util_avg_pct: float | None = Field(
        ge=0, le=100, description="GPU 사용률 평균, 단위 % (nvidia-smi 1초 샘플링)"
    )
    power_avg_w: float | None = Field(ge=0, description="GPU 전력 평균, 단위 W (nvidia-smi 1초 샘플링)")
    duration_s: float | None = Field(ge=0, description="측정 구간 길이, 단위 초")
    energy_total_j: float | None = Field(
        ge=0, description="측정 구간 GPU 에너지 합계, 단위 J (nvidia-smi 전력 샘플 적분)"
    )
    energy_per_output_token_j: float | None = Field(
        ge=0, description="energy_total_j / 측정 구간 출력 토큰 합계, 단위 J/token"
    )
    output_tokens_actual_mean: float | None = Field(
        ge=0, description="요청당 실제 출력 토큰 수 평균 (요청한 output_tokens 와 다를 수 있음)"
    )
    token_count_source: TokenCountSource | None = Field(
        description="출력 토큰 수 출처: 엔진 usage 보고 / 클라이언트 토크나이저 재계산. 둘 다 없으면 null"
    )
    reasoning_tokens_mean: float | None = Field(
        ge=0, description="요청당 추론 토큰 수 평균. 엔진이 분리해 주지 않으면 null"
    )
    content_tokens_mean: float | None = Field(
        ge=0, description="요청당 본문(추론 제외) 토큰 수 평균. 엔진이 분리해 주지 않으면 null"
    )
    finish_reason_counts: FinishReasonCounts = Field(
        description="측정 구간 요청의 종료 이유별 개수. 합 = request_count"
    )
    content_empty_rate: float = Field(
        ge=0, le=1,
        description="성공했지만 본문이 빈 요청 수 / request_count (추론만 하고 끝난 경우 등)",
    )
    prefix_cache_hit_rate: float | None = Field(
        ge=0, le=1, description="엔진 메트릭의 prefix 캐시 적중률 (0~1). 엔진이 안 주면 null"
    )
    per_request: list[PerRequest] = Field(description="측정 구간 요청별 원시값, 요청 순서대로")

    @model_validator(mode="after")
    def _percentiles_monotonic(self) -> Result:
        for name, seq in (
            ("ttft", (self.ttft_p50_ms, self.ttft_p95_ms, self.ttft_p99_ms)),
            ("tpot", (self.tpot_p50_ms, self.tpot_p95_ms)),
            ("e2e", (self.e2e_p50_ms, self.e2e_p95_ms, self.e2e_p99_ms)),
        ):
            vals = [v for v in seq if v is not None]
            if vals != sorted(vals):
                raise ValueError(f"{name} 백분위가 단조 증가하지 않음: {seq}")
        return self


class Conditions(_Strict):
    """측정 당시 조건. 같은 하드웨어·설정이라도 결과를 바꾸는 것들."""

    model_load_state: ModelLoadState = Field(
        description="측정 직전 모델 상태. cold: 엔진 기동 직후 첫 측정 / warm: 이미 요청을 처리한 엔진"
    )
    colocated_workloads: bool | None = Field(
        description="측정 중 같은 GPU 에 다른 프로세스가 있었는지. 확인 못 했으면 null"
    )
    shared_tenancy_note: str | None = Field(description="자원 공유 상황 메모 (자유 서술)")
    gpu_temp_start_c: float | None = Field(
        description="측정 구간 시작 시 GPU 온도, 단위 °C (nvidia-smi, 여러 장이면 최댓값)"
    )
    gpu_temp_end_c: float | None = Field(
        description="측정 구간 종료 시 GPU 온도, 단위 °C (nvidia-smi, 여러 장이면 최댓값)"
    )
    chat_template_variant: str | None = Field(description="적용한 chat template 이름 또는 해시")


class Run(_Strict):
    """측정 레코드 1건 = (하드웨어, 모델, 엔진, 부하) 격자 셀 1개의 실측 결과."""

    run_id: str = Field(min_length=1, description="레코드 고유 ID")
    submitted_at: datetime = Field(description="제출 시각 (ISO 8601, 타임존 포함)")
    submitter: str = Field(min_length=1, description="제출자 핸들")
    provenance: Provenance = Field(description="출처: 직접 측정 / 커뮤니티 제출 / 외부 수집")
    environment: Environment = Field(
        description="측정 장소. 사이트의 모든 셀에 표시한다. '온프레미스'는 대상 독자이지 측정 장소 주장이 아니다"
    )
    provider: Provider | None = Field(description="클라우드 제공자. on_prem 이면 null")

    hardware: Hardware
    model: Model
    engine: Engine
    load: Load
    conditions: Conditions
    result: Result

    @model_validator(mode="after")
    def _check_consistency(self) -> Run:
        if self.environment == "on_prem" and self.provider is not None:
            raise ValueError("environment='on_prem' 이면 provider 는 null 이어야 함")
        if self.environment == "cloud_rented" and self.provider is None:
            raise ValueError("environment='cloud_rented' 이면 provider 가 필요함")
        if self.submitted_at.tzinfo is None:
            raise ValueError("submitted_at 에 타임존이 없음")
        used = self.engine.tensor_parallel * self.engine.pipeline_parallel
        if used > self.hardware.gpu_count:
            raise ValueError(
                f"tensor_parallel×pipeline_parallel({used}) > gpu_count({self.hardware.gpu_count})"
            )
        if len(self.result.per_request) != self.load.request_count:
            raise ValueError(
                f"per_request 길이({len(self.result.per_request)}) != request_count({self.load.request_count})"
            )
        if self.result.token_count_source is None and (
            self.result.output_tokens_actual_mean is not None
            or any(r.output_tokens is not None for r in self.result.per_request)
        ):
            raise ValueError("token_count_source 가 null 이면 출력 토큰 수도 null 이어야 함")

        frc = self.result.finish_reason_counts
        if frc.total != self.load.request_count:
            raise ValueError(
                f"finish_reason_counts 합({frc.total}) != request_count({self.load.request_count})"
            )
        tally = Counter(r.finish_reason for r in self.result.per_request)
        if tally != Counter({k: v for k, v in frc.model_dump().items() if v}):
            raise ValueError(f"finish_reason_counts 가 per_request 집계와 다름: {dict(tally)}")
        if frc.error != self.result.error_count:
            raise ValueError(
                f"finish_reason_counts.error({frc.error}) != error_count({self.result.error_count})"
            )

        if self.load.reasoning_enabled and self.result.reasoning_tokens_mean is None:
            warnings.warn(
                f"run {self.run_id}: reasoning_enabled=true 인데 reasoning_tokens_mean 이 null "
                "— 추론/본문 토큰 미분리 레코드로 표시",
                SchemaWarning,
                stacklevel=2,
            )
        return self


def json_schema() -> dict:
    return Run.model_json_schema()
