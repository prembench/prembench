"""측정 레코드 스키마 — ROADMAP.md 1.1 과 1:1 로 일치해야 한다. 바꾸면 둘 다.

규칙:
- 모든 키는 필수(레코드에 반드시 존재). 실측하지 못한 값은 생략이 아니라 null.
- 추정·보간 값은 넣지 않는다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Provenance = Literal["self", "community", "scraped"]
Environment = Literal["cloud_rented", "on_prem"]
Provider = Literal["RunPod", "Vast", "Lambda"]
Interconnect = Literal["PCIe", "NVLink", "none"]
Quantization = Literal["bf16", "fp8", "awq-int4", "gptq-int4", "gguf-q4"]
EngineName = Literal["vllm", "sglang", "ollama", "tgi"]


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


class Load(_Strict):
    concurrency: int = Field(ge=1, description="동시 요청 수")
    input_tokens: int = Field(ge=1, description="요청당 입력 길이, 단위 토큰")
    output_tokens: int = Field(ge=1, description="요청당 출력 길이, 단위 토큰")
    request_count: int = Field(ge=1, description="측정 구간의 총 요청 수 (워밍업 제외)")
    warmup_count: int = Field(ge=0, description="측정 전 버린 워밍업 요청 수")


class Result(_Strict):
    ttft_p50_ms: float | None = Field(ge=0, description="첫 토큰까지 시간 p50, 단위 ms")
    ttft_p95_ms: float | None = Field(ge=0, description="첫 토큰까지 시간 p95, 단위 ms")
    ttft_p99_ms: float | None = Field(ge=0, description="첫 토큰까지 시간 p99, 단위 ms")
    tpot_p50_ms: float | None = Field(ge=0, description="출력 토큰당 시간 p50, 단위 ms")
    tpot_p95_ms: float | None = Field(ge=0, description="출력 토큰당 시간 p95, 단위 ms")
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

    @model_validator(mode="after")
    def _percentiles_monotonic(self) -> Result:
        for name, seq in (
            ("ttft", (self.ttft_p50_ms, self.ttft_p95_ms, self.ttft_p99_ms)),
            ("tpot", (self.tpot_p50_ms, self.tpot_p95_ms)),
        ):
            vals = [v for v in seq if v is not None]
            if vals != sorted(vals):
                raise ValueError(f"{name} 백분위가 단조 증가하지 않음: {seq}")
        return self


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
        return self


def json_schema() -> dict:
    return Run.model_json_schema()
