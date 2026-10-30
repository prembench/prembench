"""비동기 부하 생성 — OpenAI-compatible `/v1/chat/completions` 스트리밍.

절차:
1. 준비 확인: 실제 생성 요청 1건이 토큰을 돌려줘야 한다. `/v1/models` 200 은 준비로 치지 않는다
   (모델 적재 중에도 200 을 주는 엔진이 있다). 실패하면 NotReadyError — 측정을 시작하지 않는다.
2. 워밍업: 같은 동시성으로 돌리고 결과는 따로 둔다.
3. 측정: 요청 request_count 개 이상 AND min_duration_s 초 이상. 프롬프트는 측정 전에 미리 만든다.
   시작·종료 시 GPU 온도(nvidia-smi)를 기록한다. 없으면 None.

- 고정 동시성(closed loop): 워커 `concurrency` 개가 요청이 끝나는 즉시 다음 요청을 보낸다.
- 요청마다 '출력이 있는 청크'의 도착 시각을 요청 시작 기준으로 기록한다.
  청크 1개 = 토큰 1개라는 보장은 없다(엔진에 따라 여러 토큰이 한 청크로 옴).
  출력 토큰 수는 엔진이 보고한 usage 만 쓰고, 없으면 None. 청크 수로 추정하지 않는다.
- 실패는 예외로 던지지 않고 레코드에 남긴다 (OOM 셀도 기록해야 하므로). 준비 확인만 예외.

백분위·tps 계산은 metrics.py 몫. 여기서는 원시 시각만 남긴다.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from prembench.probe import GpuSample, GpuSampler, gpu_temperature_c
from prembench.schema import FinishReason, PerRequest, Sampling

_CONNECT_TIMEOUT_S = 10
# 긴 컨텍스트 + 높은 동시성에서는 첫 토큰까지 수 분이 걸릴 수 있다
_DEFAULT_TIMEOUT_S = 600
_ERROR_MAX_CHARS = 200  # schema.PerRequest.error 와 같은 한도
_READY_MAX_TOKENS = 4
_POOL_MARGIN = 2  # 최소 측정 시간용 프롬프트를 워밍업 속도 기준 몇 배 만들지
_MAX_REQUESTS = 20_000  # 측정 1회 요청 상한 (메모리·생성 시간 보호)
# 추론 출력 델타 필드. gpt-oss·Qwen3 등 (엔진마다 이름이 다름)
_REASONING_KEYS = ("reasoning_content", "reasoning")


class NotReadyError(RuntimeError):
    """준비 확인 요청이 토큰을 돌려주지 못함. 측정을 시작하지 않는다."""


@dataclass
class RequestRecord:
    index: int
    warmup: bool
    start_s: float  # 부하 시작 기준 요청 송신 시각, 단위 초
    end_s: float  # 부하 시작 기준 응답 종료 시각, 단위 초
    status: int | None  # HTTP 상태. 연결 실패 등으로 응답이 없으면 None
    error: str | None
    chunk_times_s: list[float] = field(default_factory=list)  # 요청 시작 기준 출력 청크 도착 시각 (추론 포함)
    content_chunks: int = 0  # 본문(추론 제외) 출력이 있었던 청크 수
    finish_reason_raw: str | None = None  # 엔진이 보낸 원문
    completion_tokens: int | None = None  # 엔진 보고 usage. 없으면 None
    reasoning_tokens: int | None = None  # usage.completion_tokens_details.reasoning_tokens. 없으면 None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None  # usage.prompt_tokens_details.cached_tokens (prefix 캐시 적중). 없으면 None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def ttft_s(self) -> float | None:
        return self.chunk_times_s[0] if self.chunk_times_s else None

    @property
    def itl_s(self) -> list[float]:
        """연속한 출력 청크 사이 간격, 단위 초."""
        return [b - a for a, b in itertools.pairwise(self.chunk_times_s)]

    @property
    def e2e_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def content_empty(self) -> bool:
        """성공했는데 본문이 비었는가 (추론만 하고 끝난 경우 등)."""
        return self.ok and self.content_chunks == 0

    @property
    def finish_reason(self) -> FinishReason:
        if not self.ok:
            return "error"
        if self.finish_reason_raw in ("stop", "length"):
            return self.finish_reason_raw
        return "other"

    @property
    def itl_max_s(self) -> float | None:
        return max(self.itl_s) if len(self.chunk_times_s) >= 2 else None

    def to_per_request(self, measure_start_s: float) -> PerRequest:
        """스키마 PerRequest 로 변환. 출력 토큰 수는 엔진 usage 만 쓴다 (token_count_source=engine_usage)."""
        ttft_ms = self.ttft_s * 1000 if self.ttft_s is not None else None
        e2e_ms = self.e2e_s * 1000
        n = self.completion_tokens
        tpot_ms = (e2e_ms - ttft_ms) / (n - 1) if ttft_ms is not None and n is not None and n >= 2 else None
        return PerRequest(
            start_s=max(self.start_s - measure_start_s, 0.0),
            status=self.status,
            error=self.error,
            prompt_tokens=self.prompt_tokens,
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            tpot_ms=tpot_ms,
            output_tokens=n,
            finish_reason=self.finish_reason,
            reasoning_tokens=self.reasoning_tokens,
            content_empty=self.content_empty,
            itl_max_ms=self.itl_max_s * 1000 if self.itl_max_s is not None else None,
        )


@dataclass
class LoadRun:
    ready: RequestRecord  # 준비 확인 요청
    records: list[RequestRecord]  # 측정 구간 (워밍업 제외)
    warmup_records: list[RequestRecord]
    measure_start_s: float  # 부하 시작 기준
    measure_end_s: float
    gpu_temp_start_c: float | None
    gpu_temp_end_c: float | None
    ignore_eos: bool
    gpu_samples: list[GpuSample] | None = None  # 측정 구간 nvidia-smi 샘플. 샘플러가 없으면 None
    pool_exhausted: bool = False  # 미리 만든 프롬프트가 떨어져 min_duration_s 전에 멈춤

    def per_request(self) -> list[PerRequest]:
        return [r.to_per_request(self.measure_start_s) for r in self.records]

    @property
    def duration_s(self) -> float:
        return self.measure_end_s - self.measure_start_s

    @property
    def error_count(self) -> int:
        return sum(not r.ok for r in self.records)


def _chat_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    index: int,
    warmup: bool,
    t0: float,
) -> RequestRecord:
    start = time.perf_counter()
    chunk_times: list[float] = []
    content_chunks = 0
    finish_reason: str | None = None
    usage: dict | None = None
    status: int | None = None
    error: str | None = None
    try:
        async with client.stream("POST", url, json=body) as resp:
            status = resp.status_code
            if status != 200:
                text = (await resp.aread()).decode(errors="replace")
                error = f"HTTP {status}: {text}"
            else:
                async for line in resp.aiter_lines():
                    now = time.perf_counter()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        error = f"bad SSE payload: {data}"
                        break
                    if chunk.get("error"):
                        error = f"stream error: {json.dumps(chunk['error'], ensure_ascii=False)}"
                        break
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        has_content = bool(delta.get("content"))
                        if has_content or any(delta.get(k) for k in _REASONING_KEYS):
                            chunk_times.append(now - start)
                            content_chunks += has_content
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
    except httpx.HTTPError as e:
        error = f"{type(e).__name__}: {e}"
    end = time.perf_counter()

    if error is None and not chunk_times:
        error = "no output tokens in stream"
    if error is not None:
        error = error[:_ERROR_MAX_CHARS]
    usage = usage or {}
    return RequestRecord(
        index=index,
        warmup=warmup,
        start_s=start - t0,
        end_s=end - t0,
        status=status,
        error=error,
        chunk_times_s=chunk_times,
        content_chunks=content_chunks,
        finish_reason_raw=finish_reason,
        completion_tokens=usage.get("completion_tokens"),
        reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        prompt_tokens=usage.get("prompt_tokens"),
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
    )


async def _phase(
    client: httpx.AsyncClient,
    url: str,
    bodies: list[dict],
    concurrency: int,
    warmup: bool,
    t0: float,
    min_count: int | None = None,
    min_duration_s: float = 0.0,
) -> list[RequestRecord]:
    """bodies 를 고정 동시성으로 보낸다.

    min_count 가 있으면 '요청 min_count 개 이상 AND 경과 min_duration_s 이상' 이 되는 순간 새 요청을 멈춘다
    (이미 보낸 요청은 끝까지 받는다). bodies 가 먼저 떨어지면 거기서 멈춘다 — 재사용하지 않는다.
    """
    out: list[RequestRecord] = []
    next_i = 0
    phase_start = time.perf_counter()

    def take() -> int | None:
        nonlocal next_i
        if next_i >= len(bodies):
            return None
        if (
            min_count is not None
            and next_i >= min_count
            and time.perf_counter() - phase_start >= min_duration_s
        ):
            return None
        i, next_i = next_i, next_i + 1
        return i

    async def worker() -> None:
        # 단일 이벤트 루프라 take() 는 원자적
        while (i := take()) is not None:
            out.append(await _one_request(client, url, bodies[i], i, warmup, t0))

    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(bodies)))))
    out.sort(key=lambda r: r.index)
    return out


def request_body(
    model: str,
    prompt: str,
    sampling: Sampling,
    extra_body: dict | None = None,
    ignore_eos: bool = False,
) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "repetition_penalty": sampling.repetition_penalty,
        "max_tokens": sampling.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # 빈 stop 배열을 거부하는 서버가 있어 비었으면 보내지 않는다
    if sampling.stop:
        body["stop"] = list(sampling.stop)
    if sampling.seed is not None:
        body["seed"] = sampling.seed
    # 엔진 확장 필드라 켤 때만 보낸다 (vLLM·SGLang 지원, Ollama 는 무시)
    if ignore_eos:
        body["ignore_eos"] = True
    return {**body, **(extra_body or {})}


async def run_load(
    base_url: str,
    model: str,
    prompt: Callable[[int], str],
    *,
    sampling: Sampling,
    concurrency: int,
    request_count: int,
    warmup_count: int = 0,
    min_duration_s: float = 0.0,
    max_requests: int = _MAX_REQUESTS,
    extra_body: dict | None = None,
    ignore_eos: bool = False,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    headers: dict[str, str] | None = None,
    gpu_temp: Callable[[], float | None] = gpu_temperature_c,
    gpu_sampler: GpuSampler | None = None,
) -> LoadRun:
    """준비 확인 1건 → 워밍업 `warmup_count` 개 → 측정 (요청 `request_count` 개 이상 AND `min_duration_s` 초 이상).

    prompt(i): i 번째 요청의 본문. -1 은 준비 확인용, 0.. 은 워밍업 다음 측정. 요청마다 달라야 prefix 캐시를 피한다.
    프롬프트는 측정 중에 만들지 않는다 (생성이 이벤트 루프를 막아 토큰 도착 시각이 밀린다).
    측정 분량은 워밍업이 끝난 뒤(요청이 하나도 없을 때) 워밍업 처리 속도로 min_duration_s 의
    2배(_POOL_MARGIN)를 미리 만든다. 그래도 모자라면 반복 사용하지 않고 멈추고 LoadRun.pool_exhausted 로 알린다.

    base_url 은 `/v1` 까지 포함. 예: http://localhost:11434/v1
    extra_body 는 엔진별 옵션 (예: vLLM 의 {"min_tokens": N}).
    """
    if concurrency < 1 or request_count < 1 or warmup_count < 0 or min_duration_s < 0:
        raise ValueError("concurrency>=1, request_count>=1, warmup_count>=0, min_duration_s>=0")

    def body(text: str) -> dict:
        return request_body(model, text, sampling, extra_body, ignore_eos)

    url = _chat_url(base_url)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(timeout_s, connect=_CONNECT_TIMEOUT_S)
    async with httpx.AsyncClient(limits=limits, timeout=timeout, headers=headers) as client:
        t0 = time.perf_counter()
        ready_body = body(prompt(-1))
        ready_body["max_tokens"] = min(sampling.max_tokens, _READY_MAX_TOKENS)
        ready = await _one_request(client, url, ready_body, -1, True, t0)
        if not ready.ok:
            raise NotReadyError(f"준비 확인 실패, 측정 시작 안 함: {ready.error}")

        warm_start = time.perf_counter()
        warm = await _phase(
            client, url, [body(prompt(i)) for i in range(warmup_count)], concurrency, True, t0
        )
        warm_s = time.perf_counter() - warm_start

        # 측정 분량 미리 생성 (진행 중인 요청 없음)
        pool = request_count
        if min_duration_s > 0 and warm and warm_s > 0:
            pool = max(pool, math.ceil(len(warm) / warm_s * min_duration_s * _POOL_MARGIN) + concurrency)
        pool = max(request_count, min(pool, max_requests))
        bodies = [body(prompt(warmup_count + i)) for i in range(pool)]

        temp_start = await asyncio.to_thread(gpu_temp)
        if gpu_sampler is not None:
            gpu_sampler.start()
        measure_start = time.perf_counter() - t0
        measured = await _phase(
            client, url, bodies, concurrency, False, t0, min_count=request_count, min_duration_s=min_duration_s
        )
        measure_end = time.perf_counter() - t0
        samples = await asyncio.to_thread(gpu_sampler.stop) if gpu_sampler is not None else None
        temp_end = await asyncio.to_thread(gpu_temp)
    exhausted = len(measured) == len(bodies) and (measure_end - measure_start) < min_duration_s
    return LoadRun(
        ready, measured, warm, measure_start, measure_end, temp_start, temp_end, ignore_eos, samples,
        pool_exhausted=exhausted,
    )
