"""비동기 부하 생성 — OpenAI-compatible `/v1/chat/completions` 스트리밍.

- 고정 동시성(closed loop): 워커 `concurrency` 개가 요청이 끝나는 즉시 다음 요청을 보낸다.
- 워밍업 요청을 먼저 같은 동시성으로 돌리고, 끝난 뒤 측정 구간을 시작한다.
- 요청마다 '내용이 있는 청크'의 도착 시각을 요청 시작 기준으로 기록한다.
  청크 1개 = 토큰 1개라는 보장은 없다(엔진에 따라 여러 토큰이 한 청크로 옴).
  출력 토큰 수는 엔진이 보고한 usage.completion_tokens 만 쓰고, 없으면 None. 청크 수로 추정하지 않는다.
- 실패는 예외로 던지지 않고 레코드에 남긴다 (OOM 셀도 기록해야 하므로).

백분위·tps 계산은 metrics.py 몫. 여기서는 원시 시각만 남긴다.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx

_CONNECT_TIMEOUT_S = 10
# 긴 컨텍스트 + 높은 동시성에서는 첫 토큰까지 수 분이 걸릴 수 있다
_DEFAULT_TIMEOUT_S = 600
_ERROR_BODY_MAX = 500
# 스트림 델타에서 출력으로 치는 필드. reasoning 계열은 gpt-oss 등 추론 모델용
_CONTENT_KEYS = ("content", "reasoning_content", "reasoning")


@dataclass
class RequestRecord:
    index: int
    warmup: bool
    start_s: float  # 부하 시작 기준 요청 송신 시각, 단위 초
    end_s: float  # 부하 시작 기준 응답 종료 시각, 단위 초
    status: int | None  # HTTP 상태. 연결 실패 등으로 응답이 없으면 None
    error: str | None
    chunk_times_s: list[float] = field(default_factory=list)  # 요청 시작 기준 내용 청크 도착 시각
    completion_tokens: int | None = None  # 엔진 보고 usage. 없으면 None
    prompt_tokens: int | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def ttft_s(self) -> float | None:
        return self.chunk_times_s[0] if self.chunk_times_s else None

    @property
    def itl_s(self) -> list[float]:
        """연속한 내용 청크 사이 간격, 단위 초."""
        t = self.chunk_times_s
        return [b - a for a, b in itertools.pairwise(t)]

    @property
    def e2e_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class LoadRun:
    records: list[RequestRecord]  # 측정 구간 (워밍업 제외)
    warmup_records: list[RequestRecord]
    measure_start_s: float  # 부하 시작 기준
    measure_end_s: float

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
    usage: dict | None = None
    status: int | None = None
    error: str | None = None
    try:
        async with client.stream("POST", url, json=body) as resp:
            status = resp.status_code
            if status != 200:
                text = (await resp.aread()).decode(errors="replace")
                error = f"HTTP {status}: {text[:_ERROR_BODY_MAX]}"
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
                        error = f"bad SSE payload: {data[:_ERROR_BODY_MAX]}"
                        break
                    if chunk.get("error"):
                        error = f"stream error: {json.dumps(chunk['error'], ensure_ascii=False)[:_ERROR_BODY_MAX]}"
                        break
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if any(delta.get(k) for k in _CONTENT_KEYS):
                            chunk_times.append(now - start)
                            break
    except httpx.HTTPError as e:
        error = f"{type(e).__name__}: {e}"[:_ERROR_BODY_MAX]
    end = time.perf_counter()

    if error is None and not chunk_times:
        error = "no output tokens in stream"
    return RequestRecord(
        index=index,
        warmup=warmup,
        start_s=start - t0,
        end_s=end - t0,
        status=status,
        error=error,
        chunk_times_s=chunk_times,
        completion_tokens=(usage or {}).get("completion_tokens"),
        prompt_tokens=(usage or {}).get("prompt_tokens"),
    )


async def _phase(
    client: httpx.AsyncClient,
    url: str,
    bodies: list[dict],
    concurrency: int,
    warmup: bool,
    t0: float,
) -> list[RequestRecord]:
    queue = iter(enumerate(bodies))
    out: list[RequestRecord] = []

    async def worker() -> None:
        # 단일 이벤트 루프라 next() 는 원자적
        for i, body in queue:
            out.append(await _one_request(client, url, body, i, warmup, t0))

    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(bodies)))))
    out.sort(key=lambda r: r.index)
    return out


async def run_load(
    base_url: str,
    model: str,
    prompts: Sequence[str],
    *,
    concurrency: int,
    request_count: int,
    max_tokens: int,
    warmup_count: int = 0,
    extra_body: dict | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    headers: dict[str, str] | None = None,
) -> LoadRun:
    """`prompts` 를 순환하며 워밍업 `warmup_count` 개 → 측정 `request_count` 개를 보낸다.

    base_url 은 `/v1` 까지 포함. 예: http://localhost:11434/v1
    extra_body 는 엔진별 옵션 (예: vLLM 의 {"ignore_eos": True, "min_tokens": N}).
    """
    if concurrency < 1 or request_count < 1 or warmup_count < 0:
        raise ValueError("concurrency>=1, request_count>=1, warmup_count>=0")
    if not prompts:
        raise ValueError("prompts 가 비어 있음")

    def body(i: int) -> dict:
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompts[i % len(prompts)]}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            **(extra_body or {}),
        }

    url = _chat_url(base_url)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(timeout_s, connect=_CONNECT_TIMEOUT_S)
    async with httpx.AsyncClient(limits=limits, timeout=timeout, headers=headers) as client:
        t0 = time.perf_counter()
        warm = await _phase(
            client, url, [body(i) for i in range(warmup_count)], concurrency, True, t0
        )
        measure_start = time.perf_counter() - t0
        measured = await _phase(
            client,
            url,
            [body(warmup_count + i) for i in range(request_count)],
            concurrency,
            False,
            t0,
        )
        measure_end = time.perf_counter() - t0
    return LoadRun(measured, warm, measure_start, measure_end)
