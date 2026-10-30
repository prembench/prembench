"""LoadRun(+ GPU 샘플) → schema.Result. 보간 없음.

- 백분위: nearest-rank. rank = ceil(p/100 · N), 정렬된 값의 rank 번째 (1부터). 실제 관측값만 나온다.
- 지연(TTFT·TPOT·E2E·ITL)과 처리량은 성공한 요청만으로 계산한다. 실패 요청은 error_count·finish_reason 으로 남는다.
- ITL 은 성공 요청들의 청크 간격을 전부 모아(pool) 백분위를 낸다.
- 출력 토큰 수는 엔진 usage 만 쓴다. 성공 요청 중 하나라도 usage 가 없으면 토큰 기반 집계는 None.
- GPU (nvidia-smi 1초 샘플): 메모리 최대는 장당 최대, 사용률은 전 GPU 평균,
  전력은 샘플 시점마다 전 GPU 합을 낸 뒤 평균, 에너지 = 전력 평균 × 측정 시간.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import fmean

from prembench.load import LoadRun
from prembench.probe import GpuSample
from prembench.schema import FinishReasonCounts, Result

_OOM = re.compile(r"out of memory|\bOOM\b", re.IGNORECASE)


def percentile(values: list[float], p: int) -> float | None:
    """nearest-rank 백분위. p 는 1~100 정수. 값이 없으면 None."""
    if not 1 <= p <= 100:
        raise ValueError("p 는 1~100")
    if not values:
        return None
    xs = sorted(values)
    rank = max(1, -(-p * len(xs) // 100))  # ceil(p·N/100), 정수 연산
    return xs[rank - 1]


def gpu_stats(samples: list[GpuSample] | None) -> tuple[float | None, float | None, float | None]:
    """(gpu_mem_peak_gb 장당 최대, gpu_util_avg_pct 전 GPU 평균, power_avg_w 전 GPU 합의 평균)."""
    if not samples:
        return None, None, None
    mems = [s.mem_used_gb for s in samples if s.mem_used_gb is not None]
    utils = [s.util_pct for s in samples if s.util_pct is not None]
    ticks: dict[float, list[float | None]] = defaultdict(list)
    for s in samples:
        ticks[s.t_s].append(s.power_w)
    # 한 시점에 전력을 못 읽은 GPU 가 있으면 그 시점은 합을 낼 수 없으므로 뺀다
    sums = [sum(ps) for ps in ticks.values() if None not in ps]
    return (
        max(mems) if mems else None,
        fmean(utils) if utils else None,
        fmean(sums) if sums else None,
    )


def build_result(run: LoadRun, concurrency: int) -> Result:
    per_request = run.per_request()
    if not per_request:
        raise ValueError("측정 요청 0건 — 결과를 만들 수 없음")
    if concurrency < 1:
        raise ValueError("concurrency >= 1")

    ok = [p for p in per_request if p.error is None]
    ok_records = [r for r in run.records if r.ok]
    duration = run.duration_s

    ttft = [p.ttft_ms for p in ok if p.ttft_ms is not None]
    tpot = [p.tpot_ms for p in ok if p.tpot_ms is not None]
    e2e = [p.e2e_ms for p in ok if p.e2e_ms is not None]
    itl = [g * 1000 for r in ok_records for g in r.itl_s]

    tokens_known = all(p.output_tokens is not None for p in ok)
    total_out = sum(p.output_tokens for p in ok) if tokens_known else None
    tps_total = total_out / duration if total_out is not None and duration > 0 else None
    rps = len(ok) / duration if duration > 0 else None

    reasoning_known = tokens_known and bool(ok) and all(p.reasoning_tokens is not None for p in ok)

    cache_known = bool(ok_records) and all(
        r.cached_tokens is not None and r.prompt_tokens for r in ok_records
    )
    prefix_hit = (
        sum(r.cached_tokens for r in ok_records) / sum(r.prompt_tokens for r in ok_records)
        if cache_known
        else None
    )

    mem_peak, util_avg, power_avg = gpu_stats(run.gpu_samples)
    energy = power_avg * duration if power_avg is not None else None

    counts = Counter(p.finish_reason for p in per_request)
    return Result(
        ttft_p50_ms=percentile(ttft, 50),
        ttft_p95_ms=percentile(ttft, 95),
        ttft_p99_ms=percentile(ttft, 99),
        tpot_p50_ms=percentile(tpot, 50),
        tpot_p95_ms=percentile(tpot, 95),
        e2e_p50_ms=percentile(e2e, 50),
        e2e_p95_ms=percentile(e2e, 95),
        e2e_p99_ms=percentile(e2e, 99),
        itl_p50_ms=percentile(itl, 50),
        itl_p95_ms=percentile(itl, 95),
        itl_p99_ms=percentile(itl, 99),
        output_tps_total=tps_total,
        output_tps_per_user=tps_total / concurrency if tps_total is not None else None,
        requests_per_sec=rps,
        error_count=len(per_request) - len(ok),
        oom=any(p.error and _OOM.search(p.error) for p in per_request),
        gpu_mem_peak_gb=mem_peak,
        gpu_util_avg_pct=util_avg,
        power_avg_w=power_avg,
        duration_s=duration,
        energy_total_j=energy,
        energy_per_output_token_j=energy / total_out if energy is not None and total_out else None,
        output_tokens_actual_mean=fmean(p.output_tokens for p in ok) if tokens_known and ok else None,
        token_count_source="engine_usage" if any(p.output_tokens is not None for p in per_request) else None,
        reasoning_tokens_mean=fmean(p.reasoning_tokens for p in ok) if reasoning_known else None,
        content_tokens_mean=(
            fmean(p.output_tokens - p.reasoning_tokens for p in ok) if reasoning_known else None
        ),
        finish_reason_counts=FinishReasonCounts(
            stop=counts["stop"], length=counts["length"], error=counts["error"], other=counts["other"]
        ),
        content_empty_rate=sum(p.content_empty for p in per_request) / len(per_request),
        prefix_cache_hit_rate=prefix_hit,
        per_request=per_request,
    )
