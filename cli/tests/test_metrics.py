import math

import pytest

from prembench.load import LoadRun, RequestRecord
from prembench.metrics import build_result, gpu_stats, percentile
from prembench.probe import GpuSample, GpuSampler, parse_samples


def _ok(i: int, start: float, chunks: list[float], tokens: int | None, **kw) -> RequestRecord:
    """성공 요청. chunks 는 요청 시작 기준 초, 종료 = 마지막 청크 + 0.001."""
    return RequestRecord(
        index=i, warmup=False, start_s=start, end_s=start + chunks[-1] + 0.001, status=200, error=None,
        chunk_times_s=chunks, content_chunks=len(chunks), finish_reason_raw=kw.pop("finish", "length"),
        completion_tokens=tokens, prompt_tokens=kw.pop("prompt_tokens", 100), **kw,
    )


def _fail(i: int, start: float, error: str = "HTTP 500: boom", status: int | None = 500) -> RequestRecord:
    return RequestRecord(index=i, warmup=False, start_s=start, end_s=start + 0.01, status=status, error=error)


def _run(records, start=1.0, end=3.0, samples=None) -> LoadRun:
    ready = _ok(-1, 0.0, [0.01], 1)
    return LoadRun(ready, records, [], start, end, None, None, False, samples)


# ---------------------------------------------------------------- percentile


def test_nearest_rank_known_values():
    xs = [15, 20, 35, 40, 50]  # 위키백과 nearest-rank 예제
    assert percentile(xs, 5) == 15
    assert percentile(xs, 30) == 20
    assert percentile(xs, 40) == 20
    assert percentile(xs, 50) == 35
    assert percentile(xs, 100) == 50


def test_nearest_rank_returns_observed_values_only():
    xs = [1.0, 2.0, 3.0, 4.0]
    for p in range(1, 101):
        assert percentile(xs, p) in xs  # 보간 없음


def test_nearest_rank_exact_boundary():
    # 95% × 20 = 19 → 19번째 (부동소수 오차로 20번째가 되면 안 됨)
    xs = list(range(1, 21))
    assert percentile(xs, 95) == 19
    assert percentile(xs, 99) == 20
    assert percentile(list(range(1, 101)), 99) == 99


def test_percentile_edge():
    assert percentile([], 50) is None
    assert percentile([7.0], 99) == 7.0
    with pytest.raises(ValueError):
        percentile([1.0], 0)


# ---------------------------------------------------------------- build_result


def test_zero_requests_raises():
    with pytest.raises(ValueError, match="0건"):
        build_result(_run([]), concurrency=1)


def test_basic_result():
    recs = [
        _ok(0, 1.0, [0.100, 0.110, 0.130], 3),
        _ok(1, 1.5, [0.200, 0.205, 0.215], 3),
    ]
    r = build_result(_run(recs, start=1.0, end=3.0), concurrency=2)
    assert r.ttft_p50_ms == pytest.approx(100.0)
    assert r.ttft_p99_ms == pytest.approx(200.0)
    # ITL 는 성공 요청 간격 전체 풀: [10, 20, 5, 10] ms
    assert r.itl_p50_ms == pytest.approx(10.0)
    assert r.itl_p99_ms == pytest.approx(20.0)
    assert r.duration_s == pytest.approx(2.0)
    assert r.output_tps_total == pytest.approx(6 / 2.0)
    assert r.output_tps_per_user == pytest.approx(r.output_tps_total / 2)
    assert r.requests_per_sec == pytest.approx(1.0)
    assert r.output_tokens_actual_mean == 3 and r.token_count_source == "engine_usage"
    assert r.error_count == 0 and r.oom is False
    assert r.finish_reason_counts.length == 2
    assert r.content_empty_rate == 0
    assert r.prefix_cache_hit_rate is None  # 엔진이 cached_tokens 를 안 줌
    assert r.gpu_mem_peak_gb is None and r.energy_total_j is None
    # per_request start_s 는 측정 시작 기준
    assert [p.start_s for p in r.per_request] == pytest.approx([0.0, 0.5])


def test_all_failed():
    recs = [_fail(0, 1.0), _fail(1, 1.1, error="ConnectError: refused", status=None)]
    r = build_result(_run(recs), concurrency=2)
    assert r.error_count == 2 and r.finish_reason_counts.error == 2
    for name in ("ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "e2e_p50_ms", "itl_p99_ms",
                 "output_tokens_actual_mean", "token_count_source", "prefix_cache_hit_rate"):
        assert getattr(r, name) is None, name
    # 완료 0건은 "모름"이 아니라 실측된 0
    assert r.output_tps_total == 0 and r.requests_per_sec == 0
    assert r.content_empty_rate == 0


def test_failures_excluded_from_latency():
    recs = [_ok(0, 1.0, [0.1, 0.2], 2), _fail(1, 1.0)]
    r = build_result(_run(recs), concurrency=2)
    assert r.e2e_p99_ms == pytest.approx(201.0)  # 실패 요청 10ms 는 안 섞인다
    assert r.error_count == 1 and r.requests_per_sec == pytest.approx(0.5)


def test_missing_usage_disables_token_aggregates():
    recs = [_ok(0, 1.0, [0.1, 0.2], 2), _ok(1, 1.0, [0.1, 0.2], None)]
    r = build_result(_run(recs), concurrency=2)
    assert r.output_tps_total is None and r.output_tokens_actual_mean is None
    assert r.energy_per_output_token_j is None
    assert r.token_count_source == "engine_usage"  # 1건은 엔진이 보고했음


def test_reasoning_split():
    recs = [_ok(0, 1.0, [0.1, 0.2, 0.3], 10, reasoning_tokens=6), _ok(1, 1.0, [0.1, 0.2], 8, reasoning_tokens=2)]
    r = build_result(_run(recs), concurrency=1)
    assert r.reasoning_tokens_mean == 4 and r.content_tokens_mean == 5


def test_reasoning_partial_is_none():
    recs = [_ok(0, 1.0, [0.1, 0.2], 10, reasoning_tokens=6), _ok(1, 1.0, [0.1, 0.2], 8)]
    r = build_result(_run(recs), concurrency=1)
    assert r.reasoning_tokens_mean is None and r.content_tokens_mean is None


def test_content_empty_rate():
    empty = _ok(1, 1.0, [0.1], 5)
    empty.content_chunks = 0  # 추론만 하고 끝남
    r = build_result(_run([_ok(0, 1.0, [0.1, 0.2], 2), empty, _fail(2, 1.0)]), concurrency=1)
    assert r.content_empty_rate == pytest.approx(1 / 3)


def test_prefix_cache_hit_rate():
    recs = [_ok(0, 1.0, [0.1], 1, prompt_tokens=100, cached_tokens=10),
            _ok(1, 1.0, [0.1], 1, prompt_tokens=100, cached_tokens=0)]
    assert build_result(_run(recs), 1).prefix_cache_hit_rate == pytest.approx(0.05)


def test_oom_detected_from_error_text():
    recs = [_ok(0, 1.0, [0.1], 1), _fail(1, 1.0, error="HTTP 500: CUDA out of memory. Tried to allocate")]
    assert build_result(_run(recs), 1).oom is True
    assert build_result(_run([_fail(0, 1.0, error="HTTP 500: room booked")]), 1).oom is False


def test_finish_reason_other():
    recs = [_ok(0, 1.0, [0.1], 1, finish="tool_calls"), _ok(1, 1.0, [0.1], 1, finish="stop")]
    c = build_result(_run(recs), 1).finish_reason_counts
    assert (c.stop, c.length, c.other) == (1, 0, 1)


# ---------------------------------------------------------------- GPU


def test_gpu_stats_multi_gpu():
    s = [
        GpuSample(0.0, 0, 10.0, 50.0, 100.0), GpuSample(0.0, 1, 20.0, 70.0, 200.0),
        GpuSample(1.0, 0, 12.0, 60.0, 120.0), GpuSample(1.0, 1, 18.0, 80.0, 180.0),
    ]
    mem, util, power = gpu_stats(s)
    assert mem == 20.0  # 장당 최대
    assert util == pytest.approx(65.0)  # 전 GPU 평균
    assert power == pytest.approx(300.0)  # 시점마다 합 (300, 300) 의 평균


def test_gpu_stats_skips_tick_with_missing_power():
    s = [GpuSample(0.0, 0, 1.0, 1.0, 100.0), GpuSample(0.0, 1, 1.0, 1.0, None),
         GpuSample(1.0, 0, 1.0, 1.0, 110.0), GpuSample(1.0, 1, 1.0, 1.0, 90.0)]
    assert gpu_stats(s)[2] == pytest.approx(200.0)


def test_gpu_energy_in_result():
    s = [GpuSample(0.0, 0, 30.0, 90.0, 300.0), GpuSample(1.0, 0, 31.0, 95.0, 310.0)]
    r = build_result(_run([_ok(0, 1.0, [0.1, 0.2], 4)], start=1.0, end=3.0, samples=s), 1)
    assert r.gpu_mem_peak_gb == 31.0 and r.power_avg_w == pytest.approx(305.0)
    assert r.energy_total_j == pytest.approx(305.0 * 2.0)
    assert r.energy_per_output_token_j == pytest.approx(305.0 * 2.0 / 4)


def test_parse_samples():
    out = parse_samples("0, 40960, 87, 301.52\n1, 1024, [N/A], [N/A]\ngarbage\n", t_s=5.0)
    assert [(x.gpu, x.mem_used_gb, x.util_pct, x.power_w) for x in out] == [
        (0, 40.0, 87.0, 301.52), (1, 1.0, None, None)
    ]


def test_sampler_collects_until_stop():
    calls = []

    def read():
        calls.append(1)
        return "0, 1024, 50, 100\n"

    sampler = GpuSampler(read, interval_s=0.01)
    sampler.start()
    import time

    time.sleep(0.05)
    samples = sampler.stop()
    assert len(samples) >= 2 and len(samples) == len(calls)
    assert all(math.isclose(x.mem_used_gb, 1.0) for x in samples)
