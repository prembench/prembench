import socket

import pytest

from mock_openai import MockConfig, MockServer
from prembench.load import NotReadyError, request_body, run_load
from prembench.probe import parse_temps
from prembench.schema import Sampling


def _sampling(max_tokens: int = 8, stop: list[str] | None = None, seed: int | None = None) -> Sampling:
    return Sampling(
        temperature=0.0, top_p=1.0, repetition_penalty=1.0, max_tokens=max_tokens, stop=stop or [], seed=seed
    )


def _no_gpu() -> None:
    return None


@pytest.fixture
def server(request):
    cfg = getattr(request, "param", None) or MockConfig()
    with MockServer(cfg) as s:
        yield s


async def _run(server, **kw):
    kw.setdefault("sampling", _sampling())
    kw.setdefault("concurrency", 1)
    kw.setdefault("request_count", 1)
    kw.setdefault("gpu_temp", _no_gpu)
    prompts = kw.pop("prompts", ["hello"])
    return await run_load(server.base_url, "mock", lambda i: prompts[i % len(prompts)], **kw)


async def test_single_request_timing(server):
    run = await _run(server)
    (rec,) = run.records
    assert rec.ok and rec.status == 200 and rec.error is None
    # role 만 있는 첫 청크는 토큰이 아니다 → TTFT 는 첫 출력 청크 기준
    assert 0.05 <= rec.ttft_s < 0.5
    assert len(rec.chunk_times_s) == 8
    assert rec.chunk_times_s[0] == rec.ttft_s
    assert len(rec.itl_s) == 7
    assert all(0.008 <= x < 0.2 for x in rec.itl_s)
    assert rec.completion_tokens == 8
    assert rec.reasoning_tokens is None
    assert rec.e2e_s >= rec.chunk_times_s[-1]
    assert rec.finish_reason == "stop"
    assert not rec.content_empty


async def test_request_body_carries_sampling(server):
    await _run(server, sampling=_sampling(max_tokens=6, stop=["\n\n"]))
    body = server.state.bodies[-1]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert (body["temperature"], body["top_p"], body["repetition_penalty"]) == (0.0, 1.0, 1.0)
    assert body["max_tokens"] == 6
    assert body["stop"] == ["\n\n"]


def test_empty_stop_not_sent():
    body = request_body("m", "p", _sampling())
    assert "stop" not in body and "seed" not in body and "ignore_eos" not in body


async def test_seed_and_ignore_eos_sent(server):
    run = await _run(server, sampling=_sampling(seed=7), ignore_eos=True)
    body = server.state.bodies[-1]
    assert body["seed"] == 7 and body["ignore_eos"] is True
    assert run.ignore_eos is True


async def test_per_request_conversion(server):
    run = await _run(server, warmup_count=1, request_count=3, sampling=_sampling(8))
    pr = run.per_request()
    assert len(pr) == 3
    first = pr[0]
    assert first.status == 200 and first.error is None
    assert first.prompt_tokens == 5 and first.output_tokens == 8
    assert 0 <= first.start_s < run.duration_s
    assert first.e2e_ms >= first.ttft_ms >= 50
    # tpot = (e2e - ttft) / (n - 1)
    assert abs(first.tpot_ms - (first.e2e_ms - first.ttft_ms) / 7) < 1e-9
    assert first.itl_max_ms >= 8
    assert first.finish_reason == "stop" and first.content_empty is False
    assert [p.start_s for p in pr] == sorted(p.start_s for p in pr)


@pytest.mark.parametrize("server", [MockConfig(fail_requests=frozenset({2}))], indirect=True)
async def test_per_request_failed(server):
    run = await _run(server, request_count=2)
    failed = run.per_request()[0]
    assert failed.status == 500 and failed.finish_reason == "error"
    assert failed.ttft_ms is None and failed.tpot_ms is None and failed.itl_max_ms is None
    assert failed.content_empty is False


async def test_error_truncated_to_200_chars():
    cfg = MockConfig(fail_requests=frozenset({2}), error_message="x" * 1000)
    with MockServer(cfg) as srv:
        run = await _run(srv, request_count=1)
    err = run.records[0].error
    assert err.startswith("HTTP 500") and len(err) == 200
    run.per_request()  # 스키마 max_length=200 통과


async def test_min_duration_extends_beyond_request_count(server):
    # 요청 1개 ≈ 0.05 + 7×0.01 = 0.12s. 0.6초 이상 측정하려면 request_count=2 보다 많아야 한다
    run = await _run(server, prompts=[f"p{i}" for i in range(100)], concurrency=2, request_count=2,
                     warmup_count=2, min_duration_s=0.6)
    assert run.duration_s >= 0.6
    assert len(run.records) > 2
    assert not run.pool_exhausted


async def test_prompts_are_not_reused_and_pool_exhaustion_is_flagged(server):
    seen = []

    def prompt(i):
        seen.append(i)
        return f"p{i}"

    # max_requests 로 풀을 3개로 묶으면 1초를 못 채우고 멈춘다
    run = await run_load(server.base_url, "mock", prompt, sampling=_sampling(), concurrency=1, request_count=2,
                         warmup_count=1, min_duration_s=1.0, max_requests=3, gpu_temp=_no_gpu)
    assert len(run.records) == 3 and run.pool_exhausted
    contents = [b["messages"][0]["content"] for b in server.state.bodies]
    assert len(contents) == len(set(contents))  # 준비·워밍업·측정 모두 다른 프롬프트
    assert seen[0] == -1  # 준비 확인용


async def test_prompts_generated_before_measurement(server):
    # 측정 중에 prompt() 가 불리면 토큰 시각이 밀린다 → 측정 시작 전에 전부 만들어져 있어야 한다
    calls = []
    run = None

    def prompt(i):
        calls.append(len(server.state.bodies))
        return f"p{i}"

    run = await run_load(server.base_url, "mock", prompt, sampling=_sampling(), concurrency=2, request_count=4,
                         warmup_count=2, gpu_temp=_no_gpu)
    warm_and_ready = 1 + 2
    assert max(calls) <= warm_and_ready  # 측정 요청이 나가기 시작한 뒤엔 호출 없음
    assert len(run.records) == 4


async def test_readiness_is_a_real_generation_before_warmup(server):
    run = await _run(server, warmup_count=2)
    first = server.state.bodies[0]
    assert first["max_tokens"] == 4  # 짧은 실제 생성 요청
    assert run.ready.ok and run.ready.chunk_times_s
    assert server.state.requests == 1 + 2 + 1


@pytest.mark.parametrize("server", [MockConfig(fail_requests=frozenset({1}))], indirect=True)
async def test_readiness_failure_stops_measurement(server):
    with pytest.raises(NotReadyError, match="HTTP 500"):
        await _run(server, warmup_count=2, request_count=4)
    assert server.state.requests == 1  # 워밍업·측정 요청은 보내지 않는다


async def test_readiness_failure_on_connection_refused():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(NotReadyError):
        await run_load(
            f"http://127.0.0.1:{port}/v1", "mock", lambda i: "x", sampling=_sampling(),
            concurrency=2, request_count=2, gpu_temp=_no_gpu,
        )


async def test_concurrency_is_respected(server):
    run = await _run(server, prompts=["a", "b"], concurrency=4, request_count=12)
    assert len(run.records) == 12
    assert all(r.ok for r in run.records)
    assert server.state.max_in_flight == 4
    assert run.duration_s > 0
    # 프롬프트는 순환해서 쓴다 (준비 확인 요청 다음부터)
    assert [b["messages"][0]["content"] for b in server.state.bodies[1:5]] == ["a", "b", "a", "b"]


async def test_warmup_excluded(server):
    run = await _run(server, prompts=["x"], concurrency=2, request_count=6, warmup_count=3,
                     sampling=_sampling(4))
    assert server.state.requests == 1 + 3 + 6
    assert len(run.records) == 6
    assert len(run.warmup_records) == 3
    # 측정 구간은 워밍업이 끝난 뒤 시작
    assert min(r.start_s for r in run.records) >= max(r.end_s for r in run.warmup_records)


# 요청 번호 1 = 준비 확인 → 측정 요청 2..7 중 3, 6 번째 실패
@pytest.mark.parametrize("server", [MockConfig(fail_requests=frozenset({3, 6}))], indirect=True)
async def test_http_errors_are_recorded_not_raised(server):
    run = await _run(server, request_count=6, sampling=_sampling(4))
    failed = [r for r in run.records if not r.ok]
    assert len(failed) == 2
    assert all(r.status == 500 and "mock failure" in r.error for r in failed)
    assert all(r.ttft_s is None and r.chunk_times_s == [] for r in failed)
    assert all(r.finish_reason == "error" and not r.content_empty for r in failed)
    assert run.error_count == 2


@pytest.mark.parametrize("server", [MockConfig(send_usage=False)], indirect=True)
async def test_missing_usage_stays_none(server):
    run = await _run(server)
    # 엔진이 usage 를 안 주면 청크 수로 추정하지 않는다
    assert run.records[0].completion_tokens is None
    assert len(run.records[0].chunk_times_s) == 8


async def test_finish_reason_length(server):
    run = await _run(server, sampling=_sampling(3))
    assert run.records[0].finish_reason == "length"
    assert run.records[0].completion_tokens == 3


@pytest.mark.parametrize("server", [MockConfig(reasoning_tokens=5, tokens=3)], indirect=True)
async def test_reasoning_tokens_split(server):
    run = await _run(server, sampling=_sampling(16))
    rec = run.records[0]
    # TTFT 는 추론 토큰 포함 첫 출력 기준
    assert len(rec.chunk_times_s) == 8 and rec.content_chunks == 3
    assert rec.reasoning_tokens == 5 and rec.completion_tokens == 8
    assert not rec.content_empty


@pytest.mark.parametrize("server", [MockConfig(reasoning_tokens=10, tokens=3)], indirect=True)
async def test_content_empty_when_reasoning_eats_budget(server):
    run = await _run(server, sampling=_sampling(8))
    rec = run.records[0]
    assert rec.ok and rec.content_empty
    assert rec.finish_reason == "length"


async def test_max_tokens_and_extra_body(server):
    await _run(server, sampling=_sampling(3), extra_body={"ignore_eos": True})
    assert server.state.bodies[-1]["ignore_eos"] is True


async def test_gpu_temperature_recorded_at_measure_bounds(server):
    temps = iter([61.0, 67.0])
    run = await _run(server, warmup_count=1, gpu_temp=lambda: next(temps))
    assert (run.gpu_temp_start_c, run.gpu_temp_end_c) == (61.0, 67.0)


async def test_gpu_temperature_none_without_gpu(server):
    run = await _run(server)
    assert run.gpu_temp_start_c is None and run.gpu_temp_end_c is None


def test_parse_temps():
    assert parse_temps("54\n61\n") == 61.0
    assert parse_temps("[N/A]\n") is None
