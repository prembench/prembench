import socket

import pytest

from mock_openai import MockConfig, MockServer
from prembench.load import NotReadyError, request_body, run_load
from prembench.probe import parse_temps
from prembench.schema import Sampling


def _sampling(max_tokens: int = 8, stop: list[str] | None = None) -> Sampling:
    return Sampling(temperature=0.0, top_p=1.0, repetition_penalty=1.0, max_tokens=max_tokens, stop=stop or [])


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
    return await run_load(server.base_url, "mock", prompts, **kw)


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
    assert "stop" not in request_body("m", "p", _sampling())


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
            f"http://127.0.0.1:{port}/v1", "mock", ["x"], sampling=_sampling(),
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
