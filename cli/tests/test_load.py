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
