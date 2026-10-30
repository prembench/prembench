import socket

import pytest

from prembench.load import run_load
from mock_openai import MockConfig, MockServer


@pytest.fixture
def server(request):
    cfg = getattr(request, "param", None) or MockConfig()
    with MockServer(cfg) as s:
        yield s


async def test_single_request_timing(server):
    run = await run_load(server.base_url, "mock", ["hello"], concurrency=1, request_count=1, max_tokens=8)
    (rec,) = run.records
    assert rec.ok and rec.status == 200 and rec.error is None
    # role 만 있는 첫 청크는 토큰이 아니다 → TTFT 는 첫 내용 청크 기준
    assert 0.05 <= rec.ttft_s < 0.5
    assert len(rec.chunk_times_s) == 8
    assert rec.chunk_times_s[0] == rec.ttft_s
    assert len(rec.itl_s) == 7
    assert all(0.008 <= x < 0.2 for x in rec.itl_s)
    assert rec.completion_tokens == 8
    assert rec.e2e_s >= rec.chunk_times_s[-1]

    body = server.state.bodies[0]
    assert body["stream"] is True
    assert body["max_tokens"] == 8
    assert body["stream_options"] == {"include_usage": True}
    assert body["messages"] == [{"role": "user", "content": "hello"}]


async def test_concurrency_is_respected(server):
    run = await run_load(server.base_url, "mock", ["a", "b"], concurrency=4, request_count=12, max_tokens=8)
    assert len(run.records) == 12
    assert all(r.ok for r in run.records)
    assert server.state.max_in_flight == 4
    assert run.duration_s > 0
    # 프롬프트는 순환해서 쓴다
    assert [b["messages"][0]["content"] for b in server.state.bodies[:4]] == ["a", "b", "a", "b"]


async def test_warmup_excluded(server):
    run = await run_load(
        server.base_url, "mock", ["x"], concurrency=2, request_count=6, warmup_count=3, max_tokens=4
    )
    assert server.state.requests == 9
    assert len(run.records) == 6
    assert len(run.warmup_records) == 3
    # 측정 구간은 워밍업이 끝난 뒤 시작
    assert min(r.start_s for r in run.records) >= max(r.end_s for r in run.warmup_records)


@pytest.mark.parametrize("server", [MockConfig(fail_every=3)], indirect=True)
async def test_http_errors_are_recorded_not_raised(server):
    run = await run_load(server.base_url, "mock", ["x"], concurrency=1, request_count=6, max_tokens=4)
    failed = [r for r in run.records if not r.ok]
    assert len(failed) == 2
    assert all(r.status == 500 and "mock failure" in r.error for r in failed)
    assert all(r.ttft_s is None and r.chunk_times_s == [] for r in failed)
    assert run.error_count == 2


@pytest.mark.parametrize("server", [MockConfig(send_usage=False)], indirect=True)
async def test_missing_usage_stays_none(server):
    run = await run_load(server.base_url, "mock", ["x"], concurrency=1, request_count=1, max_tokens=8)
    # 엔진이 usage 를 안 주면 청크 수로 추정하지 않는다
    assert run.records[0].completion_tokens is None
    assert len(run.records[0].chunk_times_s) == 8


async def test_max_tokens_and_extra_body(server):
    await run_load(
        server.base_url, "mock", ["x"], concurrency=1, request_count=1, max_tokens=3,
        extra_body={"ignore_eos": True},
    )
    assert server.state.bodies[0]["ignore_eos"] is True


async def test_connection_refused_is_recorded():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    run = await run_load(f"http://127.0.0.1:{port}/v1", "mock", ["x"], concurrency=2, request_count=2, max_tokens=4)
    assert run.error_count == 2
    assert all(r.status is None and r.error for r in run.records)
