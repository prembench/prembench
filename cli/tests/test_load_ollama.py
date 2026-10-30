"""실제 Ollama 대상 end-to-end. PREMBENCH_OLLAMA_URL 이 없으면 skip.

    PREMBENCH_OLLAMA_URL=http://localhost:11434/v1 pytest -q tests/test_load_ollama.py
"""

import os

import pytest

from prembench.load import run_load

URL = os.environ.get("PREMBENCH_OLLAMA_URL")
MODEL = os.environ.get("PREMBENCH_OLLAMA_MODEL", "qwen2.5:1.5b")

pytestmark = pytest.mark.skipif(not URL, reason="PREMBENCH_OLLAMA_URL 미설정")


async def test_ollama_streaming_one_cell():
    run = await run_load(
        URL, MODEL, ["Count from 1 to 100 separated by spaces."],
        concurrency=2, request_count=4, warmup_count=1, max_tokens=32,
    )
    assert run.error_count == 0, [r.error for r in run.records]
    for r in run.records:
        assert r.ttft_s is not None and r.ttft_s > 0
        assert len(r.chunk_times_s) >= 2
        assert all(x >= 0 for x in r.itl_s)
        # Ollama 는 stream_options.include_usage 를 지원한다
        assert r.completion_tokens is not None and 0 < r.completion_tokens <= 32
