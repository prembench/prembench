"""실제 Ollama 대상 end-to-end. PREMBENCH_OLLAMA_URL 이 없으면 skip.

    PREMBENCH_OLLAMA_URL=http://localhost:11434/v1 pytest -q tests/test_load_ollama.py
"""

import os

import pytest

from prembench.load import run_load
from prembench.schema import Sampling

URL = os.environ.get("PREMBENCH_OLLAMA_URL")
MODEL = os.environ.get("PREMBENCH_OLLAMA_MODEL", "qwen2.5:1.5b")

pytestmark = pytest.mark.skipif(not URL, reason="PREMBENCH_OLLAMA_URL 미설정")


async def test_ollama_streaming_one_cell():
    run = await run_load(
        URL, MODEL, ["Count from 1 to 100 separated by spaces."],
        sampling=Sampling(temperature=0.0, top_p=1.0, repetition_penalty=1.0, max_tokens=32, stop=[], seed=None),
        concurrency=2, request_count=4, warmup_count=1,
    )
    assert run.ready.ok
    assert run.error_count == 0, [r.error for r in run.records]
    for r in run.records:
        assert r.ttft_s is not None and r.ttft_s > 0
        assert len(r.chunk_times_s) >= 2
        assert all(x >= 0 for x in r.itl_s)
        # Ollama 는 stream_options.include_usage 를 지원한다
        assert r.completion_tokens is not None and 0 < r.completion_tokens <= 32
        assert r.finish_reason in ("stop", "length")
    # 맥북엔 nvidia-smi 가 없으므로 null
    assert run.gpu_temp_start_c is None
    # 실제 엔진 응답이 스키마 PerRequest 를 통과해야 한다
    pr = run.per_request()
    assert all(p.prompt_tokens and p.e2e_ms >= p.ttft_ms for p in pr)


async def test_ollama_exact_input_tokens():
    """HF 토크나이저(첫 실행 시 다운로드) + 템플릿 오버헤드 실측으로 usage.prompt_tokens 가 정확히 맞는지."""
    from prembench.prompts import build_prompts, content_tokens_for, load_hf_tokenizer, measure_template_overhead, token_pools

    tok = load_hf_tokenizer(os.environ.get("PREMBENCH_OLLAMA_HF_ID", "Qwen/Qwen2.5-1.5B-Instruct"))
    pools = token_pools(tok)
    overhead = await measure_template_overhead(URL, MODEL, tok, seed=0, pools=pools)
    for target, ko in [(512, 0.0), (512, 0.5)]:
        prompts = build_prompts(tok, 2, content_tokens_for(target, overhead), seed=0, ko_ratio=ko, pools=pools)
        run = await run_load(
            URL, MODEL, prompts,
            sampling=Sampling(temperature=0.0, top_p=1.0, repetition_penalty=1.0, max_tokens=2, stop=[], seed=None),
            concurrency=1, request_count=2,
        )
        assert [r.prompt_tokens for r in run.records] == [target, target]
