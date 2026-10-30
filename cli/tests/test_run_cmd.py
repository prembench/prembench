"""`prembench run` 끝까지: mock Ollama + 로컬 BPE 토크나이저 (네트워크 없음)."""

import json
import warnings

import pytest
from typer.testing import CliRunner

from conftest import HF_ID, MODEL, OVERHEAD, mock_ollama
from mock_openai import MockConfig, MockServer
from prembench.__main__ import app
from prembench.cell import is_local_url
from prembench.prompts import HFTokenizer
from prembench.schema import Run

runner = CliRunner()


def _invoke(srv, *extra):
    args = ["run", "--url", srv.base_url, "--model", MODEL, "--concurrency", "2", "--in", "64", "--out", "8",
            "--min-duration", "0", *extra]
    return runner.invoke(app, args)


def test_run_emits_valid_record(env):
    out = env / "rec.json"
    with mock_ollama() as srv:
        res = _invoke(srv, "--out-file", str(out), "--seed", "3")
    assert res.exit_code == 0, res.output
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run = Run.model_validate_json(out.read_text())
    assert run.engine.name == "ollama" and run.engine.max_model_len == 4096
    assert run.model.hf_id == HF_ID and run.model.quantization == "gguf-q4_k_m"
    assert run.model.params_b == pytest.approx(1.5) and run.model.moe is False
    assert run.load.request_count == 10 and run.load.warmup_count == 2  # 기본: ×5, =concurrency
    assert run.load.random_seed == 3 and run.load.sampling.seed == 3
    assert run.load.ignore_eos is None  # Ollama 는 적용 안 함
    assert run.conditions.model_load_state == "cold"  # mock 은 첫 요청 전엔 미적재
    assert run.conditions.chat_template_variant.startswith("sha256:")
    assert all(p.prompt_tokens == 64 for p in run.result.per_request)
    assert run.result.error_count == 0 and run.result.ttft_p50_ms is not None


def test_ollama_ignore_eos_is_null_and_not_sent(env):
    with mock_ollama() as srv:
        res = _invoke(srv, "--ignore-eos")
        bodies = srv.state.bodies
    assert res.exit_code == 0, res.output
    record = json.loads(res.stdout[res.stdout.index("{"):])
    assert record["load"]["ignore_eos"] is None
    assert not any("ignore_eos" in b for b in bodies)


def test_missing_identity_errors_without_prompt(env, monkeypatch):
    (env / "config").write_text("", encoding="utf-8")
    with mock_ollama() as srv:
        res = _invoke(srv)
        assert srv.state.requests == 0  # 아무 요청도 보내지 않는다
    assert res.exit_code == 2
    for name in ("submitter", "provenance", "environment", "hf_id"):
        assert name in res.output


def test_cloud_rented_requires_provider(env):
    with mock_ollama() as srv:
        res = _invoke(srv, "--environment", "cloud_rented")
    assert res.exit_code == 2 and "provider" in res.output


def test_invalid_choice(env):
    with mock_ollama() as srv:
        res = _invoke(srv, "--provenance", "friend")
    assert res.exit_code == 2 and "provenance" in res.output


def test_prompt_token_mismatch_aborts(env):
    # 엔진이 긴 프롬프트를 잘라 버리는 상황 (Ollama num_ctx 초과 등)
    tok = HFTokenizer.from_file(mock_ollama.bpe_path)
    cfg = MockConfig(models=(MODEL,), first_token_delay_s=0.01, inter_token_delay_s=0.001, tokens=4,
                     prompt_tokens=lambda s: min(OVERHEAD + len(tok.encode(s)), 120))  # 보정(≤93)은 통과
    with MockServer(cfg) as srv:
        res = runner.invoke(app, ["run", "--url", srv.base_url, "--model", MODEL, "--concurrency", "1",
                                  "--in", "200", "--out", "4", "--min-duration", "0"])
    assert res.exit_code == 1
    assert "프롬프트 토큰" in res.output and "{" not in res.stdout  # 레코드는 내지 않는다


def test_unknown_model(env):
    with mock_ollama() as srv:
        res = runner.invoke(app, ["run", "--url", srv.base_url, "--model", "nope", "--hf-id", HF_ID,
                                  "--concurrency", "1", "--in", "64", "--out", "4", "--min-duration", "0"])
    assert res.exit_code == 1 and "nope" in res.output


def test_remote_url_rejected(env):
    res = runner.invoke(app, ["run", "--url", "http://10.255.255.1:8000/v1", "--model", MODEL,
                              "--concurrency", "1", "--in", "64", "--out", "4", "--min-duration", "0"])
    assert res.exit_code == 1 and "원격" in res.output


def test_is_local_url():
    assert is_local_url("http://localhost:11434/v1")
    assert is_local_url("http://127.0.0.1:8000/v1")
    assert is_local_url("http://[::1]:8000/v1")
    assert not is_local_url("http://10.255.255.1:8000/v1")
    assert not is_local_url("http://no-such-host.invalid/v1")
