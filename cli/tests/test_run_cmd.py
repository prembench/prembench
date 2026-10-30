"""`prembench run` 끝까지: mock Ollama + 로컬 BPE 토크나이저 (네트워크 없음)."""

import json
import warnings

import pytest
from tokenizers import Tokenizer as HFRaw
from tokenizers import decoders, models, pre_tokenizers, trainers
from typer.testing import CliRunner

from mock_openai import MockConfig, MockServer
from prembench import cell
from prembench.__main__ import app
from prembench.cell import is_local_url
from prembench.prompts import HFTokenizer
from prembench.schema import Hardware, Run

HF_ID = "example-org/example-model"
MODEL = "example:1b"
OVERHEAD = 29

runner = CliRunner()


@pytest.fixture(scope="module")
def bpe_path(tmp_path_factory):
    raw = HFRaw(models.BPE())
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    raw.train_from_iterator(["the quick brown fox jumps over the lazy dog 0123456789"] * 50, trainer)
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    raw.save(str(path))
    return path


@pytest.fixture
def env(tmp_path, monkeypatch, bpe_path):
    # 토크나이저 캐시
    cache = tmp_path / "cache" / "hf" / HF_ID.replace("/", "--") / "main"
    cache.mkdir(parents=True)
    (cache / "tokenizer.json").write_bytes(bpe_path.read_bytes())
    monkeypatch.setenv("PREMBENCH_CACHE", str(tmp_path / "cache"))
    # 설정 (가상 값)
    cfg = tmp_path / "config"
    cfg.write_text(
        f'submitter = "example-user"\nprovenance = "self"\nenvironment = "on_prem"\n'
        f'[hf_ids]\n"{MODEL}" = "{HF_ID}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PREMBENCH_CONFIG", str(cfg))
    # 하드웨어는 실행 머신과 무관하게 고정
    hw = Hardware(
        gpu_model="NVIDIA L4", gpu_count=1, gpu_vram_gb=22.5, interconnect="none", cpu_model=None, ram_gb=None,
        driver_version=None, cuda_version=None, gpu_power_limit_w=None, gpu_clock_locked=None,
        pcie_gen=None, pcie_lanes=None, numa_nodes=None,
    )
    monkeypatch.setattr(cell, "_hardware", lambda: hw)
    monkeypatch.setattr(cell.GpuSampler, "create", classmethod(lambda cls: None))
    return tmp_path


def _mock(**kw) -> MockServer:
    tok = HFTokenizer.from_file(_mock.bpe_path)
    cfg = MockConfig(
        models=(MODEL,), first_token_delay_s=0.01, inter_token_delay_s=0.001, tokens=8,
        prompt_tokens=lambda s: OVERHEAD + len(tok.encode(s)), **kw,
    )
    return MockServer(cfg)


@pytest.fixture(autouse=True)
def _bind_bpe(bpe_path):
    _mock.bpe_path = bpe_path


def _invoke(srv, *extra):
    args = ["run", "--url", srv.base_url, "--model", MODEL, "--concurrency", "2", "--in", "64", "--out", "8", *extra]
    return runner.invoke(app, args)


def test_run_emits_valid_record(env):
    out = env / "rec.json"
    with _mock() as srv:
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
    with _mock() as srv:
        res = _invoke(srv, "--ignore-eos")
        bodies = srv.state.bodies
    assert res.exit_code == 0, res.output
    record = json.loads(res.stdout[res.stdout.index("{"):])
    assert record["load"]["ignore_eos"] is None
    assert not any("ignore_eos" in b for b in bodies)


def test_missing_identity_errors_without_prompt(env, monkeypatch):
    (env / "config").write_text("", encoding="utf-8")
    with _mock() as srv:
        res = _invoke(srv)
        assert srv.state.requests == 0  # 아무 요청도 보내지 않는다
    assert res.exit_code == 2
    for name in ("submitter", "provenance", "environment", "hf_id"):
        assert name in res.output


def test_cloud_rented_requires_provider(env):
    with _mock() as srv:
        res = _invoke(srv, "--environment", "cloud_rented")
    assert res.exit_code == 2 and "provider" in res.output


def test_invalid_choice(env):
    with _mock() as srv:
        res = _invoke(srv, "--provenance", "friend")
    assert res.exit_code == 2 and "provenance" in res.output


def test_prompt_token_mismatch_aborts(env):
    # 엔진이 긴 프롬프트를 잘라 버리는 상황 (Ollama num_ctx 초과 등)
    tok = HFTokenizer.from_file(_mock.bpe_path)
    cfg = MockConfig(models=(MODEL,), first_token_delay_s=0.01, inter_token_delay_s=0.001, tokens=4,
                     prompt_tokens=lambda s: min(OVERHEAD + len(tok.encode(s)), 120))  # 보정(≤93)은 통과
    with MockServer(cfg) as srv:
        res = runner.invoke(app, ["run", "--url", srv.base_url, "--model", MODEL, "--concurrency", "1",
                                  "--in", "200", "--out", "4"])
    assert res.exit_code == 1
    assert "프롬프트 토큰" in res.output and "{" not in res.stdout  # 레코드는 내지 않는다


def test_unknown_model(env):
    with _mock() as srv:
        res = runner.invoke(app, ["run", "--url", srv.base_url, "--model", "nope", "--hf-id", HF_ID,
                                  "--concurrency", "1", "--in", "64", "--out", "4"])
    assert res.exit_code == 1 and "nope" in res.output


def test_remote_url_rejected(env):
    res = runner.invoke(app, ["run", "--url", "http://10.255.255.1:8000/v1", "--model", MODEL,
                              "--concurrency", "1", "--in", "64", "--out", "4"])
    assert res.exit_code == 1 and "원격" in res.output


def test_is_local_url():
    assert is_local_url("http://localhost:11434/v1")
    assert is_local_url("http://127.0.0.1:8000/v1")
    assert is_local_url("http://[::1]:8000/v1")
    assert not is_local_url("http://10.255.255.1:8000/v1")
    assert not is_local_url("http://no-such-host.invalid/v1")
