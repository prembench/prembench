"""공용 fixture: mock Ollama + 로컬 BPE 토크나이저 + 가상 설정 (네트워크 없음)."""

import pytest
from tokenizers import Tokenizer as HFRaw
from tokenizers import decoders, models, pre_tokenizers, trainers

from mock_openai import MockConfig, MockServer
from prembench import cell
from prembench.prompts import HFTokenizer
from prembench.schema import Hardware

HF_ID = "example-org/example-model"
MODEL = "example:1b"
OVERHEAD = 29


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


def mock_ollama(**kw) -> MockServer:
    """Ollama 흉내 mock. 프롬프트 토큰 = 템플릿 오버헤드 + 테스트 BPE 토큰 수."""
    tok = HFTokenizer.from_file(mock_ollama.bpe_path)
    cfg = MockConfig(
        models=(MODEL,), first_token_delay_s=0.01, inter_token_delay_s=0.001, tokens=8,
        prompt_tokens=lambda s: OVERHEAD + len(tok.encode(s)), **kw,
    )
    return MockServer(cfg)


@pytest.fixture(autouse=True)
def _bind_bpe(bpe_path):
    mock_ollama.bpe_path = bpe_path


