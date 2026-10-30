import pytest
from tokenizers import Tokenizer as HFRaw
from tokenizers import decoders, models, pre_tokenizers, trainers

from mock_openai import MockConfig, MockServer
from prembench.prompts import (
    HFTokenizer,
    TemplateMismatchError,
    build_prompts,
    content_tokens_for,
    fit_text,
    korean_share,
    load_hf_tokenizer,
    measure_template_overhead,
    random_text,
    token_pools,
)


class GreedyTokenizer:
    """최장 일치 토크나이저. decode→encode 가 항등이 아니다: decode([a, b]) = "ab" → encode = [ab]."""

    VOCAB = {"a": 0, "b": 1, "ab": 2, "c": 3, "abc": 4, "가": 5, "나": 6, "가나": 7, " ": 8}
    SPECIAL = {"<s>": 99}

    def __init__(self):
        self.inv = {v: k for k, v in {**self.VOCAB, **self.SPECIAL}.items()}

    def encode(self, text: str) -> list[int]:
        out, i = [], 0
        while i < len(text):
            for j in range(min(len(text), i + 3), i, -1):  # 최장 토큰 3글자
                if text[i:j] in self.VOCAB:
                    out.append(self.VOCAB[text[i:j]])
                    i = j
                    break
            else:
                raise ValueError(text[i])
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inv[i] for i in ids)

    def vocab_ids(self) -> list[int]:
        return sorted(self.VOCAB.values())


@pytest.fixture(scope="module")
def bpe(tmp_path_factory) -> HFTokenizer:
    """한글이 섞인 코퍼스로 학습한 작은 byte-level BPE (네트워크 없이)."""
    raw = HFRaw(models.BPE())
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=800,
        special_tokens=["<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    corpus = ["the quick brown fox jumps over the lazy dog 12345"] * 50 + ["안녕하세요 측정 서버 모델 토큰 한국어 문장"] * 50
    raw.train_from_iterator(corpus, trainer)
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    raw.save(str(path))
    return HFTokenizer.from_file(path)


def test_pools_exclude_special_and_split_korean():
    pools = token_pools(GreedyTokenizer())
    assert 99 not in pools.general + pools.korean
    assert 8 not in pools.general  # 공백뿐인 토큰 제외
    assert set(pools.korean) == {5, 6, 7}


def test_fit_text_compensates_merges():
    tok = GreedyTokenizer()
    import random

    # a, b, c → decode "abc" → encode 1 토큰. 정확히 3 토큰이 될 때까지 보충해야 한다
    text = fit_text(tok, [0, 1, 3], 3, random.Random(0), [0, 1, 3])
    assert len(tok.encode(text)) == 3


@pytest.mark.parametrize("n", [1, 7, 128, 2048])
def test_exact_token_count_greedy(n):
    tok = GreedyTokenizer()
    text = random_text(tok, n, 42, token_pools(tok))
    assert len(tok.encode(text)) == n


@pytest.mark.parametrize("n", [1, 64, 512, 4096])
def test_exact_token_count_bpe(bpe, n):
    text = random_text(bpe, n, 7, token_pools(bpe))
    assert len(bpe.encode(text)) == n


def test_deterministic_and_distinct(bpe):
    pools = token_pools(bpe)
    a = build_prompts(bpe, 4, 64, seed=1, pools=pools)
    b = build_prompts(bpe, 6, 64, seed=1, pools=pools)
    assert a == b[:4]  # 같은 시드면 같고, 앞부분은 count 와 무관
    assert len(set(b)) == 6  # 요청마다 다름 → prefix 캐시 효과 제거
    assert build_prompts(bpe, 1, 64, seed=2, pools=pools) != a[:1]


def test_ko_ratio(bpe):
    pools = token_pools(bpe)
    assert pools.korean, "학습 코퍼스에 한글 병합 토큰이 있어야 함"
    plain = random_text(bpe, 256, 3, pools, ko_ratio=0.0)
    ko = random_text(bpe, 256, 3, pools, ko_ratio=0.5)
    assert korean_share(bpe, plain) == 0
    assert 0.3 < korean_share(bpe, ko) < 0.7
    assert len(bpe.encode(ko)) == 256


def test_ko_ratio_needs_korean_tokens():
    tok = GreedyTokenizer()
    pools = token_pools(tok)
    pools.korean.clear()
    with pytest.raises(ValueError, match="한글"):
        random_text(tok, 4, 0, pools, ko_ratio=0.5)


async def test_measure_template_overhead():
    tok = GreedyTokenizer()
    cfg = MockConfig(prompt_tokens=lambda s: 23 + len(tok.encode(s)))
    with MockServer(cfg) as srv:
        assert await measure_template_overhead(srv.base_url, "mock", tok, seed=0) == 23
        body = srv.state.bodies[0]
        assert body["max_tokens"] == 1 and not body.get("stream")


async def test_overhead_mismatch_means_wrong_tokenizer():
    # 엔진이 다른 토크나이저(여기선 글자 수)로 세면 길이별 오버헤드가 달라진다
    tok = GreedyTokenizer()
    with MockServer(MockConfig(prompt_tokens=lambda s: 23 + len(s))) as srv:
        with pytest.raises(TemplateMismatchError, match="토크나이저"):
            await measure_template_overhead(srv.base_url, "mock", tok, seed=0)


async def test_overhead_requires_usage():
    with MockServer(MockConfig(send_usage=False)) as srv:
        with pytest.raises(TemplateMismatchError, match="usage.prompt_tokens"):
            await measure_template_overhead(srv.base_url, "mock", GreedyTokenizer(), seed=0)


def test_content_tokens_for():
    assert content_tokens_for(512, 30) == 482
    with pytest.raises(ValueError):
        content_tokens_for(30, 30)


def test_load_hf_tokenizer_uses_cache(bpe, tmp_path, monkeypatch):
    # 캐시에 있으면 네트워크를 타지 않는다
    dest = tmp_path / "org--model" / "main" / "tokenizer.json"
    dest.parent.mkdir(parents=True)
    bpe._tok.save(str(dest))
    monkeypatch.setattr("prembench.prompts.httpx.get", lambda *a, **k: pytest.fail("network"))
    tok = load_hf_tokenizer("org/model", cache_dir=tmp_path)
    assert tok.encode("the fox") == bpe.encode("the fox")
