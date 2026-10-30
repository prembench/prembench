"""고정 시드 랜덤 프롬프트 — 엔진이 세는 프롬프트 토큰 수가 정확히 input_tokens 가 되게 만든다.

정의: load.input_tokens = 엔진이 본 전체 프롬프트 토큰 (chat template 포함) = usage.prompt_tokens.
1. 토크나이저 어휘에서 고정 시드로 토큰 ID 를 뽑아 decode → 다시 encode 해 정확히 N 토큰인 본문을 만든다.
   decode→encode 가 항등이 아니라서(BPE 병합 등) 맞을 때까지 자르고 덧붙인다.
2. chat template 오버헤드는 추정하지 않고 엔진에 실제 요청을 보내 usage.prompt_tokens 로 잰다.
   길이가 다른 두 본문으로 재서 오버헤드가 다르면 토크나이저가 엔진과 다르다는 뜻 → 실패.

토크나이저는 선택 의존성 `prembench[tokenizer]` (HF tokenizers). tokenizer.json 은 httpx 로 받아 캐시한다.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

_HANGUL = re.compile(r"[가-힣]")
_MAX_FIT_ROUNDS = 50
_CALIBRATION_LENGTHS = (16, 64)
_HF_URL = "https://huggingface.co/{hf_id}/resolve/{revision}/tokenizer.json"
_CACHE_DIR = Path(os.environ.get("PREMBENCH_CACHE", Path.home() / ".cache" / "prembench")) / "tokenizers"


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def vocab_ids(self) -> list[int]:
        """일반 토큰 ID (special/added 토큰 제외)."""
        ...


class HFTokenizer:
    """`tokenizers.Tokenizer` 래퍼. special 토큰 없이 순수 본문만 encode 한다."""

    def __init__(self, tok) -> None:
        self._tok = tok

    @classmethod
    def from_file(cls, path: str | Path) -> HFTokenizer:
        try:
            from tokenizers import Tokenizer as _T
        except ImportError as e:
            raise ImportError("토크나이저가 필요함: pip install 'prembench[tokenizer]'") from e
        return cls(_T.from_file(str(path)))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

    def vocab_ids(self) -> list[int]:
        added = set(self._tok.get_added_tokens_decoder())
        return sorted(i for i in self._tok.get_vocab(with_added_tokens=False).values() if i not in added)


def load_hf_tokenizer(hf_id: str, revision: str = "main", cache_dir: Path = _CACHE_DIR) -> HFTokenizer:
    """HF Hub 에서 tokenizer.json 을 받아 캐시. gated 모델은 HF_TOKEN 환경변수."""
    path = cache_dir / hf_id.replace("/", "--") / revision / "tokenizer.json"
    if not path.exists():
        headers = {"Authorization": f"Bearer {t}"} if (t := os.environ.get("HF_TOKEN")) else {}
        resp = httpx.get(
            _HF_URL.format(hf_id=hf_id, revision=revision), headers=headers, follow_redirects=True, timeout=60
        )
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(path)
    return HFTokenizer.from_file(path)


@dataclass
class TokenPools:
    general: list[int]  # 디코드가 깨끗하고 한글이 없는 토큰
    korean: list[int]  # 디코드에 한글 음절이 있는 토큰


def token_pools(tok: Tokenizer) -> TokenPools:
    general, korean = [], []
    for i in tok.vocab_ids():
        s = tok.decode([i])
        # 빈 문자열·바이트 조각(�)·공백뿐인 토큰은 재인코딩이 불안정해서 뺀다
        if not s.strip() or "�" in s:
            continue
        (korean if _HANGUL.search(s) else general).append(i)
    return TokenPools(general, korean)


def fit_text(tok: Tokenizer, ids: list[int], n: int, rng: random.Random, pool: list[int]) -> str:
    """ids 로 시작해 encode 결과가 정확히 n 토큰인 문자열을 만든다."""
    for _ in range(_MAX_FIT_ROUNDS):
        text = tok.decode(ids)
        got = tok.encode(text)
        if len(got) == n:
            return text
        # 재인코딩 결과를 기준으로 자르거나 덧붙이고 다시 확인
        ids = got[:n] if len(got) > n else got + [rng.choice(pool) for _ in range(n - len(got))]
    raise ValueError(f"{n} 토큰 정확히 맞추기 실패 ({_MAX_FIT_ROUNDS}회)")


def random_text(tok: Tokenizer, n: int, seed: int | str, pools: TokenPools, ko_ratio: float = 0.0) -> str:
    """정확히 n 토큰(encode 기준)인 랜덤 본문. 같은 (seed, 토크나이저) 면 같은 결과."""
    if n < 1:
        raise ValueError("n >= 1")
    if not 0 <= ko_ratio <= 1:
        raise ValueError("ko_ratio 는 0~1")
    if ko_ratio > 0 and not pools.korean:
        raise ValueError("토크나이저에 한글 토큰이 없음")
    rng = random.Random(seed)
    ids = [rng.choice(pools.korean if rng.random() < ko_ratio else pools.general) for _ in range(n)]
    return fit_text(tok, ids, n, rng, pools.general if ko_ratio < 1 else pools.korean)


def build_prompts(
    tok: Tokenizer,
    count: int,
    content_tokens: int,
    seed: int,
    ko_ratio: float = 0.0,
    pools: TokenPools | None = None,
) -> list[str]:
    """요청 count 개 분량의 서로 다른 본문. i 번째는 시드 f"{seed}:{i}" 라 count 와 무관하게 같다."""
    pools = pools or token_pools(tok)
    return [random_text(tok, content_tokens, f"{seed}:{i}", pools, ko_ratio) for i in range(count)]


def korean_share(tok: Tokenizer, text: str) -> float:
    """본문 토큰 중 한글이 들어간 토큰 비율 (실측 확인용)."""
    ids = tok.encode(text)
    return sum(bool(_HANGUL.search(tok.decode([i]))) for i in ids) / len(ids) if ids else 0.0


class TemplateMismatchError(RuntimeError):
    """토크나이저가 엔진과 다르거나 템플릿 오버헤드가 길이에 따라 변함."""


async def measure_template_overhead(
    base_url: str,
    model: str,
    tok: Tokenizer,
    seed: int,
    pools: TokenPools | None = None,
    client: httpx.AsyncClient | None = None,
) -> int:
    """chat template 이 더하는 토큰 수 = usage.prompt_tokens − 본문 토큰 수. 엔진에 실제 요청으로 잰다."""
    pools = pools or token_pools(tok)
    url = base_url.rstrip("/") + "/chat/completions"
    own = client is None
    client = client or httpx.AsyncClient(timeout=120)
    overheads = []
    try:
        for n in _CALIBRATION_LENGTHS:
            text = random_text(tok, n, f"{seed}:calibrate:{n}", pools)
            resp = await client.post(
                url,
                json={"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 1},
            )
            resp.raise_for_status()
            prompt_tokens = (resp.json().get("usage") or {}).get("prompt_tokens")
            if prompt_tokens is None:
                raise TemplateMismatchError("엔진이 usage.prompt_tokens 를 보고하지 않음 — 입력 길이를 맞출 수 없음")
            overheads.append(prompt_tokens - n)
    finally:
        if own:
            await client.aclose()
    if len(set(overheads)) != 1:
        raise TemplateMismatchError(
            f"길이별 템플릿 오버헤드가 다름 {dict(zip(_CALIBRATION_LENGTHS, overheads))} — 토크나이저가 엔진과 다를 수 있음"
        )
    return overheads[0]


def content_tokens_for(input_tokens: int, overhead: int) -> int:
    n = input_tokens - overhead
    if n < 1:
        raise ValueError(f"input_tokens({input_tokens}) 가 템플릿 오버헤드({overhead}) 이하")
    return n
