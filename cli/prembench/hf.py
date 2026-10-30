"""Hugging Face Hub 파일·메타데이터 (httpx 만 사용, huggingface_hub 의존성 없음). gated 모델은 HF_TOKEN."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

_FILE_URL = "https://huggingface.co/{hf_id}/resolve/{revision}/{filename}"
_API_URL = "https://huggingface.co/api/models/{hf_id}"
_TIMEOUT_S = 60


def cache_dir() -> Path:
    return Path(os.environ.get("PREMBENCH_CACHE", Path.home() / ".cache" / "prembench"))


def _headers() -> dict[str, str]:
    t = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {t}"} if t else {}


def hf_download(hf_id: str, filename: str, revision: str = "main", root: Path | None = None) -> Path:
    """리포 파일을 받아 캐시하고 경로를 돌려준다. 캐시에 있으면 네트워크를 타지 않는다."""
    path = (root or cache_dir()) / "hf" / hf_id.replace("/", "--") / revision / filename
    if not path.exists():
        resp = httpx.get(
            _FILE_URL.format(hf_id=hf_id, revision=revision, filename=filename),
            headers=_headers(), follow_redirects=True, timeout=_TIMEOUT_S,
        )
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(path)
    return path


def hf_config(hf_id: str, revision: str = "main") -> dict:
    return json.loads(hf_download(hf_id, "config.json", revision).read_text(encoding="utf-8"))


def hf_param_count(hf_id: str, root: Path | None = None) -> int | None:
    """HF API 의 safetensors 메타데이터 파라미터 수. 없으면 None. 응답은 캐시한다."""
    path = (root or cache_dir()) / "hf" / hf_id.replace("/", "--") / "api.json"
    if not path.exists():
        resp = httpx.get(_API_URL.format(hf_id=hf_id), headers=_headers(), timeout=_TIMEOUT_S)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    total = (json.loads(path.read_text(encoding="utf-8")).get("safetensors") or {}).get("total")
    return int(total) if total else None
