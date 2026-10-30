"""~/.prembench/config (TOML). 명령행 옵션이 우선. 경로는 PREMBENCH_CONFIG 로 바꿀 수 있다.

예 (값은 가상):

    submitter = "example-user"
    provenance = "self"
    environment = "on_prem"        # cloud_rented 면 provider 도
    # provider = "RunPod"

    [hf_ids]                       # 엔진 모델 이름 → Hugging Face ID (토크나이저·메타데이터용)
    "qwen2.5:1.5b" = "Qwen/Qwen2.5-1.5B-Instruct"
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


def config_path() -> Path:
    return Path(os.environ.get("PREMBENCH_CONFIG", Path.home() / ".prembench" / "config"))


def load_config(path: Path | None = None) -> dict:
    path = path or config_path()
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)
