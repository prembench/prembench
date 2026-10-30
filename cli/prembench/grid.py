"""격자 순회 — ROADMAP 1.2 격자를 셀마다 반복 측정하고 대표 회차를 고른다.

반복 규칙 (ROADMAP 1.4):
- 같은 셀 2회. output_tps_total 또는 ttft_p95_ms 의 편차 (max−min)/min 가 10% 를 넘거나
  둘 중 하나라도 값이 없으면 3회째.
- 모든 회차를 레코드로 남기고, 대표 회차는 output_tps_total 의 nearest-rank 중앙값에 해당하는
  실제 회차 (3회면 가운데, 2회면 낮은 쪽). 평균을 내지 않는다.
- OOM 이 난 셀은 반복하지 않는다 (같은 OOM 을 다시 일으킬 뿐).
- 프롬프트 토큰 불일치(PromptTokenMismatch)는 그 회차만 새 시드로 1번 다시 잰다. 드물게 특정 무작위
  본문을 엔진 토크나이저가 다르게 나누는 경우가 있어서다. 다시 실패하면(context 한도 등 체계적 원인) 셀 오류.

OOM 감지: 요청 오류 문구(metrics) + 엔진 로그 (--engine-log). 로그는 회차 동안 새로 쓰인 부분만 본다.
엔진이 죽으면 남은 셀은 재지 않는다 — 측정 안 한 셀은 레코드가 없다 (지어내지 않음).
"""

from __future__ import annotations

import asyncio
import re
import secrets
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from prembench.cell import Cell, CellError, PromptTokenMismatch, Session
from prembench.load import NotReadyError
from prembench.metrics import percentile
from prembench.probe import probe_engine
from prembench.schema import Repetition, Run

DEFAULT_CONCURRENCY = (1, 4, 16, 64)
DEFAULT_SHAPES = ((512, 128), (2048, 512), (8192, 512), (32768, 512))
TOLERANCE = 0.10
MAX_REPETITIONS = 3

# 엔진 로그의 OOM 흔적. vLLM 의 KV 캐시 preemption 경고는 OOM 이 아니므로 넣지 않는다
_OOM_LOG = re.compile(
    r"CUDA out of memory|OutOfMemoryError|out of memory|No available memory for the cache blocks",
    re.IGNORECASE,
)


def cells(concurrency: Iterable[int] = DEFAULT_CONCURRENCY,
          shapes: Iterable[tuple[int, int]] = DEFAULT_SHAPES) -> list[Cell]:
    """입출력 길이 오름차순, 그 안에서 동시성 오름차순 (메모리를 덜 쓰는 셀부터)."""
    return [Cell(c, i, o) for i, o in shapes for c in concurrency]


def deviation(values: list[float | None]) -> float | None:
    """(max−min)/min. 값이 하나라도 없으면 None, min=0 이면 (전부 0 이 아닌 한) inf."""
    if not values or any(v is None for v in values):
        return None
    lo, hi = min(values), max(values)
    if lo == 0:
        return 0.0 if hi == 0 else float("inf")
    return (hi - lo) / lo


def needs_another(runs: list[Run]) -> bool:
    if len(runs) >= MAX_REPETITIONS:
        return False
    if len(runs) < 2:
        return True
    for metric in ("output_tps_total", "ttft_p95_ms"):
        d = deviation([getattr(r.result, metric) for r in runs])
        if d is None or d > TOLERANCE:
            return True
    return False


def representative(runs: list[Run]) -> int:
    """대표 회차의 위치 (0부터). output_tps_total 이 모두 없으면 첫 회차."""
    vals = [r.result.output_tps_total for r in runs]
    known = [v for v in vals if v is not None]
    if not known:
        return 0
    return vals.index(percentile(known, 50))


def finalize(runs: list[Run]) -> list[Run]:
    rep = representative(runs)
    n = len(runs)
    return [
        r.model_copy(update={"repetition": Repetition(index=i + 1, count=n, representative=i == rep)})
        for i, r in enumerate(runs)
    ]


def with_oom(run: Run) -> Run:
    return run.model_copy(update={"result": run.result.model_copy(update={"oom": True})})


class EngineLog:
    """엔진 로그 파일에서 마지막 표시 이후 새로 쓰인 OOM 줄을 찾는다."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def mark(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0

    def oom_lines(self, since: int) -> list[str]:
        if not self.path.exists():
            return []
        with self.path.open("rb") as f:
            size = f.seek(0, 2)
            f.seek(since if since <= size else 0)  # 로그가 회전돼 작아졌으면 처음부터
            text = f.read().decode(errors="replace")
        return [line for line in text.splitlines() if _OOM_LOG.search(line)]


@dataclass
class CellOutcome:
    cell: Cell
    records: list[Run] = field(default_factory=list)
    skipped: str | None = None  # 측정 전에 불가능하다고 판단
    error: str | None = None  # 측정 중 실패 (기록된 회차가 있을 수도 있음)
    oom_log: list[str] = field(default_factory=list)
    short_duration: bool = False  # min_duration_s 를 못 채운 회차가 있음
    retried: list[str] = field(default_factory=list)  # 프롬프트 토큰 불일치로 다시 잰 회차의 사유

    @property
    def oom(self) -> bool:
        return bool(self.oom_log) or any(r.result.oom for r in self.records)

    @property
    def representative(self) -> Run | None:
        return next((r for r in self.records if r.repetition.representative), None)


def _seeds(base: int | None) -> Iterator[int]:
    k = 0
    while True:
        # 회차마다 다른 시드 — 같은 시드면 이전 회차 프롬프트가 prefix 캐시에 남아 있다
        yield base + k if base is not None else secrets.randbelow(2**31)
        k += 1


async def _engine_alive(url: str) -> bool:
    return (await asyncio.to_thread(probe_engine, url)).reachable


async def run_grid(
    session: Session,
    grid: list[Cell],
    *,
    seed: int | None = None,
    engine_log: Path | None = None,
    on_cell: Callable[[CellOutcome], None] | None = None,
) -> list[CellOutcome]:
    """셀마다 반복 측정해 CellOutcome 을 낸다. on_cell 은 셀이 끝날 때마다 (레코드 저장용)."""
    seeds = _seeds(seed)
    log = EngineLog(engine_log) if engine_log else None
    outcomes: list[CellOutcome] = []
    dead: str | None = None

    for cell in grid:
        out = CellOutcome(cell)
        outcomes.append(out)
        if dead:
            out.skipped = dead
        elif reason := session.skip_reason(cell):
            out.skipped = reason
        else:
            runs: list[Run] = []
            while needs_another(runs):
                pos = log.mark() if log else 0
                try:
                    try:
                        run = await session.measure(cell, next(seeds))
                    except PromptTokenMismatch as e:
                        out.retried.append(str(e))
                        pos = log.mark() if log else 0
                        run = await session.measure(cell, next(seeds))
                except (CellError, NotReadyError) as e:
                    # 준비 확인 중에 죽은 경우도 엔진 로그에 OOM 흔적이 있으면 남긴다 (레코드는 없음)
                    hits = log.oom_lines(pos) if log else []
                    out.oom_log.extend(hits)
                    out.error = f"OOM (엔진 로그) — {e}" if hits else str(e)
                    break
                out.short_duration |= session.last_pool_exhausted
                hits = log.oom_lines(pos) if log else []
                if hits:
                    out.oom_log.extend(hits)
                    run = with_oom(run)
                runs.append(run)
                if run.result.oom:
                    break
            out.records = finalize(runs) if runs else []

            failed = out.error is not None or any(r.result.error_count for r in runs)
            if failed and not await _engine_alive(session.target.url):
                dead = f"엔진 응답 없음 ({cell} 이후) — 측정 안 함"
        if on_cell:
            on_cell(out)
    return outcomes
