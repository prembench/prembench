import json
import warnings
from pathlib import Path

import pytest
from typer.testing import CliRunner

from prembench import grid as grid_mod
from prembench.__main__ import app
from prembench.cell import Cell, CellError, PromptTokenMismatch
from prembench.grid import (
    EngineLog,
    cells,
    deviation,
    finalize,
    needs_another,
    representative,
    run_grid,
)
from prembench.load import NotReadyError
from prembench.schema import Run
from conftest import MODEL, mock_ollama
from test_schema import _record


def _run(tps: float | None, ttft_p95: float | None = 100.0, oom: bool = False) -> Run:
    rec = _record()
    rec["result"]["output_tps_total"] = tps
    rec["result"]["output_tps_per_user"] = tps
    rec["result"]["ttft_p50_ms"] = ttft_p95
    rec["result"]["ttft_p95_ms"] = ttft_p95
    rec["result"]["ttft_p99_ms"] = ttft_p95
    rec["result"]["oom"] = oom
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Run.model_validate(rec)


# ---------------------------------------------------------------- 규칙


def test_cells_order_and_count():
    g = cells()
    assert len(g) == 16
    assert g[0] == Cell(1, 512, 128) and g[3] == Cell(64, 512, 128) and g[-1] == Cell(64, 32768, 512)


def test_deviation():
    assert deviation([100.0, 110.0]) == pytest.approx(0.10)
    assert deviation([100.0, 90.0, 95.0]) == pytest.approx(10 / 90)
    assert deviation([100.0, None]) is None
    assert deviation([0.0, 0.0]) == 0.0
    assert deviation([0.0, 1.0]) == float("inf")


def test_needs_another():
    assert needs_another([])
    assert needs_another([_run(100)])
    assert not needs_another([_run(100), _run(105)])  # 둘 다 10% 이내
    assert not needs_another([_run(100), _run(110)])  # 정확히 10% 는 통과
    assert needs_another([_run(100), _run(111)])  # tps 편차
    assert needs_another([_run(100, 100), _run(100, 120)])  # ttft p95 편차
    assert needs_another([_run(None), _run(100)])  # 판정 불가 → 3회째
    assert not needs_another([_run(100), _run(200), _run(150)])  # 최대 3회


def test_representative_is_observed_median():
    assert representative([_run(10), _run(30), _run(20)]) == 2  # 3회 → 가운데 값의 실제 회차
    assert representative([_run(12), _run(10)]) == 1  # 2회 → 낮은 쪽
    assert representative([_run(None), _run(None)]) == 0
    assert representative([_run(None), _run(50), _run(40)]) == 2  # 값 있는 것 중 nearest-rank


def test_finalize_marks_one_representative():
    out = finalize([_run(10), _run(30), _run(20)])
    assert [(r.repetition.index, r.repetition.count, r.repetition.representative) for r in out] == [
        (1, 3, False), (2, 3, False), (3, 3, True)
    ]


def test_engine_log_reads_only_new_oom_lines(tmp_path):
    p = tmp_path / "engine.log"
    p.write_text("INFO old\nERROR CUDA out of memory (old)\n")
    log = EngineLog(p)
    pos = log.mark()
    with p.open("a") as f:
        f.write("INFO step\nWARNING preempted seq due to insufficient KV cache\n"
                "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")
    hits = log.oom_lines(pos)
    assert len(hits) == 1 and "Tried to allocate" in hits[0]  # 예전 줄·preemption 경고는 아님
    assert EngineLog(tmp_path / "missing.log").oom_lines(0) == []


def test_engine_log_rotation(tmp_path):
    p = tmp_path / "engine.log"
    p.write_text("x" * 1000)
    pos = EngineLog(p).mark()
    p.write_text("No available memory for the cache blocks\n")  # 회전돼 작아짐
    assert len(EngineLog(p).oom_lines(pos)) == 1


# ---------------------------------------------------------------- run_grid (가짜 세션)


class FakeSession:
    class _T:
        url = "http://127.0.0.1:1/v1"

    def __init__(self, plan, skip=None, log: Path | None = None):
        self.plan = plan  # cell → [tps 또는 예외 또는 "oom-log"]
        self.skip = skip or {}
        self.log = log
        self.target = self._T()
        self.last_pool_exhausted = False
        self.seeds = []

    def skip_reason(self, cell):
        return self.skip.get(cell)

    async def measure(self, cell, seed):
        self.seeds.append(seed)
        step = self.plan[cell].pop(0)
        if isinstance(step, Exception):
            raise step
        if step == "oom-log":
            with self.log.open("a") as f:
                f.write("RuntimeError: CUDA out of memory\n")
            return _run(1.0)
        return _run(step)


A, B, C = Cell(1, 512, 128), Cell(4, 512, 128), Cell(16, 512, 128)


@pytest.fixture(autouse=True)
def engine_alive(monkeypatch):
    # 기본: 엔진 살아 있음 (가짜 레코드에 오류가 있어도 실제 포트로 확인하러 가지 않게)
    monkeypatch.setattr(grid_mod, "_engine_alive", lambda url: _true())


async def test_two_reps_when_stable_three_when_not():
    s = FakeSession({A: [100.0, 104.0], B: [100.0, 150.0, 120.0]})
    out = await run_grid(s, [A, B], seed=7)
    assert [len(o.records) for o in out] == [2, 3]
    assert out[1].representative.result.output_tps_total == 120.0
    assert s.seeds == [7, 8, 9, 10, 11]  # 회차마다 다른 시드


async def test_skip_cell_not_measured():
    s = FakeSession({B: [1.0, 1.0]}, skip={A: "입력+출력 > max_model_len"})
    out = await run_grid(s, [A, B])
    assert out[0].skipped and out[0].records == []
    assert len(out[1].records) == 2


async def test_oom_from_engine_log_stops_repetition(tmp_path, monkeypatch):
    log = tmp_path / "engine.log"
    log.write_text("")
    s = FakeSession({A: ["oom-log", 1.0]}, log=log)
    out = await run_grid(s, [A], engine_log=log)
    assert len(out[0].records) == 1  # OOM 셀은 반복 안 함
    r = out[0].records[0]
    assert r.result.oom and r.repetition.count == 1 and r.repetition.representative
    assert out[0].oom_log


async def test_engine_down_stops_remaining_cells(monkeypatch):
    s = FakeSession({A: [NotReadyError("ConnectError")], B: [1.0, 1.0], C: [1.0, 1.0]})
    monkeypatch.setattr(grid_mod, "_engine_alive", lambda url: _false())
    out = await run_grid(s, [A, B, C])
    assert out[0].error and out[0].records == []
    assert all(o.skipped and "엔진 응답 없음" in o.skipped and not o.records for o in out[1:])


async def test_cell_error_keeps_earlier_reps_and_continues(monkeypatch):
    s = FakeSession({A: [100.0, CellError("프롬프트 토큰 불일치")], B: [1.0, 1.0]})
    out = await run_grid(s, [A, B])
    assert out[0].error and len(out[0].records) == 1 and out[0].records[0].repetition.representative
    assert len(out[1].records) == 2


async def test_prompt_mismatch_retried_once_with_new_seed():
    s = FakeSession({A: [PromptTokenMismatch("#3=510"), 100.0, 101.0], B: [1.0, 1.0]})
    out = await run_grid(s, [A, B], seed=0)
    assert len(out[0].records) == 2 and out[0].error is None
    assert out[0].retried == ["#3=510"]
    assert s.seeds[:3] == [0, 1, 2]  # 재시도는 새 시드


async def test_prompt_mismatch_twice_is_cell_error():
    s = FakeSession({A: [PromptTokenMismatch("a"), PromptTokenMismatch("b")], B: [1.0, 1.0]})
    out = await run_grid(s, [A, B])
    assert out[0].error == "b" and out[0].records == []
    assert len(out[1].records) == 2


async def _true():
    return True


async def _false():
    return False


# ---------------------------------------------------------------- prembench grid (mock Ollama)


def test_grid_command_writes_records(env):
    out_dir = env / "out"
    with mock_ollama() as srv:
        res = CliRunner().invoke(app, [
            "grid", "--url", srv.base_url, "--model", MODEL, "--out-dir", str(out_dir),
            "--concurrency", "1,2", "--shapes", "64x8,4000x200", "--min-duration", "0", "--requests", "2",
        ])
    assert res.exit_code == 0, res.output
    files = sorted(out_dir.glob("*.json"))
    runs = [Run.model_validate_json(f.read_text()) for f in files]
    # 4000+200 > mock context 4096 → 64x8 두 셀만 측정
    assert "skip" in res.output and "max_model_len" in res.output
    by_cell = {}
    for r in runs:
        by_cell.setdefault(r.load.concurrency, []).append(r)
    assert sorted(by_cell) == [1, 2]
    for reps in by_cell.values():
        assert len(reps) in (2, 3)
        assert sum(r.repetition.representative for r in reps) == 1
        assert {r.repetition.count for r in reps} == {len(reps)}
        assert len({r.load.random_seed for r in reps}) == len(reps)
    assert len({r.batch_id for r in runs}) == 1 and runs[0].batch_id
    # 두 번째 셀부터는 이미 적재된 모델
    assert {r.conditions.model_load_state for r in by_cell[2]} == {"warm"}
    assert "done: 2/4 cells" in res.output
    json.loads(files[0].read_text())
