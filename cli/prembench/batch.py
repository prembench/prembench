"""조합 파일(matrix.toml) 실행 — 모델마다 엔진 기동 → 준비 확인 → 격자 → 엔진 종료.

runner/bootstrap.sh 가 클라우드 인스턴스에서 부른다. 같은 GPU 서버에서 직접 돌려도 된다.
--dry-run 은 아무것도 띄우지 않고 계획과 예상 시간·비용을 낸다. 예상치는 예산용이고 레코드에 들어가지 않는다.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shlex
import signal
import subprocess
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from prembench.cell import Cell, CellError, Session, Target
from prembench.grid import DEFAULT_CONCURRENCY, DEFAULT_SHAPES, CellOutcome, cells, run_grid
from prembench.load import NotReadyError
from prembench.prompts import TemplateMismatchError
from prembench.schema import Run

# 예상 시간용 상수 (예산 추정에만 쓴다)
_REP_OVERHEAD_S = 20  # 회차당 준비 확인·워밍업·프롬프트 생성·마무리
_READY_POLL_S = 5.0
_STOP_GRACE_S = 30.0
_MAX_LEN_ARG = re.compile(r"--max-model-len[ =](\d+)")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Instance(_Strict):
    provider: str = Field(min_length=1, description="클라우드 제공자. 레코드 provider 로 들어간다")
    gpu: str = Field(min_length=1, description="빌린 GPU (계획 표시용. 레코드의 GPU 는 probe 가 잰다)")
    price_per_hour_usd: float = Field(gt=0, description="인스턴스 시간당 가격 (선언값, 예상 비용용)")
    usd_krw: float = Field(gt=0, description="예상 비용 원화 환산용 환율 (선언값)")
    max_hours: float = Field(gt=0, description="이 시간을 넘으면 bootstrap 이 강제 종료 (비용 상한)")
    setup_min: float = Field(10, ge=0, description="설치·준비 예상 분 (예상 비용용)")


class EngineCfg(_Strict):
    name: Literal["vllm", "sglang"] = "vllm"
    version: str = Field(min_length=1, description="설치할 엔진 버전 (bootstrap 이 읽는다)")
    command: str = Field(
        "vllm serve {hf_id} --host 127.0.0.1 --port {port} "
        "--tensor-parallel-size {tp} --pipeline-parallel-size {pp} {args}",
        description="엔진 기동 명령 템플릿. {hf_id} {served} {port} {tp} {pp} {args}",
    )
    port: int = Field(8000, ge=1, le=65535)
    startup_timeout_min: float = Field(30, gt=0, description="모델 다운로드·적재까지 기다릴 최대 분")


class GridCfg(_Strict):
    concurrency: list[int] = Field(default_factory=lambda: list(DEFAULT_CONCURRENCY))
    shapes: list[str] = Field(default_factory=lambda: [f"{i}x{o}" for i, o in DEFAULT_SHAPES])
    min_duration_s: float = Field(60, ge=0)
    warmup: int | None = Field(None, ge=0)
    requests: int | None = Field(None, ge=1)
    ignore_eos: bool = True

    @field_validator("concurrency")
    @classmethod
    def _positive(cls, v: list[int]) -> list[int]:
        if not v or min(v) < 1:
            raise ValueError("1 이상 정수 목록")
        return v

    @field_validator("shapes")
    @classmethod
    def _shapes(cls, v: list[str]) -> list[str]:
        for s in v:
            if not re.fullmatch(r"\d+x\d+", s):
                raise ValueError(f"'입력x출력' 형식이어야 함: {s!r}")
        return v

    def parsed_shapes(self) -> list[tuple[int, int]]:
        return [tuple(int(x) for x in s.split("x")) for s in self.shapes]  # type: ignore[misc]


class ModelCfg(_Strict):
    hf_id: str = Field(min_length=1)
    served: str | None = Field(None, description="엔진이 쓰는 모델 이름. 기본: hf_id")
    args: str = Field("", description="엔진 추가 인자 원문")
    tp: int = Field(1, ge=1)
    pp: int = Field(1, ge=1)
    quantization: str | None = None
    weight_dtype: str | None = None
    startup_min: float = Field(10, ge=0, description="다운로드·적재 예상 분 (예상 비용용)")

    @property
    def served_name(self) -> str:
        return self.served or self.hf_id

    @property
    def max_model_len(self) -> int | None:
        m = _MAX_LEN_ARG.search(self.args)
        return int(m.group(1)) if m else None


class Matrix(_Strict):
    submitter: str = Field(min_length=1)
    provenance: Literal["self", "community", "scraped", "vendor"] = "self"
    instance: Instance
    engine: EngineCfg
    grid: GridCfg = Field(default_factory=GridCfg)
    models: list[ModelCfg] = Field(min_length=1)


def load_matrix(path: Path) -> Matrix:
    with path.open("rb") as f:
        return Matrix.model_validate(tomllib.load(f))


# ---------------------------------------------------------------- 계획·예상 비용


@dataclass
class ModelPlan:
    model: ModelCfg
    cells: list[Cell]
    skipped: list[Cell]  # args 의 --max-model-len 으로 미리 알 수 있는 건너뛸 셀
    minutes: tuple[float, float]  # (2회씩, 3회씩)


@dataclass
class Plan:
    models: list[ModelPlan]
    minutes: tuple[float, float]
    usd: tuple[float, float]
    krw: tuple[float, float]
    over_cap: bool  # 최악 예상이 max_hours 초과


def plan(m: Matrix) -> Plan:
    grid = cells(m.grid.concurrency, m.grid.parsed_shapes())
    rep_min = (m.grid.min_duration_s + _REP_OVERHEAD_S) / 60
    out = []
    for mc in m.models:
        limit = mc.max_model_len
        run = [c for c in grid if limit is None or c.input_tokens + c.output_tokens <= limit]
        skip = [c for c in grid if c not in run]
        out.append(ModelPlan(mc, run, skip, (mc.startup_min + len(run) * 2 * rep_min,
                                             mc.startup_min + len(run) * 3 * rep_min)))
    lo = m.instance.setup_min + sum(p.minutes[0] for p in out)
    hi = m.instance.setup_min + sum(p.minutes[1] for p in out)
    usd = (lo / 60 * m.instance.price_per_hour_usd, hi / 60 * m.instance.price_per_hour_usd)
    return Plan(out, (lo, hi), usd, (usd[0] * m.instance.usd_krw, usd[1] * m.instance.usd_krw),
                over_cap=hi / 60 > m.instance.max_hours)


# ---------------------------------------------------------------- 엔진 프로세스


class EngineStartError(RuntimeError):
    pass


def engine_command(m: Matrix, mc: ModelCfg) -> str:
    cmd = m.engine.command.format(
        hf_id=mc.hf_id, served=mc.served_name, port=m.engine.port, tp=mc.tp, pp=mc.pp, args=mc.args
    )
    return " ".join(cmd.split())


def start_engine(cmd: str, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    # 새 세션: 엔진이 띄운 하위 프로세스까지 한 번에 종료하기 위해
    return subprocess.Popen(shlex.split(cmd), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop_engine(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=_STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    except ProcessLookupError:
        pass


async def wait_ready(url: str, model: str, proc: subprocess.Popen, timeout_s: float,
                     poll_s: float = _READY_POLL_S) -> float:
    """실제 생성 요청이 성공할 때까지 기다린다 (/v1/models 200 은 준비로 치지 않는다). 걸린 초를 돌려준다."""
    start = time.monotonic()
    body = {"model": model, "messages": [{"role": "user", "content": "ready?"}], "max_tokens": 1}
    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            if proc.poll() is not None:
                raise EngineStartError(f"엔진이 준비 전에 종료됨 (exit {proc.returncode})")
            try:
                r = await client.post(url.rstrip("/") + "/chat/completions", json=body)
                if r.status_code == 200:
                    return time.monotonic() - start
            except httpx.HTTPError:
                pass
            if time.monotonic() - start > timeout_s:
                raise EngineStartError(f"{timeout_s / 60:.0f}분 안에 준비되지 않음")
            await asyncio.sleep(poll_s)


# ---------------------------------------------------------------- 실행


@dataclass
class ModelResult:
    hf_id: str
    batch_id: str
    engine_command: str
    log: str
    ready_s: float | None = None
    error: str | None = None
    cells: list[dict] = field(default_factory=list)

    @property
    def records(self) -> int:
        return sum(c["records"] for c in self.cells)


def _slug(hf_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", hf_id)


async def run_matrix(
    m: Matrix,
    out_dir: Path,
    batch_prefix: str,
    *,
    seed: int | None = None,
    on_cell: Callable[[ModelCfg, CellOutcome], None] | None = None,
) -> list[ModelResult]:
    """모델마다 엔진을 띄워 격자를 돈다. 레코드는 셀이 끝날 때마다 out_dir/records/ 에 쓴다."""
    records_dir = out_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    grid = cells(m.grid.concurrency, m.grid.parsed_shapes())
    url = f"http://127.0.0.1:{m.engine.port}/v1"
    results = []

    for mc in m.models:
        cmd = engine_command(m, mc)
        log = out_dir / "logs" / f"{_slug(mc.hf_id)}.log"
        res = ModelResult(mc.hf_id, f"{batch_prefix}-{_slug(mc.hf_id)}", cmd, str(log))
        results.append(res)

        def save(o: CellOutcome, mc: ModelCfg = mc, res: ModelResult = res) -> None:
            for r in o.records:
                _write(records_dir / f"{r.run_id}.json", r)
            res.cells.append({
                "cell": str(o.cell), "records": len(o.records), "skipped": o.skipped, "error": o.error,
                "oom": o.oom, "retried": len(o.retried),
                "short_duration": o.short_duration,
            })
            if on_cell:
                on_cell(mc, o)

        proc = start_engine(cmd, log)
        try:
            res.ready_s = await wait_ready(url, mc.served_name, proc, m.engine.startup_timeout_min * 60)
            target = Target(
                url=url, model=mc.served_name, hf_id=mc.hf_id, submitter=m.submitter,
                provenance=m.provenance, environment="cloud_rented", provider=m.instance.provider,
                batch_id=res.batch_id, ignore_eos=m.grid.ignore_eos, warmup_count=m.grid.warmup,
                request_count=m.grid.requests, min_duration_s=m.grid.min_duration_s,
                tensor_parallel=mc.tp, pipeline_parallel=mc.pp, quantization=mc.quantization,
                weight_dtype=mc.weight_dtype,
                # 방금 띄운 엔진의 첫 측정 = cold (준비 확인·보정 요청만 처리한 상태)
                model_load_state="cold", server_command=cmd,
            )
            session = await Session.open(target, calibration_seed=seed if seed is not None else 0)
            await run_grid(session, grid, seed=seed, engine_log=log, on_cell=save)
        except (EngineStartError, CellError, NotReadyError, TemplateMismatchError) as e:
            res.error = str(e)
        finally:
            stop_engine(proc)

    (out_dir / "summary.json").write_text(
        json.dumps([r.__dict__ | {"records": r.records} for r in results], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def _write(path: Path, run: Run) -> None:
    path.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")


def fmt_minutes(lo: float, hi: float) -> str:
    return f"{math.ceil(lo)}~{math.ceil(hi)}분"
