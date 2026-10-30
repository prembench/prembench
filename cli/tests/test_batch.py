"""prembench batch / runner/bootstrap.sh — mock 엔진을 실제 하위 프로세스로 띄워 끝까지."""

import json
import os
import shutil
import socket
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from conftest import HF_ID
from prembench.batch import (
    EngineStartError,
    Matrix,
    engine_command,
    load_matrix,
    plan,
    run_matrix,
    start_engine,
    stop_engine,
    wait_ready,
)
from prembench.probe import apple_hardware
from prembench.schema import Run

ROOT = Path(__file__).resolve().parents[2]
MOCK = Path(__file__).with_name("mock_openai.py")
BOOTSTRAP = ROOT / "runner" / "bootstrap.sh"
VENV_BIN = Path(sys.executable).parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _matrix(port: int, bpe_path: Path, **over) -> dict:
    m = {
        "submitter": "example-user",
        "instance": {"provider": "ExampleCloud", "gpu": "Example GPU", "price_per_hour_usd": 1.0,
                     "usd_krw": 1000, "max_hours": 1},
        "engine": {
            "version": "0.0.0",
            "port": port,
            "startup_timeout_min": 0.5,
            "command": (f"{sys.executable} {MOCK} --port {{port}} --model {{served}} --flavor vllm "
                        f"--tokenizer {bpe_path} --max-model-len 40960 {{args}}"),
        },
        "grid": {"min_duration_s": 0, "warmup": 1, "requests": 2},
        "models": [{"hf_id": HF_ID}],
    }
    for k, v in over.items():
        m[k] = {**m.get(k, {}), **v} if isinstance(v, dict) else v
    return m


def _toml(d: dict, path: Path) -> Path:
    """테스트용 최소 TOML 직렬화 (표 1단계 + models 배열)."""
    def val(v):
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, list):
            return "[" + ", ".join(val(x) for x in v) + "]"
        return str(v)

    lines = [f"{k} = {val(v)}" for k, v in d.items() if not isinstance(v, (dict, list))]
    for k, v in d.items():
        if isinstance(v, dict):
            lines += [f"\n[{k}]"] + [f"{kk} = {val(vv)}" for kk, vv in v.items()]
    for mdl in d.get("models", []):
        lines += ["\n[[models]]"] + [f"{kk} = {val(vv)}" for kk, vv in mdl.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def hf_cache(env):
    """env(conftest) 가 만든 토크나이저 캐시 옆에 config.json·api.json (vLLM 경로는 HF 메타데이터를 쓴다)."""
    d = env / "cache" / "hf" / HF_ID.replace("/", "--")
    (d / "main" / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16", "max_position_embeddings": 32768}))
    (d / "api.json").write_text(json.dumps({"safetensors": {"total": 1_500_000_000}}))
    return env


# ---------------------------------------------------------------- 조합 파일·계획


def test_matrix_rejects_unknown_and_missing(tmp_path, bpe_path):
    m = _matrix(8000, bpe_path)
    Matrix.model_validate(m)
    with pytest.raises(Exception, match="max_hours"):
        Matrix.model_validate({**m, "instance": {k: v for k, v in m["instance"].items() if k != "max_hours"}})
    with pytest.raises(Exception, match="extra"):
        Matrix.model_validate({**m, "grid": {"shape": ["1x1"]}})
    with pytest.raises(Exception, match="입력x출력"):
        Matrix.model_validate({**m, "grid": {"shapes": ["512-128"]}})


def test_example_matrix_parses():
    m = load_matrix(ROOT / "runner" / "matrix.toml")
    assert m.instance.provider and m.models and m.engine.version


def test_plan_cost_and_skips(bpe_path):
    m = Matrix.model_validate(_matrix(8000, bpe_path, instance={"setup_min": 10, "price_per_hour_usd": 2.0, "max_hours": 2,
                                                                "usd_krw": 1000},
                                      grid={"min_duration_s": 60},
                                      models=[{"hf_id": "a/b", "args": "--max-model-len 9000", "startup_min": 5}]))
    p = plan(m)
    mp = p.models[0]
    assert len(mp.cells) == 12 and len(mp.skipped) == 4  # 32768x512 네 셀은 9000 초과
    rep_min = (60 + 20) / 60
    assert p.minutes == pytest.approx((10 + 5 + 12 * 2 * rep_min, 10 + 5 + 12 * 3 * rep_min))
    assert p.usd == pytest.approx((p.minutes[0] / 60 * 2.0, p.minutes[1] / 60 * 2.0))
    assert p.krw == pytest.approx((p.usd[0] * 1000, p.usd[1] * 1000))
    assert not p.over_cap


def test_plan_over_cap(bpe_path):
    m = Matrix.model_validate(_matrix(8000, bpe_path, instance={"max_hours": 0.1}, grid={"min_duration_s": 60}))
    assert plan(m).over_cap


def test_engine_command_template(bpe_path):
    m = Matrix.model_validate({**_matrix(8000, bpe_path), "engine": {"version": "1"},
                               "models": [{"hf_id": "o/m", "tp": 2, "args": "--max-model-len 4096"}]})
    assert engine_command(m, m.models[0]) == (
        "vllm serve o/m --host 127.0.0.1 --port 8000 --tensor-parallel-size 2 "
        "--pipeline-parallel-size 1 --max-model-len 4096"
    )


# ---------------------------------------------------------------- 엔진 프로세스


async def test_wait_ready_fails_when_engine_exits(tmp_path):
    proc = start_engine(f"{sys.executable} -c 'import sys; sys.exit(3)'", tmp_path / "e.log")
    with pytest.raises(EngineStartError, match="exit 3"):
        await wait_ready(f"http://127.0.0.1:{_free_port()}/v1", "m", proc, timeout_s=10, poll_s=0.05)


async def test_wait_ready_waits_for_real_generation(tmp_path, bpe_path):
    port = _free_port()
    proc = start_engine(f"{sys.executable} {MOCK} --port {port} --model m --startup-delay 0.5", tmp_path / "e.log")
    try:
        waited = await wait_ready(f"http://127.0.0.1:{port}/v1", "m", proc, timeout_s=20, poll_s=0.1)
        assert waited >= 0.4
    finally:
        stop_engine(proc)
    assert proc.poll() is not None


async def test_wait_ready_timeout(tmp_path):
    proc = start_engine(f"{sys.executable} -c 'import time; time.sleep(30)'", tmp_path / "e.log")
    try:
        with pytest.raises(EngineStartError, match="준비되지 않음"):
            await wait_ready(f"http://127.0.0.1:{_free_port()}/v1", "m", proc, timeout_s=0.3, poll_s=0.05)
    finally:
        stop_engine(proc)


# ---------------------------------------------------------------- 16셀 완주 (W3 완료 기준)


async def test_run_matrix_full_grid_with_mock_engine(hf_cache, bpe_path):
    port = _free_port()
    m = Matrix.model_validate(_matrix(port, bpe_path, models=[{"hf_id": HF_ID, "args": "--itl 0"}]))
    out = hf_cache / "out"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = await run_matrix(m, out, "test", seed=1)
    (res,) = results
    assert res.error is None, res.error
    assert len(res.cells) == 16 and all(c["records"] >= 2 and not c["error"] for c in res.cells)
    runs = [Run.model_validate_json(p.read_text()) for p in (out / "records").glob("*.json")]
    assert len(runs) == res.records
    keys = {(r.load.concurrency, r.load.input_tokens, r.load.output_tokens) for r in runs}
    assert len(keys) == 16
    for r in runs:
        assert r.environment == "cloud_rented" and r.provider == "ExampleCloud"
        assert r.engine.name == "vllm" and r.engine.server_command.startswith(sys.executable)
        assert r.engine.max_model_len == 40960 and r.load.ignore_eos is True
        assert r.model.quantization == "none" and r.model.weight_dtype == "bf16"
        assert all(p.prompt_tokens == r.load.input_tokens for p in r.result.per_request)
    assert {r.conditions.model_load_state for r in runs} == {"cold", "warm"}
    assert sum(r.conditions.model_load_state == "cold" for r in runs) == 1  # 기동 후 첫 측정만
    assert sum(r.repetition.representative for r in runs) == 16
    summary = json.loads((out / "summary.json").read_text())
    assert summary[0]["records"] == res.records


async def test_run_matrix_engine_oom_crash(hf_cache, bpe_path):
    """엔진이 측정 중 OOM 로그를 남기고 죽으면: 그 셀은 oom=true, 남은 셀은 재지 않는다."""
    port = _free_port()
    m = Matrix.model_validate(_matrix(port, bpe_path, grid={"concurrency": [1, 2], "shapes": ["64x8", "128x8"]},
                                      models=[{"hf_id": HF_ID, "args": "--oom-at 12"}]))
    out = hf_cache / "out"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (res,) = await run_matrix(m, out, "oom", seed=1)
    # 죽는 시점(측정 중 / 다음 회차 준비 확인 중)에 따라 레코드의 oom 또는 셀 오류로 남는다
    oom_cells = [c for c in res.cells if c["oom"]]
    assert len(oom_cells) == 1
    after = res.cells[res.cells.index(oom_cells[0]) + 1:]
    assert after and all(c["skipped"] and "엔진 응답 없음" in c["skipped"] for c in after)


# ---------------------------------------------------------------- bootstrap.sh


def _has_hardware() -> bool:
    return shutil.which("nvidia-smi") is not None or apple_hardware() is not None


def _bootstrap(args, workdir: Path, extra_env: dict, timeout=300):
    env = {**os.environ, "SKIP_INSTALL": "1", "PYTHON": sys.executable, "WORKDIR": str(workdir),
           "PATH": f"{VENV_BIN}:{os.environ['PATH']}", **extra_env}
    return subprocess.run(["bash", str(BOOTSTRAP), *args], env=env, capture_output=True, text=True, timeout=timeout)


def test_bootstrap_dry_run(tmp_path):
    res = _bootstrap([str(ROOT / "runner" / "matrix.toml"), "--dry-run"], tmp_path,
                     {"SHUTDOWN_CMD": f"touch {tmp_path}/SHUTDOWN"})
    assert res.returncode == 0, res.stderr
    assert "예상" in res.stderr and "$" in res.stderr and "원" in res.stderr
    assert not (tmp_path / "SHUTDOWN").exists()  # dry-run 은 끄지 않는다
    assert not (tmp_path / "results").exists()


def test_bootstrap_requires_matrix(tmp_path):
    res = _bootstrap([], tmp_path, {})
    assert res.returncode == 2 and "matrix" in res.stderr


@pytest.mark.skipif(not _has_hardware(), reason="probe 할 GPU 가 없음")
def test_bootstrap_full_run_with_mock_engine_then_shutdown(hf_cache, bpe_path):
    port = _free_port()
    matrix = _toml(_matrix(port, bpe_path, grid={"concurrency": [1, 4], "shapes": ["64x8", "256x16"]}),
                   hf_cache / "matrix.toml")
    work = hf_cache / "work"
    res = _bootstrap([str(matrix)], work, {
        "SHUTDOWN_CMD": f"touch {work}/SHUTDOWN",
        "UPLOAD_CMD": "false",  # 업로드가 실패해도 종료는 진행돼야 한다
        "PREMBENCH_CACHE": str(hf_cache / "cache"),
    })
    assert res.returncode == 0, res.stdout[-3000:]
    assert (work / "SHUTDOWN").exists()
    (result_dir,) = (work / "results").iterdir()
    assert len(list((result_dir / "records").glob("*.json"))) >= 8  # 4셀 × 2회 이상
    assert (result_dir / "summary.json").exists() and (result_dir / "matrix.toml").exists()
    log = (result_dir / "bootstrap.log").read_text()
    assert "업로드 실패" in log and "인스턴스 종료" in log


@pytest.mark.skipif(not _has_hardware(), reason="probe 할 GPU 가 없음")
def test_bootstrap_watchdog_stops_at_max_hours(hf_cache, bpe_path):
    port = _free_port()
    # 엔진이 오래 안 뜨는 상황: max_hours 1초 뒤 감시자가 끝내고 인스턴스를 끈다
    matrix = _toml(_matrix(port, bpe_path, instance={"max_hours": 0.0003}, engine={"startup_timeout_min": 5},
                           models=[{"hf_id": HF_ID, "args": "--startup-delay 60"}]), hf_cache / "matrix.toml")
    work = hf_cache / "work"
    res = _bootstrap([str(matrix)], work, {"SHUTDOWN_CMD": f"touch {work}/SHUTDOWN",
                                           "PREMBENCH_CACHE": str(hf_cache / "cache")}, timeout=60)
    assert res.returncode != 0
    assert (work / "SHUTDOWN").exists()
    assert "watchdog" in res.stdout
