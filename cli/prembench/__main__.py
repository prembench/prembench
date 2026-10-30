import asyncio
import json
import secrets
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

import typer
from rich.console import Console
from rich.table import Table

from prembench import __version__
from prembench.batch import fmt_minutes, load_matrix, plan, run_matrix
from prembench.cell import Cell, CellError, Session, Target, run_cell
from prembench.config import config_path, load_config
from prembench.load import NotReadyError
from prembench.probe import as_dict, probe_engine, probe_host
from prembench.prompts import TemplateMismatchError
from prembench.grid import DEFAULT_CONCURRENCY, DEFAULT_SHAPES, CellOutcome, cells, run_grid
from prembench.schema import Environment, Provenance, Run, SchemaWarning, json_schema

app = typer.Typer(help="온프레미스 LLM 서빙 실측 벤치마크 CLI.", no_args_is_help=True)


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True, help="버전 출력 후 종료."
    ),
) -> None:
    pass


@app.command()
def schema(
    out: Path | None = typer.Option(None, "--out", "-o", help="저장 경로. 없으면 stdout."),
) -> None:
    """측정 레코드(Run)의 JSON Schema 를 내보낸다."""
    text = json.dumps(json_schema(), ensure_ascii=False, indent=2) + "\n"
    if out is None:
        typer.echo(text, nl=False)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out}")


@app.command()
def probe(
    url: str | None = typer.Option(
        None, "--url", help="엔진 OpenAI-compatible base URL. 예: http://localhost:8000/v1"
    ),
    as_json: bool = typer.Option(False, "--json", help="JSON 으로 출력."),
) -> None:
    """GPU·드라이버·CUDA·CPU·RAM 과 (--url 이 있으면) 서빙 엔진을 감지한다."""
    host = probe_host()
    engine = probe_engine(url) if url else None

    if as_json:
        typer.echo(json.dumps(as_dict(host, engine), ensure_ascii=False, indent=2))
        return

    console = Console()
    if not host.gpus:
        console.print(f"[bold red]no GPU[/] — NVIDIA GPU 미검출 ({host.no_gpu_reason})")
    else:
        t = Table(title="GPU")
        t.add_column("#")
        t.add_column("name")
        t.add_column("VRAM (GB)", justify="right")
        for i, g in enumerate(host.gpus):
            t.add_row(str(i), g.name, f"{g.vram_gb:g}")
        console.print(t)

    na = "[dim]n/a[/]"
    cuda = f"{host.cuda_version} ({host.cuda_version_source})" if host.cuda_version else na
    for label, value in [
        ("driver", host.driver_version or na),
        ("cuda", cuda),
        ("interconnect", host.interconnect or na),
        ("cpu", host.cpu_model or na),
        ("ram_gb", f"{host.ram_gb:g}" if host.ram_gb is not None else na),
    ]:
        console.print(f"{label:>12}: {value}")

    if engine is not None:
        console.print()
        if not engine.reachable:
            console.print(f"[bold red]engine unreachable[/] — {engine.error}")
        else:
            console.print(f"{'engine':>12}: {engine.name or '[yellow]unknown[/]'} {engine.version or ''}")
            console.print(f"{'models':>12}: {', '.join(engine.models) or na}")
            if engine.max_model_len is not None:
                console.print(f"{'max_model_len':>12}: {engine.max_model_len}")


_CONFIG_EXAMPLE = """\
submitter = "example-user"
provenance = "self"          # self | community | scraped | vendor
environment = "on_prem"      # cloud_rented 면 provider 도
# provider = "RunPod"

[hf_ids]
"qwen2.5:1.5b" = "Qwen/Qwen2.5-1.5B-Instruct"
"""

# run·grid 공통 옵션
_URL = typer.Option(..., "--url", help="엔진 OpenAI-compatible base URL. 예: http://localhost:11434/v1")
_MODEL = typer.Option(..., "--model", help="엔진에 요청할 모델 이름")
_WARMUP = typer.Option(None, "--warmup", min=0, help="워밍업 요청 수. 기본: concurrency")
_REQUESTS = typer.Option(None, "--requests", min=1, help="측정 최소 요청 수. 기본: concurrency × 5")
_MIN_DURATION = typer.Option(60.0, "--min-duration", min=0, help="측정 최소 시간(초). 요청 수와 둘 다 채운다")
_IGNORE_EOS = typer.Option(False, "--ignore-eos", help="EOS 무시하고 --out 까지 생성 (vLLM·SGLang 만 적용)")
_HF_ID = typer.Option(None, "--hf-id", help="HF 모델 ID (토크나이저·메타데이터). 기본: config [hf_ids] 또는 --model")
_SUBMITTER = typer.Option(None, "--submitter")
_PROVENANCE = typer.Option(None, "--provenance", help="self | community | scraped | vendor")
_ENVIRONMENT = typer.Option(None, "--environment", help="cloud_rented | on_prem")
_PROVIDER = typer.Option(None, "--provider", help="cloud_rented 일 때 클라우드 제공자")
_BATCH_ID = typer.Option(None, "--batch-id")
_TP = typer.Option(None, "--tp", min=1, help="tensor parallel (GPU 2장 이상이면 필수)")
_PP = typer.Option(None, "--pp", min=1, help="pipeline parallel (GPU 2장 이상이면 필수)")
_QUANT = typer.Option(None, "--quantization", help="자동 감지 대신 쓸 양자화 방식")
_DTYPE = typer.Option(None, "--weight-dtype", help="자동 감지 대신 쓸 가중치 dtype")
_LOAD_STATE = typer.Option(None, "--model-load-state", help="첫 측정 직전 cold | warm (Ollama 외 엔진은 필수)")


def _target(
    *, url: str, model: str, hf_id: str | None, submitter: str | None, provenance: str | None,
    environment: str | None, provider: str | None, batch_id: str | None, ignore_eos: bool,
    warmup: int | None, requests: int | None, min_duration: float, tp: int | None, pp: int | None,
    quantization: str | None, weight_dtype: str | None, model_load_state: str | None,
) -> Target:
    """옵션 > config 순으로 채운다. 빠지거나 잘못된 값은 묻지 않고 목록을 보여준 뒤 종료(2)."""
    cfg = load_config()
    submitter = submitter or cfg.get("submitter")
    provenance = provenance or cfg.get("provenance")
    environment = environment or cfg.get("environment")
    provider = provider or cfg.get("provider")
    hf_id = hf_id or (cfg.get("hf_ids") or {}).get(model) or (model if "/" in model else None)

    missing = [
        name for name, value in (
            ("submitter", submitter), ("provenance", provenance), ("environment", environment),
        ) if not value
    ]
    if environment == "cloud_rented" and not provider:
        missing.append("provider")
    if hf_id is None:
        missing.append(f"hf_id (--hf-id 또는 config [hf_ids] 에 \"{model}\")")
    bad = []
    if provenance and provenance not in get_args(Provenance):
        bad.append(f"provenance={provenance!r} (허용: {', '.join(get_args(Provenance))})")
    if environment and environment not in get_args(Environment):
        bad.append(f"environment={environment!r} (허용: {', '.join(get_args(Environment))})")
    if environment == "on_prem" and provider:
        bad.append("environment=on_prem 인데 provider 가 있음")
    if model_load_state not in (None, "cold", "warm"):
        bad.append(f"model-load-state={model_load_state!r} (허용: cold, warm)")
    if missing or bad:
        lines = []
        if missing:
            lines.append("빠진 값: " + ", ".join(missing))
        if bad:
            lines.append("잘못된 값: " + "; ".join(bad))
        lines.append(f"명령행 옵션으로 주거나 {config_path()} 에 설정 (TOML). 예:\n{_CONFIG_EXAMPLE}")
        typer.echo("error: " + "\n".join(lines), err=True)
        raise typer.Exit(2)

    return Target(
        url=url, model=model, hf_id=hf_id, submitter=submitter, provenance=provenance,
        environment=environment, provider=provider, batch_id=batch_id, ignore_eos=ignore_eos,
        warmup_count=warmup, request_count=requests, min_duration_s=min_duration,
        tensor_parallel=tp, pipeline_parallel=pp, quantization=quantization, weight_dtype=weight_dtype,
        model_load_state=model_load_state,
    )


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def _echo_warnings(caught: list[warnings.WarningMessage]) -> None:
    for w in caught:
        if issubclass(w.category, SchemaWarning):
            typer.echo(f"warning: {w.message}", err=True)


def _summary(r: Run) -> str:
    x = r.result
    return (
        f"ttft p95 {_fmt(x.ttft_p95_ms)} ms  tpot p50 {_fmt(x.tpot_p50_ms)} ms  "
        f"out tps {_fmt(x.output_tps_total)}  errors {x.error_count}{'  OOM' if x.oom else ''}"
    )


_SEED_HELP = (
    "프롬프트·샘플링 난수 시드. 기본: 매번 무작위로 뽑아 레코드에 기록 "
    "(같은 시드를 다시 쓰면 엔진 prefix 캐시에 이전 실행의 프롬프트가 남아 있을 수 있다)"
)


@app.command()
def run(
    url: str = _URL,
    model: str = _MODEL,
    concurrency: int = typer.Option(..., "--concurrency", min=1, help="동시 요청 수"),
    input_tokens: int = typer.Option(..., "--in", min=1, help="입력 길이 (엔진이 센 프롬프트 토큰, chat template 포함)"),
    output_tokens: int = typer.Option(..., "--out", min=1, help="출력 길이 = max_tokens"),
    seed: int | None = typer.Option(None, "--seed", help=_SEED_HELP),
    warmup: int | None = _WARMUP,
    requests: int | None = _REQUESTS,
    min_duration: float = _MIN_DURATION,
    ignore_eos: bool = _IGNORE_EOS,
    out_file: Path | None = typer.Option(None, "--out-file", help="레코드 JSON 저장 경로. 없으면 stdout"),
    hf_id: str | None = _HF_ID,
    submitter: str | None = _SUBMITTER,
    provenance: str | None = _PROVENANCE,
    environment: str | None = _ENVIRONMENT,
    provider: str | None = _PROVIDER,
    batch_id: str | None = _BATCH_ID,
    tp: int | None = _TP,
    pp: int | None = _PP,
    quantization: str | None = _QUANT,
    weight_dtype: str | None = _DTYPE,
    model_load_state: str | None = _LOAD_STATE,
) -> None:
    """격자 셀 1개를 1회 측정해 스키마 레코드 JSON 1개를 낸다."""
    target = _target(
        url=url, model=model, hf_id=hf_id, submitter=submitter, provenance=provenance,
        environment=environment, provider=provider, batch_id=batch_id, ignore_eos=ignore_eos,
        warmup=warmup, requests=requests, min_duration=min_duration, tp=tp, pp=pp,
        quantization=quantization, weight_dtype=weight_dtype, model_load_state=model_load_state,
    )
    if seed is None:
        seed = secrets.randbelow(2**31)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SchemaWarning)
        try:
            record = asyncio.run(run_cell(target, Cell(concurrency, input_tokens, output_tokens), seed))
        except (CellError, NotReadyError, TemplateMismatchError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1) from e
    _echo_warnings(caught)
    typer.echo(f"ok: {record.run_id}  {_summary(record)}", err=True)
    text = record.model_dump_json(indent=2) + "\n"
    if out_file is None:
        typer.echo(text, nl=False)
    else:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out_file}", err=True)


def _int_list(text: str) -> list[int]:
    try:
        vals = [int(x) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise typer.BadParameter(f"정수 목록이어야 함: {text!r}") from e
    if not vals or min(vals) < 1:
        raise typer.BadParameter(f"1 이상 정수 목록이어야 함: {text!r}")
    return vals


def _shapes(text: str) -> list[tuple[int, int]]:
    out = []
    for part in text.split(","):
        try:
            i, o = (int(x) for x in part.strip().lower().split("x"))
        except ValueError as e:
            raise typer.BadParameter(f"'입력x출력' 목록이어야 함 (예: 512x128,2048x512): {text!r}") from e
        if i < 1 or o < 1:
            raise typer.BadParameter(f"1 이상이어야 함: {part!r}")
        out.append((i, o))
    return out


@app.command()
def grid(
    url: str = _URL,
    model: str = _MODEL,
    out_dir: Path = typer.Option(..., "--out-dir", help="레코드 JSON 을 셀이 끝날 때마다 여기에 쓴다"),
    concurrency: str = typer.Option(",".join(map(str, DEFAULT_CONCURRENCY)), "--concurrency",
                                    help="동시성 목록"),
    shapes: str = typer.Option(",".join(f"{i}x{o}" for i, o in DEFAULT_SHAPES), "--shapes",
                               help="입력x출력 목록"),
    engine_log: Path | None = typer.Option(None, "--engine-log", help="엔진 로그 파일 (OOM 감지)"),
    seed: int | None = typer.Option(None, "--seed", help="기준 시드. 회차마다 +1. 기본: 회차마다 무작위"),
    warmup: int | None = _WARMUP,
    requests: int | None = _REQUESTS,
    min_duration: float = _MIN_DURATION,
    ignore_eos: bool = _IGNORE_EOS,
    hf_id: str | None = _HF_ID,
    submitter: str | None = _SUBMITTER,
    provenance: str | None = _PROVENANCE,
    environment: str | None = _ENVIRONMENT,
    provider: str | None = _PROVIDER,
    batch_id: str | None = _BATCH_ID,
    tp: int | None = _TP,
    pp: int | None = _PP,
    quantization: str | None = _QUANT,
    weight_dtype: str | None = _DTYPE,
    model_load_state: str | None = _LOAD_STATE,
) -> None:
    """격자 전체를 측정한다. 셀마다 2회(편차 10% 넘으면 3회), 모든 회차를 레코드로 쓴다."""
    grid_cells = cells(_int_list(concurrency), _shapes(shapes))
    if batch_id is None:
        batch_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"
    target = _target(
        url=url, model=model, hf_id=hf_id, submitter=submitter, provenance=provenance,
        environment=environment, provider=provider, batch_id=batch_id, ignore_eos=ignore_eos,
        warmup=warmup, requests=requests, min_duration=min_duration, tp=tp, pp=pp,
        quantization=quantization, weight_dtype=weight_dtype, model_load_state=model_load_state,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(o: CellOutcome) -> None:
        for r in o.records:
            (out_dir / f"{r.run_id}.json").write_text(r.model_dump_json(indent=2) + "\n", encoding="utf-8")
        if o.skipped:
            typer.echo(f"skip  {o.cell}: {o.skipped}", err=True)
        for r in o.records:
            mark = "*" if r.repetition.representative else " "
            typer.echo(f"{mark}{r.repetition.index}/{r.repetition.count} {o.cell}: {_summary(r)}", err=True)
        if o.short_duration:
            typer.echo(f"note  {o.cell}: 최소 측정 시간을 못 채운 회차가 있음 (프롬프트 풀 소진)", err=True)
        if o.oom_log:
            typer.echo(f"oom   {o.cell}: 엔진 로그 {len(o.oom_log)}줄 — {o.oom_log[0][:120]}", err=True)
        for reason in o.retried:
            typer.echo(f"retry {o.cell}: {reason}", err=True)
        if o.error:
            typer.echo(f"fail  {o.cell}: {o.error}", err=True)

    async def main() -> list[CellOutcome]:
        session = await Session.open(target, calibration_seed=seed if seed is not None else 0)
        typer.echo(
            f"batch {batch_id}: {session.engine.name} {session.engine.version}, {session.hardware.gpu_model}, "
            f"{len(grid_cells)} cells → {out_dir}",
            err=True,
        )
        return await run_grid(session, grid_cells, seed=seed, engine_log=engine_log, on_cell=save)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SchemaWarning)
        try:
            outcomes = asyncio.run(main())
        except (CellError, NotReadyError, TemplateMismatchError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1) from e
    _echo_warnings(caught)

    measured = sum(1 for o in outcomes if o.records)
    records = sum(len(o.records) for o in outcomes)
    typer.echo(
        f"done: {measured}/{len(outcomes)} cells, {records} records, "
        f"{sum(1 for o in outcomes if o.skipped)} skipped, {sum(1 for o in outcomes if o.error)} failed",
        err=True,
    )
    if measured == 0:
        raise typer.Exit(1)


@app.command()
def batch(
    matrix: Path = typer.Argument(..., exists=True, dir_okay=False, help="조합 파일 (matrix.toml)"),
    out_dir: Path | None = typer.Option(None, "--out-dir", help="records/ logs/ summary.json 을 쓸 곳"),
    dry_run: bool = typer.Option(False, "--dry-run", help="실행하지 않고 계획과 예상 시간·비용만 출력"),
    seed: int | None = typer.Option(None, "--seed", help="기준 시드. 기본: 회차마다 무작위"),
) -> None:
    """조합 파일의 모델마다 엔진을 띄워 격자를 측정한다 (runner/bootstrap.sh 가 부른다)."""
    try:
        m = load_matrix(matrix)
    except Exception as e:  # TOML 문법·스키마 오류를 그대로 보여준다
        typer.echo(f"error: {matrix}: {e}", err=True)
        raise typer.Exit(2) from e

    p = plan(m)
    console = Console(stderr=True)
    t = Table(title=f"{m.instance.provider} {m.instance.gpu} — ${m.instance.price_per_hour_usd}/h")
    for col in ("model", "cells", "skip (max-model-len)", "time"):
        t.add_column(col)
    for mp in p.models:
        t.add_row(mp.model.hf_id, str(len(mp.cells)), str(len(mp.skipped)), fmt_minutes(*mp.minutes))
    console.print(t)
    console.print(
        f"예상: {fmt_minutes(*p.minutes)} (셀당 2~3회, 설치 {m.instance.setup_min:g}분 포함)  "
        f"비용 ${p.usd[0]:.2f}~${p.usd[1]:.2f}  ≈ {p.krw[0]:,.0f}~{p.krw[1]:,.0f}원 "
        f"(환율 {m.instance.usd_krw:g})"
    )
    console.print("[dim]예상치는 예산용 — 레코드에 들어가지 않는다[/]")
    if p.over_cap:
        console.print(f"[bold yellow]warning[/]: 최악 예상이 max_hours {m.instance.max_hours:g}h 를 넘음 — 중간에 강제 종료될 수 있다")
    if dry_run:
        return
    if out_dir is None:
        typer.echo("error: --out-dir 필요", err=True)
        raise typer.Exit(2)

    batch_prefix = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"

    def progress(mc, o: CellOutcome) -> None:
        if o.skipped:
            typer.echo(f"skip  {mc.served_name} {o.cell}: {o.skipped}", err=True)
        for r in o.records:
            mark = "*" if r.repetition.representative else " "
            typer.echo(f"{mark}{r.repetition.index}/{r.repetition.count} {mc.served_name} {o.cell}: {_summary(r)}",
                       err=True)
        if o.error:
            typer.echo(f"fail  {mc.served_name} {o.cell}: {o.error}", err=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SchemaWarning)
        results = asyncio.run(run_matrix(m, out_dir, batch_prefix, seed=seed, on_cell=progress))
    _echo_warnings(caught)
    for r in results:
        status = f"error: {r.error}" if r.error else "ok"
        typer.echo(f"model {r.hf_id}: {r.records} records, {len(r.cells)} cells — {status}", err=True)
    if not any(r.records for r in results):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
