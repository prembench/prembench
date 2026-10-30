import asyncio
import json
import secrets
import warnings
from pathlib import Path
from typing import get_args

import typer
from rich.console import Console
from rich.table import Table

from prembench import __version__
from prembench.cell import CellError, CellSpec, run_cell
from prembench.config import config_path, load_config
from prembench.load import NotReadyError
from prembench.probe import as_dict, probe_engine, probe_host
from prembench.prompts import TemplateMismatchError
from prembench.schema import Environment, Provenance, SchemaWarning, json_schema

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


@app.command()
def run(
    url: str = typer.Option(..., "--url", help="엔진 OpenAI-compatible base URL. 예: http://localhost:11434/v1"),
    model: str = typer.Option(..., "--model", help="엔진에 요청할 모델 이름"),
    concurrency: int = typer.Option(..., "--concurrency", min=1, help="동시 요청 수"),
    input_tokens: int = typer.Option(..., "--in", min=1, help="입력 길이 (엔진이 센 프롬프트 토큰, chat template 포함)"),
    output_tokens: int = typer.Option(..., "--out", min=1, help="출력 길이 = max_tokens"),
    seed: int | None = typer.Option(
        None, "--seed",
        help="프롬프트·샘플링 난수 시드. 기본: 매번 무작위로 뽑아 레코드에 기록 "
        "(같은 시드를 다시 쓰면 엔진 prefix 캐시에 이전 실행의 프롬프트가 남아 있을 수 있다)",
    ),
    warmup: int | None = typer.Option(None, "--warmup", min=0, help="워밍업 요청 수. 기본: concurrency"),
    requests: int | None = typer.Option(None, "--requests", min=1, help="측정 요청 수. 기본: concurrency × 5"),
    ignore_eos: bool = typer.Option(False, "--ignore-eos", help="EOS 무시하고 --out 까지 생성 (vLLM·SGLang 만 적용)"),
    out_file: Path | None = typer.Option(None, "--out-file", help="레코드 JSON 저장 경로. 없으면 stdout"),
    hf_id: str | None = typer.Option(None, "--hf-id", help="HF 모델 ID (토크나이저·메타데이터). 기본: config [hf_ids] 또는 --model"),
    submitter: str | None = typer.Option(None, "--submitter"),
    provenance: str | None = typer.Option(None, "--provenance", help="self | community | scraped | vendor"),
    environment: str | None = typer.Option(None, "--environment", help="cloud_rented | on_prem"),
    provider: str | None = typer.Option(None, "--provider", help="cloud_rented 일 때 클라우드 제공자"),
    batch_id: str | None = typer.Option(None, "--batch-id"),
    tp: int | None = typer.Option(None, "--tp", min=1, help="tensor parallel (GPU 2장 이상이면 필수)"),
    pp: int | None = typer.Option(None, "--pp", min=1, help="pipeline parallel (GPU 2장 이상이면 필수)"),
    quantization: str | None = typer.Option(None, "--quantization", help="자동 감지 대신 쓸 양자화 방식"),
    weight_dtype: str | None = typer.Option(None, "--weight-dtype", help="자동 감지 대신 쓸 가중치 dtype"),
    model_load_state: str | None = typer.Option(None, "--model-load-state", help="cold | warm (Ollama 외 엔진은 필수)"),
) -> None:
    """격자 셀 1개를 측정해 스키마 레코드 JSON 1개를 낸다."""
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

    if seed is None:
        seed = secrets.randbelow(2**31)
    spec = CellSpec(
        url=url, model=model, hf_id=hf_id, concurrency=concurrency,
        input_tokens=input_tokens, output_tokens=output_tokens,
        submitter=submitter, provenance=provenance, environment=environment, provider=provider,
        seed=seed, warmup_count=warmup, request_count=requests, ignore_eos=ignore_eos,
        batch_id=batch_id, tensor_parallel=tp, pipeline_parallel=pp,
        quantization=quantization, weight_dtype=weight_dtype, model_load_state=model_load_state,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SchemaWarning)
        try:
            record = asyncio.run(run_cell(spec))
        except (CellError, NotReadyError, TemplateMismatchError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1) from e
    for w in caught:
        if issubclass(w.category, SchemaWarning):
            typer.echo(f"warning: {w.message}", err=True)

    r = record.result

    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.1f}"

    typer.echo(
        f"ok: {record.run_id}  ttft p95 {fmt(r.ttft_p95_ms)} ms  tpot p50 {fmt(r.tpot_p50_ms)} ms  "
        f"out tps {fmt(r.output_tps_total)}  errors {r.error_count}",
        err=True,
    )
    text = record.model_dump_json(indent=2) + "\n"
    if out_file is None:
        typer.echo(text, nl=False)
    else:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out_file}", err=True)


if __name__ == "__main__":
    app()
