import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from prembench import __version__
from prembench.probe import as_dict, probe_engine, probe_host
from prembench.schema import json_schema

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


if __name__ == "__main__":
    app()
