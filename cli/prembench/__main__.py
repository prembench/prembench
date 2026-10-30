import json
from pathlib import Path

import typer

from prembench import __version__
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


if __name__ == "__main__":
    app()
