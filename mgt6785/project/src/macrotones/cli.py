from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import typer
import yaml

from macrotones.config.schema import ConfigModel, load_config

app = typer.Typer(help="MacroTone local workflow helper.")

CONFIG_OPTION = typer.Option(
    Path("config/project.yaml"),
    "--config",
    "-c",
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Path to project configuration YAML.",
)


def _format_errors(errors: Iterable[dict]) -> list[str]:
    formatted = []
    for err in errors:
        location = ".".join(str(part) for part in err.get("loc", []))
        message = err.get("msg", "invalid value")
        formatted.append(f"{location or '<root>'}: {message}")
    return formatted


def _load_and_print(config_path: Path) -> ConfigModel:
    try:
        cfg = load_config(config_path)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(code=1) from exc
    except yaml.YAMLError as exc:
        typer.secho(f"Failed to parse {config_path}: {exc}", fg="red", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        if hasattr(exc, "errors"):
            typer.secho("Configuration validation failed:", fg="red", err=True)
            for line in _format_errors(exc.errors()):
                typer.echo(f"  - {line}", err=True)
        else:
            typer.secho(f"Unexpected error: {exc}", fg="red", err=True)
        raise typer.Exit(code=1) from exc

    typer.secho("Effective configuration:", fg="cyan")
    typer.echo(
        yaml.safe_dump(
            cfg.model_dump(by_alias=True, exclude_none=True),
            sort_keys=False,
        ).rstrip()
    )
    return cfg


def _run(command: Callable[[], None]) -> None:
    command()


@app.command()
def crawl(config: Path = CONFIG_OPTION) -> None:
    """Fetch FOMC source documents."""

    _load_and_print(config)
    from macrotones.data.fetch_fomc import main as fetch_fomc

    _run(fetch_fomc)


@app.command()
def score(config: Path = CONFIG_OPTION) -> None:
    """Run FinBERT scoring pipeline."""

    _load_and_print(config)
    from macrotones.nlp.finbert_scoring import main as score_fomc

    _run(score_fomc)


@app.command()
def features(config: Path = CONFIG_OPTION) -> None:
    """Build feature dataset."""

    _load_and_print(config)
    from macrotones.features.dataset import main as build_features

    _run(build_features)


@app.command()
def train(config: Path = CONFIG_OPTION) -> None:
    """Train predictive models."""

    _load_and_print(config)
    from macrotones.models.train import main as train_models

    _run(train_models)


@app.command()
def backtest(config: Path = CONFIG_OPTION) -> None:
    """Run backtest engine."""

    _load_and_print(config)
    from macrotones.backtest.engine import main as run_backtest

    _run(run_backtest)


@app.command()
def report(config: Path = CONFIG_OPTION) -> None:
    """Generate backtest reports."""

    _load_and_print(config)
    from macrotones.backtest.reports import main as run_reports

    _run(run_reports)


@app.command()
def diag(config: Path = CONFIG_OPTION) -> None:
    """Generate diagnostics."""

    _load_and_print(config)
    from macrotones.backtest.diagnostics import main as run_diag

    _run(run_diag)


@app.command()
def walkforward(config: Path = CONFIG_OPTION) -> None:
    """Run walk-forward analysis."""

    _load_and_print(config)
    from macrotones.backtest.walkforward import main as run_walkforward

    _run(run_walkforward)


@app.command()
def costs(config: Path = CONFIG_OPTION) -> None:
    """Sweep trading cost assumptions."""

    _load_and_print(config)
    from macrotones.backtest.sweep_costs import main as run_costs

    _run(run_costs)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
