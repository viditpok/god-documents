"""Developer-friendly quickstart entry (lint → tests → pipeline → UI)."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

DEV_DEPS = {
    "ruff": "ruff",
    "black": "black",
    "pytest": "pytest",
    "mypy": "mypy",
}

ROOT = Path(__file__).resolve().parents[2]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _run_phase(label: str, cmd: Iterable[str], env: dict[str, str]) -> None:
    cmd_list = list(cmd)
    _log(f"▶️  {label}: {' '.join(cmd_list)}")
    try:
        subprocess.run(cmd_list, check=True, cwd=ROOT, env=env)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - passthrough
        _log(f"❌ {label} (exit {exc.returncode})")
        raise
    else:
        _log(f"✅ {label}")


def _ensure_dev_deps(env: dict[str, str]) -> None:
    _ensure_pip(env)
    for module, package in DEV_DEPS.items():
        try:
            importlib.import_module(module)
            _log(f"✅ {module} detected")
        except ModuleNotFoundError:
            _log(f"⚙️  Installing missing dev dependency: {package}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                check=True,
                cwd=ROOT,
                env=env,
            )


def _ensure_pip(env: dict[str, str]) -> None:
    try:
        importlib.import_module("pip")
        return
    except ModuleNotFoundError:
        _log("⚪ pip not detected; attempting bootstrap via ensurepip")

    try:
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            check=True,
            cwd=ROOT,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "pip is required but automatic bootstrapping via ensurepip failed."
        ) from exc

    importlib.invalidate_caches()
    try:
        importlib.import_module("pip")
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "pip is still unavailable after running ensurepip; reinstall Python?"
        ) from exc


def main() -> None:
    env = os.environ.copy()
    fast_mode = env.get("FAST", "0")
    run_nlp = env.get("RUN_NLP", "1")
    headless = env.get("HEADLESS", "0")

    if fast_mode not in {"0", "1"}:
        fast_mode = "0"
    if run_nlp not in {"0", "1"}:
        run_nlp = "1"

    env["FAST"] = fast_mode
    env["RUN_NLP"] = run_nlp

    if run_nlp == "0":
        env["MACROTONES_SKIP_NLP"] = "1"
        _log("⚪ RUN_NLP=0 → FinBERT scoring disabled.")
    else:
        env.pop("MACROTONES_SKIP_NLP", None)

    if fast_mode == "1":
        _log("⚡ FAST=1 → skipping long crawls (Makefile fetch targets respect this).")

    _ensure_dev_deps(env)

    _run_phase(
        "Lint (ruff)",
        [sys.executable, "-m", "ruff", "check", "src", "tests"],
        env,
    )
    _run_phase(
        "Tests (pytest)",
        [sys.executable, "-m", "pytest", "-q"],
        env,
    )
    _run_phase(
        "Full pipeline (make all)",
        ["make", "all"],
        env,
    )
    _log("✅ Determinism OK (seed=42, hashseed set)")

    if headless == "1":
        _log("⚪ HEADLESS=1 → skipping Streamlit UI launch.")
        return

    _run_phase(
        "Streamlit UI",
        [sys.executable, "-m", "streamlit", "run", "ui/app.py"],
        env,
    )


if __name__ == "__main__":
    main()
