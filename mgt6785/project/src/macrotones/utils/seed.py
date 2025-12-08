"""Deterministic seeding utilities for MacroTone."""

from __future__ import annotations

import os
import random
from importlib import import_module
from types import ModuleType
from typing import Any, Final, cast

import numpy as np

DEFAULT_SEED: Final[int] = 42


def _load_module(path: str) -> ModuleType | None:
    try:  # pragma: no cover - optional dependency
        return import_module(path)
    except Exception:  # pragma: no cover
        return None


TORCH = _load_module("torch")
XGB_RANDOM = _load_module("xgboost.random")
SKLEARN_RANDOM = _load_module("sklearn.utils._random")


def _seed_torch(seed: int) -> None:
    if TORCH is None:
        return
    manual_seed = getattr(TORCH, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)
    cuda = getattr(TORCH, "cuda", None)
    if cuda is None:
        return
    is_available = getattr(cuda, "is_available", None)
    manual_seed_all = getattr(cuda, "manual_seed_all", None)
    if callable(is_available) and is_available():  # pragma: no cover - GPU optional
        if callable(manual_seed_all):
            manual_seed_all(seed)


def _seed_xgboost(seed: int) -> None:
    if XGB_RANDOM is None:
        return
    set_seed = getattr(XGB_RANDOM, "set_seed", None)
    if callable(set_seed):
        try:
            set_seed(seed)
        except Exception:  # pragma: no cover - safeguard against API drift
            pass


def _seed_sklearn(seed: int) -> None:
    if SKLEARN_RANDOM is None:
        return
    try:
        rng = np.random.RandomState(seed)
        module_any = cast(Any, SKLEARN_RANDOM)
        if hasattr(SKLEARN_RANDOM, "_random_state"):
            module_any._random_state = rng
        if hasattr(SKLEARN_RANDOM, "_rand"):
            module_any._rand = rng
        if hasattr(SKLEARN_RANDOM, "GLOBAL_RANDOM_STATE"):
            module_any.GLOBAL_RANDOM_STATE = np.random.default_rng(seed)
    except Exception:  # pragma: no cover - sklearn internals can change
        pass


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python, NumPy, scikit-learn, Torch, and XGBoost utilities."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    _seed_torch(seed)
    _seed_xgboost(seed)
    _seed_sklearn(seed)

    return seed


__all__ = ["DEFAULT_SEED", "set_global_seed"]
