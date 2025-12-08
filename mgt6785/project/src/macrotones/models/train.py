from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from macrotones.config.schema import ConfigModel, load_config
from macrotones.models.baselines import fit_ridge, predict_ridge
from macrotones.utils.seed import DEFAULT_SEED, set_global_seed

CONFIG_PATH = Path("config/project.yaml")
TARGETS = ["HML", "SMB", "UMD"]
MIN_TRAIN_MONTHS = 24
INNER_TEST_WINDOW = 6


def _purged_splits(
    n_obs: int,
    purge: int,
    embargo: int,
    min_train: int = MIN_TRAIN_MONTHS,
    test_window: int = INNER_TEST_WINDOW,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if n_obs <= min_train + test_window:
        return
    start = min_train
    while start + test_window <= n_obs:
        test_idx = np.arange(start, start + test_window)
        train_mask = np.ones(n_obs, dtype=bool)
        train_mask[test_idx] = False
        if purge > 0:
            purge_start = max(0, start - purge)
            train_mask[purge_start:start] = False
        if embargo > 0:
            embargo_end = min(n_obs, start + test_window + embargo)
            train_mask[start + test_window : embargo_end] = False
        train_idx = np.where(train_mask)[0]
        if len(train_idx) < min_train:
            break
        yield train_idx, test_idx
        start += test_window


def crossval_select(
    X: pd.DataFrame,
    y: pd.DataFrame,
    alphas: list[float],
    seed: int,
    purge: int,
    embargo: int,
) -> float:
    """Select the ridge alpha that minimises purged validation MSE."""
    if not alphas:
        raise ValueError("alpha grid must contain at least one value.")
    cleaned = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[cleaned]
    y = y.loc[cleaned, TARGETS].astype(float)
    n_obs = len(X)
    if n_obs < MIN_TRAIN_MONTHS + INNER_TEST_WINDOW:
        return float(alphas[0])

    splits = list(_purged_splits(n_obs, purge, embargo))
    if not splits:
        return float(alphas[0])

    alpha_grid = [float(a) for a in sorted(set(alphas))]
    best_alpha = alpha_grid[0]
    best_loss = float("inf")
    rng = np.random.default_rng(seed)

    for alpha in alpha_grid:
        losses: list[float] = []
        for train_idx, test_idx in splits:
            Xtr_fold = X.iloc[train_idx]
            Xte_fold = X.iloc[test_idx]
            non_nan_cols = Xtr_fold.columns[Xtr_fold.notna().any(axis=0)]
            Xtr_fold = Xtr_fold[non_nan_cols]
            Xte_fold = Xte_fold[non_nan_cols]
            for target in TARGETS:
                ytr = y[target].iloc[train_idx]
                yte = y[target].iloc[test_idx]
                model = fit_ridge(Xtr_fold, ytr, alpha=alpha)
                preds = predict_ridge(model, Xte_fold)
                diff = yte.values - preds
                losses.append(float(np.mean(diff**2)))
        avg_loss = float(np.mean(losses)) if losses else float("inf")
        jitter = rng.random() * 1e-9
        if avg_loss + jitter < best_loss:
            best_loss = avg_loss
            best_alpha = alpha

    return float(best_alpha)


def expanding_splits(
    n: int, min_train: int = 120, test_size: int = 1
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for test_end in range(min_train, n - test_size + 1):
        train_idx = np.arange(0, test_end)
        test_idx = np.arange(test_end, test_end + test_size)
        yield train_idx, test_idx


def _default_ridge_alpha(config: ConfigModel, override: float | None = None) -> float:
    if override is not None:
        return override
    if config.model.ridge_alpha is not None:
        return float(config.model.ridge_alpha)
    return 3.0


def _run_training(
    config: ConfigModel,
    ridge_alpha: float | None = None,
    capture_coeffs: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    processed_dir = Path(config.data.out_processed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(processed_dir / "panel.parquet")
    X = panel.drop(columns=[*TARGETS, "Mkt_RF", "RF"])
    Y = panel[TARGETS]

    feature_columns = list(X.columns)
    idx = X.index
    alpha = _default_ridge_alpha(config, ridge_alpha)

    preds = pd.DataFrame(index=idx, columns=TARGETS, dtype=float)
    coeff_history: dict[str, list[pd.Series]] = {target: [] for target in TARGETS}
    coeff_dates: dict[str, list[pd.Timestamp]] = {target: [] for target in TARGETS}

    n_samples = len(X)
    test_size = 1
    if n_samples <= test_size:
        msg = (
            "Panel has insufficient observations "
            f"({n_samples}) for train/test split."
        )
        raise ValueError(msg)

    max_train_len = max(test_size, n_samples - test_size)
    effective_min_train = min(config.model.min_train_months, max_train_len)

    min_buffer_obs = 12
    min_reasonable_train = max(test_size + 4, 5)
    if n_samples - effective_min_train < min_buffer_obs:
        candidate = max(min_reasonable_train, n_samples - min_buffer_obs)
        effective_min_train = max(test_size, min(candidate, max_train_len))

    if effective_min_train < config.model.min_train_months:
        warnings.warn(
            "Reducing `min_train_months` from "
            f"{config.model.min_train_months} to {effective_min_train} "
            "to accommodate short sample size.",
            RuntimeWarning,
            stacklevel=2,
        )

    splits = expanding_splits(n_samples, effective_min_train, test_size)

    for ttrain, ttest in splits:
        Xtr, Xte = X.iloc[ttrain], X.iloc[ttest]
        non_all_nan_cols = Xtr.columns[Xtr.notna().any(axis=0)]
        Xtr = Xtr[non_all_nan_cols]
        Xte = Xte[non_all_nan_cols]
        current_date = idx[ttest[0]]

        for ycol in TARGETS:
            ytr = Y[ycol].iloc[ttrain]

            ridge_model = fit_ridge(Xtr, ytr, alpha=alpha)
            ridge_pred = predict_ridge(ridge_model, Xte)

            model = XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
            model.fit(Xtr, ytr)
            xgb_pred = float(model.predict(Xte)[0])

            preds.loc[current_date, ycol] = 0.5 * ridge_pred + 0.5 * xgb_pred

            if capture_coeffs:
                coef_series = pd.Series(0.0, index=feature_columns, dtype=float)
                ridge_coefs = ridge_model.named_steps["ridge"].coef_
                base_len = len(non_all_nan_cols)
                # SimpleImputer with add_indicator=True appends indicator columns;
                # truncate so we only map back to original feature names.
                coef_series.loc[non_all_nan_cols] = ridge_coefs[:base_len]
                coeff_history[ycol].append(coef_series)
                coeff_dates[ycol].append(current_date)

    out = pd.concat([preds, panel[["Mkt_RF", config.project.rf_col]]], axis=1).dropna()

    coeff_frames: dict[str, pd.DataFrame] = {}
    if capture_coeffs:
        for target, series_list in coeff_history.items():
            if not series_list:
                continue
            df = pd.DataFrame(
                series_list,
                index=pd.Index(coeff_dates[target], name="date"),
            ).sort_index()
            coeff_frames[target] = df

    return out, coeff_frames


def save_predictions(config: ConfigModel, preds: pd.DataFrame) -> None:
    processed_dir = Path(config.data.out_processed)
    out_path = processed_dir / "preds.parquet"
    preds.to_parquet(out_path)
    print(f"Saved predictions: {preds.shape} -> {out_path}")

def save_regimes(config: ConfigModel, regimes: pd.DataFrame) -> None:
    processed_dir = Path(config.data.out_processed)
    out_path = processed_dir / "regimes.parquet"
    regimes.to_parquet(out_path)
    print(f"Saved regimes: {regimes.shape} -> {out_path}")


def plot_ridge_coeffs(
    config: ConfigModel, coeff_frames: dict[str, pd.DataFrame]
) -> None:
    if not coeff_frames:
        return

    processed_dir = Path(config.data.out_processed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    top_n = 5
    window = 12
    n_targets = len(coeff_frames)
    fig, axes = plt.subplots(n_targets, 1, figsize=(10, 3 * n_targets), sharex=True)
    if n_targets == 1:
        axes = [axes]

    for ax, (target, df) in zip(axes, coeff_frames.items(), strict=False):
        df = df.sort_index()
        if df.empty:
            continue
        top_features = df.abs().mean().nlargest(top_n).index
        rolling = (
            df[top_features]
            .rolling(window=window, min_periods=max(1, window // 3))
            .mean()
        )
        rolling.plot(ax=ax)
        ax.set_title(f"Ridge coefficients (rolling {window}m) - {target}")
        ax.set_ylabel("Coefficient")
        ax.legend(loc="best")

    axes[-1].set_xlabel("Date")
    output_path = processed_dir / "ridge_coeffs.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved ridge coefficient plot -> {output_path}")


def train_models(
    ridge_alpha: float | None = None,
    capture_coeffs: bool = False,
    save_outputs: bool = True,
    config_path: Path = CONFIG_PATH,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    config = load_config(config_path)
    set_global_seed(DEFAULT_SEED)
    preds, coeff_frames = _run_training(
        config=config, ridge_alpha=ridge_alpha, capture_coeffs=capture_coeffs
    )
    if save_outputs:
        save_predictions(config, preds)
    return preds, coeff_frames


def main() -> None:
    config = load_config(CONFIG_PATH)
    set_global_seed(DEFAULT_SEED)
    _preds, coeff_frames = train_models(
        capture_coeffs=True, save_outputs=True
    )
    plot_ridge_coeffs(config, coeff_frames)


if __name__ == "__main__":
    main()
