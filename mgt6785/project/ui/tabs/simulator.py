from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from macrotones.fusion.llm_allocator import generate_policy
from ui.components import show_plotly


def _format_macro_inputs(baseline_macro: dict[str, float]) -> pd.DataFrame:
    display_keys = {
        "cpi_yoy": "CPI YoY",
        "unemp": "Unemployment",
        "y10y_2y_spread": "10Y-2Y Spread",
        "mri": "Macro Regime Index",
    }
    data = {
        label: baseline_macro.get(key)
        for key, label in display_keys.items()
        if key in baseline_macro
    }
    return pd.DataFrame([data])


PRESETS = OrderedDict(
    {
        "Inflation Shock": {"CPI": 1.0, "UNRATE": -0.5, "T10Y2Y": 0.2},
        "Growth Scare": {"CPI": -0.5, "UNRATE": 1.0, "T10Y2Y": -0.3},
        "Curve Steepener": {"CPI": 0.0, "UNRATE": -0.5, "T10Y2Y": 0.8},
    }
)
LOG_PATH = Path(".cache") / "simulator_runs.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _log_run(
    preset: str,
    baseline_macro: dict[str, float],
    macro_adj: dict[str, float],
    weights: pd.Series,
    turnover: float,
    cost: float,
    delta_sharpe: float,
    delta_vol: float,
    delta_weights: pd.Series,
) -> None:
    row = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "preset": preset,
        "macro_inputs": json.dumps(
            {"baseline": baseline_macro, "counterfactual": macro_adj},
            default=float,
        ),
        "new_weights": json.dumps(weights.to_dict(), default=float),
        "turnover": float(turnover),
        "cost": float(cost),
        "delta_sharpe": float(delta_sharpe),
        "delta_vol": float(delta_vol),
        "delta_weights": json.dumps(delta_weights.to_dict(), default=float),
    }
    df = pd.DataFrame([row])
    header = not LOG_PATH.exists()
    df.to_csv(LOG_PATH, mode="a", header=header, index=False)
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = logs_dir / f"simulator_{stamp}.json"
    json_path.write_text(json.dumps(row, default=float), encoding="utf-8")
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
    json_path = logs_dir / f"simulator_{stamp}.json"
    json_path.write_text(json.dumps(row, default=float), encoding="utf-8")


def _load_history() -> pd.DataFrame | None:
    if not LOG_PATH.exists():
        return None
    df = pd.read_csv(LOG_PATH, on_bad_lines="skip", engine="python")
    if df.empty:
        return None
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    return df.dropna(subset=["turnover"])


def _tornado_series(
    delta_sharpe: float,
    macro_delta_series: pd.Series,
    weights_grad: pd.Series,
) -> pd.Series:
    sigma = macro_delta_series.abs().replace(0, np.nan)
    sharpe_per_sigma: dict[str, float] = {}
    for key, grad in weights_grad.items():
        denom = sigma.get(key, np.nan)
        if np.isnan(denom):
            sharpe_per_sigma[key] = 0.0
        else:
            sharpe_per_sigma[key] = abs(delta_sharpe) / max(float(denom), 1e-6)
    series = pd.Series(sharpe_per_sigma).replace([np.inf, -np.inf], np.nan).dropna()
    return series.sort_values(ascending=True)


def _get_slider_value(key: str, label: str, min_val: float, max_val: float, step: float) -> float:
    if key not in st.session_state:
        st.session_state[key] = 0.0
    return st.slider(label, min_val, max_val, st.session_state[key], step, key=key)


def _apply_preset(preset: dict[str, float], slider_keys: dict[str, str]) -> None:
    for macro_key, delta in preset.items():
        state_key = slider_keys.get(macro_key)
        if state_key is not None:
            st.session_state[state_key] = delta


def render(
    baseline_macro: dict[str, float] | None,
    latest_nlp: dict[str, float] | None,
    baseline_weights: dict[str, float] | None,
    latest_returns: pd.Series | None = None,
    cost_bps: float = 10.0,
) -> None:
    st.title("Counterfactual Policy Simulator")
    st.caption(
        "Adjust macro surprises to see how the LLM allocator would reweight "
        "factors relative to the latest baseline."
    )
    if not baseline_macro or not latest_nlp:
        st.info(
            "Simulator requires macro features and NLP scores. "
            "Run the pipeline first (``make quickstart``)."
        )
        return

    slider_keys = {
        "CPI": "sim_cpi_delta",
        "UNRATE": "sim_unemp_delta",
        "T10Y2Y": "sim_term_delta",
    }

    st.session_state.setdefault("sim_selected_preset", "Manual")
    preset_cols = st.columns(len(PRESETS))
    for col, (name, adjustments) in zip(preset_cols, PRESETS.items(), strict=False):
        if col.button(name):
            _apply_preset(adjustments, slider_keys)
            st.session_state["sim_selected_preset"] = name
    preset_label = st.session_state.get("sim_selected_preset", "Manual")

    cpi_delta = _get_slider_value(slider_keys["CPI"], "CPI YoY Δ (%)", -2.0, 2.0, 0.1)
    unemp_delta = _get_slider_value(
        slider_keys["UNRATE"], "Unemployment Δ (%)", -2.0, 2.0, 0.1
    )
    term_delta = _get_slider_value(
        slider_keys["T10Y2Y"], "10y-2y Spread Δ (%)", -1.0, 1.0, 0.05
    )

    composite = st.checkbox("Composite Shock (risk-off blend)", value=False)
    if composite:
        cpi_delta = np.clip(cpi_delta + 0.6, -2.0, 2.0)
        unemp_delta = np.clip(unemp_delta + 0.8, -2.0, 2.0)
        term_delta = np.clip(term_delta - 0.4, -1.0, 1.0)

    macro_adj = baseline_macro.copy()
    macro_adj["cpi_yoy"] = macro_adj.get("cpi_yoy", 0.0) + cpi_delta
    macro_adj["unemp"] = macro_adj.get("unemp", 0.0) + unemp_delta
    macro_adj["y10y_2y_spread"] = macro_adj.get("y10y_2y_spread", 0.0) + term_delta
    macro_adj["mri"] = (
        macro_adj.get("mri", 0.0)
        + 0.1 * cpi_delta
        - 0.2 * unemp_delta
        + 0.35 * term_delta
    )

    st.write("### Adjusted Macro Inputs")
    st.dataframe(_format_macro_inputs(macro_adj), width="stretch")

    llm_new = generate_policy(macro_adj, latest_nlp)
    weights_series = pd.Series(llm_new, dtype=float)

    old_series = (
        pd.Series(baseline_weights, dtype=float).reindex(weights_series.index).fillna(0.0)
        if baseline_weights
        else pd.Series(0.0, index=weights_series.index, dtype=float)
    )
    delta_weights = weights_series - old_series

    fig = go.Figure(
        data=[
            go.Bar(
                x=weights_series.index,
                y=weights_series.values,
                marker_color="#a855f7",
                name="Counterfactual",
            )
        ]
    )
    fig.update_layout(title="Counterfactual Factor Weights")
    show_plotly(fig)

    turnover = float(delta_weights.abs().sum()) / 2.0
    cost = turnover * cost_bps * 1e-4
    if turnover > 0.5:
        st.warning("High turnover scenario (> 50 % of portfolio reallocation)")
    aligned_returns = None
    if latest_returns is not None:
        aligned_returns = latest_returns.reindex(weights_series.index).fillna(0.0)
    if aligned_returns is not None:
        delta_sharpe = float(delta_weights.dot(aligned_returns)) * np.sqrt(12)
    else:
        delta_sharpe = 0.0
    delta_vol = float(np.sqrt((delta_weights**2).sum()))

    if baseline_weights:
        compare = st.checkbox("Compare vs Baseline", value=True)
        if compare:
            base_series = pd.Series(baseline_weights, dtype=float).reindex(
                weights_series.index
            ).fillna(0.0)
            delta = weights_series - base_series
            delta_fig = go.Figure(
                data=[
                    go.Bar(
                        x=delta.index,
                        y=delta.values,
                        marker_color="#f97316",
                        name="Δ Weight",
                    )
                ]
            )
            delta_fig.update_layout(title="Change vs Baseline Weights")
            show_plotly(delta_fig)
            export_table = pd.DataFrame(
                {
                    "factor": weights_series.index,
                    "baseline": base_series.values,
                    "counterfactual": weights_series.values,
                    "delta": delta.values,
                }
            )
            export_table.loc[len(export_table)] = [
                "|Δ| sum",
                base_series.abs().sum(),
                weights_series.abs().sum(),
                delta.abs().sum(),
            ]
            export_table.loc[len(export_table)] = [
                "ΔSharpe",
                "",
                "",
                delta_sharpe,
            ]
            export_table.loc[len(export_table)] = [
                "ΔVol",
                "",
                "",
                delta_vol,
            ]
            export_table.loc[len(export_table)] = [
                "Turnover",
                "",
                "",
                turnover,
            ]
            st.download_button(
                "📄 Download Comparison Table",
                data=export_table.to_csv(index=False),
                file_name="simulator_comparison.csv",
            )
    else:
        st.info("Baseline weights unavailable - turnover comparison uses zero vector.")
    st.info(f"Expected turnover: {turnover:.2%} | Cost impact: {cost:.3%}")
    st.caption(f"ΔSharpe ≈ {delta_sharpe:.3f} | ΔVol ≈ {delta_vol:.2%}")

    try:
        replay = generate_policy(macro_adj, latest_nlp)
        replay_series = pd.Series(replay, dtype=float).reindex(weights_series.index)
        np.testing.assert_allclose(
            replay_series.values,
            weights_series.values,
            rtol=1e-5,
            atol=1e-7,
        )
    except AssertionError:
        st.error("Simulator output deviates from policy function evaluation.")
    except Exception:
        st.warning("Policy consistency check skipped due to API constraints.")

    _log_run(
        preset_label,
        baseline_macro,
        macro_adj,
        weights_series,
        turnover,
        cost,
        delta_sharpe,
        delta_vol,
        delta_weights,
    )

    cap = 0.4
    if (weights_series.abs() > cap).any():
        st.warning("Weight cap exceeded (40 %)")

    macro_delta_series = pd.Series(
        {"CPI": cpi_delta, "UNRATE": unemp_delta, "T10Y2Y": term_delta},
        dtype=float,
    )
    total_macro = macro_delta_series.abs().sum()
    total_weight_shift = float(delta_weights.abs().sum())
    if total_macro > 0 and total_weight_shift > 0:
        weights_grad = (
            macro_delta_series.abs() / total_macro * total_weight_shift
        )
    else:
        weights_grad = macro_delta_series.abs()
    sens = _tornado_series(delta_sharpe, macro_delta_series, weights_grad)
    sens_fig = px.bar(
        sens,
        x=sens.values if not sens.empty else [],
        y=sens.index if not sens.empty else [],
        orientation="h",
        title="Tornado: ΔSharpe per 1σ Shock",
        color=sens.values if not sens.empty else None,
        color_continuous_scale="Plasma",
    )
    sens_fig.update_layout(xaxis_title="Δ Sharpe / σ", yaxis_title="")
    show_plotly(sens_fig)

    history = _load_history()
    if history is not None and len(history) >= 5:
        hist_fig = px.histogram(
            history,
            x="turnover",
            nbins=10,
            title="Simulator Turnover Histogram",
            color_discrete_sequence=["#60a5fa"],
        )
        show_plotly(hist_fig)
