from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:  # noqa: F401  # pragma: no cover - optional dependency
    import openai  # type: ignore
except ImportError:  # pragma: no cover - runtime optional
    openai = None

from macrotones.datahub import cache_regime_correlation, load_regime_correlation
from macrotones.diagnostics.attribution import factor_attribution
from macrotones.utils.narrative import generate_llm_summary
from ui.components import show_plotly

SIGNIFICANCE_COLORS = {
    "p < 0.05": "#22c55e",
    "n.s.": "#94a3b8",
}


def _alpha_beta_section(alpha_df: pd.DataFrame) -> None:
    if alpha_df.empty:
        st.info("Alpha decomposition artifact missing.")
        return
    df = alpha_df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df["alpha_ann"] = df["alpha"] * 12
    alpha_fig = px.line(
        df,
        y="alpha_ann",
        title="Rolling α (annualized, 36M)",
        markers=False,
        labels={"value": "Alpha"},
    )
    alpha_fig.update_layout(xaxis_title="Date", yaxis_title="α (annualized)")
    show_plotly(alpha_fig)

    beta_cols = [col for col in df.columns if col.startswith("beta_")]
    if beta_cols:
        beta_frame = df[beta_cols].rename(
            columns=lambda col: col.replace("beta_", "")
        )
        beta_fig = px.line(beta_frame, title="Rolling β Exposures")
        beta_fig.update_layout(xaxis_title="Date", yaxis_title="β")
        show_plotly(beta_fig)

    contrib_cols = [col for col in df.columns if col.startswith("contrib_")]
    if contrib_cols and st.checkbox("Show factor contributions", key="contrib_toggle"):
        contrib_frame = df[contrib_cols].rename(
            columns=lambda col: col.replace("contrib_", "")
        )
        contrib_fig = px.area(
            contrib_frame,
            title="Average Factor Contributions (36M window)",
        )
        contrib_fig.update_layout(xaxis_title="Date", yaxis_title="Contribution")
        show_plotly(contrib_fig)


def _tracking_error_section(tracking_df: pd.DataFrame) -> None:
    if tracking_df.empty:
        st.info("Tracking error artifact missing.")
        return
    df = tracking_df.copy()
    df.index = pd.to_datetime(df.index)
    te_fig = px.line(
        df,
        y="tracking_error",
        title="Rolling Tracking Error (12M)",
        color_discrete_sequence=["#f87171"],
    )
    te_fig.update_layout(xaxis_title="Date", yaxis_title="Tracking Error")
    show_plotly(te_fig)

    corr_fig = go.Figure()
    if "rolling_corr" in df.columns:
        corr_fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["rolling_corr"],
                name="Corr(Strategy, Market)",
                line={"color": "#60a5fa"},
            )
        )
    if "information_ratio" in df.columns:
        corr_fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["information_ratio"],
                name="Information Ratio",
                line={"color": "#34d399"},
                yaxis="y2",
            )
        )
        corr_fig.update_layout(
            yaxis2=dict(
                title="Information Ratio",
                overlaying="y",
                side="right",
            )
        )
    corr_fig.update_layout(
        title="Rolling Correlation & Information Ratio",
        xaxis_title="Date",
        yaxis_title="Correlation",
    )
    show_plotly(corr_fig)


def _excess_chart(excess_df: pd.DataFrame) -> None:
    if excess_df.empty:
        st.info("Excess vs market parquet missing.")
        return
    df = excess_df.copy()
    df.index = pd.to_datetime(df.index)
    cum_excess = df["excess_vs_market"].cumsum()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cum_excess.index,
            y=cum_excess.values,
            name="Cumulative Excess",
            line={"color": "#fde047"},
        )
    )
    fig.update_layout(
        title="Cumulative Excess Return (Strategy – Market)",
        xaxis_title="Date",
        yaxis_title="Cum Excess (bps)",
    )
    show_plotly(fig)


def _drawdown_section(drawdown_df: pd.DataFrame) -> None:
    if drawdown_df.empty:
        st.info("Drawdown attribution file missing.")
        return
    st.dataframe(drawdown_df, width="stretch")
    pivot = (
        drawdown_df.pivot_table(
            index="event",
            columns="factor",
            values="contribution",
            aggfunc="sum",
        )
        .fillna(0.0)
        .reset_index()
    )
    long_df = pivot.melt("event", var_name="factor", value_name="contribution")
    fig = px.bar(
        long_df,
        x="event",
        y="contribution",
        color="factor",
        title="Drawdown Attribution by Factor",
    )
    fig.update_layout(xaxis_title="Drawdown (start)", yaxis_title="P&L Contribution")
    show_plotly(fig)


def _prepare_ic_frame(ic_summary: pd.DataFrame) -> pd.DataFrame:
    df = ic_summary.copy()
    if "factor" not in df.columns and df.index.name == "factor":
        df = df.reset_index()
    if "factor" not in df.columns:
        df = df.rename_axis("factor").reset_index()
    df = df.rename(columns={"pearson_ic": "IC"})
    if "significant" not in df.columns and "pearson_pvalue" in df.columns:
        df["significant"] = df["pearson_pvalue"] < 0.05
    if "significant" not in df.columns:
        df["significant"] = False
    df["significance_label"] = df["significant"].map(
        {True: "p < 0.05", False: "n.s."}
    )
    if "pearson_pvalue" in df.columns:
        df["pvalue_display"] = (
            df["pearson_pvalue"].fillna(1.0).clip(upper=0.9).round(3)
        )
    else:
        df["pvalue_display"] = "n/a"
    if "pearson_pvalue" in df.columns:
        df["Signal Quality"] = np.where(
            (df["IC"].abs() < 0.1) | (df["pearson_pvalue"] > 0.05),
            "Weak",
            "Strong",
        )
    else:
        df["Signal Quality"] = np.where(df["IC"].abs() < 0.1, "Weak", "Strong")
    return df


def quintile_portfolios(signal: pd.Series, fwd_ret: pd.Series) -> pd.Series:
    df = pd.DataFrame({"signal": signal, "fwd": fwd_ret}).dropna()
    if df.empty or df["signal"].nunique() < 5:
        return pd.Series(dtype=float)
    try:
        buckets = pd.qcut(df["signal"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    except ValueError:
        return pd.Series(dtype=float)
    return df.groupby(buckets)["fwd"].mean()


def _quintile_section(bt: pd.DataFrame) -> None:
    score_cols = [col for col in bt.columns if col.startswith("s_")]
    if not score_cols:
        st.info("Signal columns not available for quintile test.")
        return
    factors = sorted(col.replace("s_", "") for col in score_cols)
    selected = st.selectbox("Quintile Test Factor", factors, index=0)
    signal = bt[f"s_{selected}"]
    ret_col = f"ret_{selected}"
    if ret_col not in bt.columns:
        st.warning(f"Forward return column {ret_col} missing.")
        return
    quintile = quintile_portfolios(signal, bt[ret_col].shift(-1))
    if quintile.empty:
        st.info("Insufficient signal diversity for quintile buckets.")
        return
    q_fig = px.bar(
        quintile,
        x=quintile.index,
        y=quintile.values,
        title=f"{selected} Q1–Q5 Forward Returns",
        color=quintile.values,
        color_continuous_scale="Turbo",
    )
    q_fig.update_layout(xaxis_title="Quintile", yaxis_title="Return")
    show_plotly(q_fig)


def _factor_stability(bt: pd.DataFrame) -> pd.DataFrame:
    weights = bt.filter(like="w_")
    if weights.empty:
        return pd.DataFrame()
    stability = (weights.var() / weights.abs().mean()).to_frame("FSI")
    stability.index = [col.replace("w_", "") for col in stability.index]
    stability = stability.reset_index().rename(columns={"index": "factor"})
    stability = stability.sort_values("FSI")
    return stability


def _rolling_beta(bt: pd.DataFrame) -> px.line | None:
    weights = bt.filter(like="w_")
    returns = bt.filter(like="ret_")
    if weights.empty or returns.empty:
        return None
    rolling = weights.rolling(12, min_periods=3).mean()
    rolling.index = pd.to_datetime(rolling.index)
    rolling.columns = [col.replace("w_", "") for col in rolling.columns]
    fig = px.line(rolling, title="Rolling β(t) (12M EMA)")
    fig.update_layout(xaxis_title="Date", yaxis_title="β")
    return fig


def _ic_significance_heatmap(ic_rolling: pd.DataFrame) -> px.imshow | None:
    if ic_rolling.empty:
        return None
    df = ic_rolling.copy()
    df.index = pd.to_datetime(df.index)
    df = df.loc["2000":]
    if df.empty:
        return None
    decade = (df.index.year // 10) * 10
    counts = {}
    for factor in df.columns:
        series = df[factor].abs() >= 0.1
        grouped = series.groupby(decade).sum()
        counts[factor] = grouped
    heat = pd.DataFrame(counts).fillna(0).astype(int)
    heat.index = heat.index.astype(str)
    fig = px.imshow(
        heat,
        text_auto=True,
        color_continuous_scale="Blues",
        title="IC Significance Count by Decade",
        labels={"x": "Factor", "y": "Decade", "color": "Count"},
    )
    return fig


def _attribution_section(bt: pd.DataFrame) -> pd.DataFrame:
    return_cols = [col for col in bt.columns if col.startswith("ret_")]
    weight_cols = [col for col in bt.columns if col.startswith("w_")]
    if not return_cols or not weight_cols:
        st.info("Attribution requires ret_* and w_* columns.")
        return pd.DataFrame()
    returns = bt[return_cols]
    weights = bt[weight_cols]
    aligned_returns = returns.copy()
    aligned_returns.columns = [col.replace("ret_", "") for col in aligned_returns.columns]
    aligned_weights = weights.copy()
    aligned_weights.columns = [col.replace("w_", "") for col in aligned_weights.columns]
    common = sorted(set(aligned_returns.columns) & set(aligned_weights.columns))
    if not common:
        st.info("No overlapping factors for attribution.")
        return pd.DataFrame()
    attribution = factor_attribution(aligned_returns[common], aligned_weights[common])
    st.dataframe(attribution, width="stretch")
    st.download_button(
        "📊 Download Attribution Table",
        data=attribution.to_csv(),
        file_name="attribution_summary.csv",
    )
    attr_plot = attribution.reset_index()
    first_col = attr_plot.columns[0]
    attr_plot = attr_plot.rename(columns={first_col: "year"})
    attr_fig = px.bar(
        attr_plot,
        x="year",
        y="sum",
        title="Yearly Contribution (Sharpe-scaled)",
        color="sum",
        color_continuous_scale="RdBu",
    )
    attr_fig.update_layout(xaxis_title="Year", yaxis_title="Contribution")
    show_plotly(attr_fig)
    return attribution


def _regime_correlation_heatmap(
    llm_log: pd.DataFrame,
) -> px.imshow | None:
    cached = load_regime_correlation()
    if cached is not None:
        corr = cached
    else:
        if llm_log.empty or not {"mri", "nlp_regime"} <= set(llm_log.columns):
            return None
        df = llm_log[["mri", "nlp_regime"]].dropna()
        if df.empty:
            return None
        corr = df.corr()
        cache_regime_correlation(corr)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        title="Macro vs NLP Regime Correlation",
    )
    return fig


def _case_studies(llm_log: pd.DataFrame) -> None:
    if llm_log.empty or not {"mri", "nlp_regime", "date"} <= set(llm_log.columns):
        st.info("Regime case studies unavailable.")
        return
    df = llm_log.dropna(subset=["mri", "nlp_regime"]).copy()
    if df.empty:
        st.info("Regime case studies unavailable.")
        return
    df["divergence"] = (df["mri"] - df["nlp_regime"]).abs()
    cases = df.nlargest(3, "divergence")
    if cases.empty:
        st.info("No significant macro–tone divergences detected.")
        return
    for _, row in cases.iterrows():
        st.info(
            f"**{pd.Timestamp(row['date']).strftime('%b %Y')}** — "
            f"Macro regime: {row['mri']:.2f}, NLP tone: {row['nlp_regime']:.2f} "
            f"(Δ={row['divergence']:.2f})."
        )


def _bootstrap_section(summary: dict[str, float]) -> None:
    if not summary:
        st.info("Bootstrap Sharpe summary missing.")
        return
    cols = st.columns(3)
    cols[0].metric("Sharpe*", f"{summary.get('point_estimate', float('nan')):0.2f}")
    cols[1].metric("Boot Mean", f"{summary.get('bootstrap_mean', float('nan')):0.2f}")
    cols[2].metric("t-stat", f"{summary.get('t_stat', float('nan')):0.2f}")
    st.caption(
        f"n={summary.get('n', 0)} | "
        f"p05={summary.get('p05', float('nan')):0.2f} "
        f"p95={summary.get('p95', float('nan')):0.2f} "
        f"| p-value={summary.get('p_value', float('nan')):0.3f}"
    )


def _regime_information(regime_df: pd.DataFrame) -> None:
    if regime_df.empty:
        st.info("Regime information ratio table missing.")
        return
    st.dataframe(regime_df, width="stretch")
    fig = px.bar(
        regime_df.reset_index(),
        x="regime",
        y="information_ratio",
        color="regime",
        title="Information Ratio by Regime",
    )
    fig.update_layout(xaxis_title="Regime", yaxis_title="Information Ratio")
    show_plotly(fig)


def render(
    ic_summary: pd.DataFrame,
    ic_rolling: pd.DataFrame,
    ablation: pd.DataFrame,
    sweep: pd.DataFrame,
    backtest: pd.DataFrame,
    macro_panel: pd.DataFrame,
    llm_log: pd.DataFrame,
    alpha_hint: float | None = None,
    alpha_decomp: pd.DataFrame | None = None,
    tracking_error: pd.DataFrame | None = None,
    drawdown_attr: pd.DataFrame | None = None,
    bootstrap_summary: dict[str, float] | None = None,
    regime_info: pd.DataFrame | None = None,
    excess_market: pd.DataFrame | None = None,
) -> None:
    alpha_df = alpha_decomp if alpha_decomp is not None else pd.DataFrame()
    tracking_df = tracking_error if tracking_error is not None else pd.DataFrame()
    drawdown_df = drawdown_attr if drawdown_attr is not None else pd.DataFrame()
    excess_df = excess_market if excess_market is not None else pd.DataFrame()
    regime_df = regime_info if regime_info is not None else pd.DataFrame()
    bootstrap_stats = bootstrap_summary or {}

    st.subheader("Performance Diagnostics")
    perf_cols = st.columns(2, gap="large")
    with perf_cols[0]:
        _alpha_beta_section(alpha_df)
    with perf_cols[1]:
        _tracking_error_section(tracking_df)

    st.subheader("Excess Return vs. Market")
    _excess_chart(excess_df)

    st.subheader("Drawdown Attribution")
    _drawdown_section(drawdown_df)

    st.subheader("Resampled Sharpe")
    _bootstrap_section(bootstrap_stats)

    st.subheader("Regime Information Ratio")
    _regime_information(regime_df)

    ic_df = pd.DataFrame()
    cols = st.columns(2, gap="large")
    if not ic_summary.empty:
        ic_df = _prepare_ic_frame(ic_summary)
        cols[0].subheader("Information Coefficients")
        cols[0].dataframe(ic_df, width="stretch")
        cols[0].download_button(
            "📄 Download IC Table",
            data=ic_df.to_csv(index=False),
            file_name="ic_summary.csv",
        )
        fig = px.bar(
            ic_df,
            x="factor",
            y="IC",
            color="significance_label",
            color_discrete_map=SIGNIFICANCE_COLORS,
            title="IC by Factor (p < 0.05 highlighted)",
        )
        show_plotly(fig)
    else:
        cols[0].info("IC summary not available.")
        ic_df = pd.DataFrame()

    if not ic_rolling.empty:
        ic_rolling.index = pd.to_datetime(ic_rolling.index)
        filtered = ic_rolling.loc["2010":]
        fig = px.line(filtered, title="Rolling IC (36M, 2010–2025)")
        fig.update_layout(xaxis_title="Date", yaxis_title="IC")
        show_plotly(fig)
        heatmap = _ic_significance_heatmap(ic_rolling)
        if heatmap is not None:
            show_plotly(heatmap)
    else:
        cols[1].info("Rolling IC file not found.")

    st.divider()
    subcols = st.columns(2, gap="large")
    if not ablation.empty:
        subcols[0].subheader("Factor Ablation")
        subcols[0].dataframe(ablation, width="stretch")
        ablation_fig = px.bar(
            ablation,
            x="factor",
            y="delta_sharpe",
            color="delta_sharpe",
            color_continuous_scale="RdBu",
            title="ΔSharpe by Factor",
        )
        show_plotly(ablation_fig)
    else:
        subcols[0].info("Ablation summary not available.")

    if not sweep.empty:
        subcols[1].subheader("Hyperparameter Sweep")
        subcols[1].dataframe(sweep, width="stretch")
        sweep_fig = px.scatter(
            sweep,
            x="temperature",
            y="sharpe",
            color="ridge_alpha",
            size="ann_ret",
            hover_data=["ann_vol", "max_dd"],
            color_continuous_scale="Turbo",
            title="Sharpe by Hyperparameters",
        )
        if alpha_hint is not None:
            sweep_fig.add_vline(
                x=alpha_hint,
                line_dash="dash",
                line_color="#facc15",
                annotation_text=f"α={alpha_hint:.2f}",
                annotation_position="top left",
            )
        show_plotly(sweep_fig)
    else:
        subcols[1].info("Sweep summary not available.")

    st.divider()
    st.subheader("Quintile Payoff")
    _quintile_section(backtest)

    st.subheader("Factor Attribution")
    attribution_df = _attribution_section(backtest)

    stability_df = _factor_stability(backtest)
    if not stability_df.empty:
        st.subheader("Factor Stability Index")
        st.dataframe(stability_df, width="stretch")
        if st.checkbox("Show rolling β(t)"):
            beta_fig = _rolling_beta(backtest)
            if beta_fig is not None:
                show_plotly(beta_fig)


    if not llm_log.empty:
        st.subheader("Regime Correlation")
        corr_fig = _regime_correlation_heatmap(llm_log)
        if corr_fig is not None:
            show_plotly(corr_fig)
        st.subheader("Regime Case Studies")
        _case_studies(llm_log)

    if not ic_df.empty or (attribution_df is not None and not attribution_df.empty):
        if st.checkbox("Generate Narrative Summary"):
            if openai is None or not getattr(openai, "api_key", None):
                st.warning(
                    "OpenAI integration unavailable. Using offline summary instead."
                )
            summary = generate_llm_summary(ic_df, attribution_df)
            st.markdown(summary)
            st.download_button(
                "📝 Download Narrative",
                data=summary.encode("utf-8"),
                file_name="diagnostics_narrative.txt",
            )
