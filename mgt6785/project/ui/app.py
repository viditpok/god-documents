import base64
import json
import sys
from functools import lru_cache
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from macrotones import __release_date__, __version__

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from macrotones.api import loader  # noqa: E402
from macrotones.backtest.config import BacktestConfig  # noqa: E402
from macrotones.backtest.report_summary import (  # noqa: E402
    report_summary as build_report_summary,
)
from macrotones.config.schema import ConfigModel  # noqa: E402
from macrotones.data.macro_loader import get_macro_features  # noqa: E402
from macrotones.data.nlp_sentiment import get_nlp_regime  # noqa: E402
from macrotones.datahub import (  # noqa: E402
    cache_regime_correlation,
    load_regime_correlation,
)
from macrotones.utils.regime import inverse_variance, smooth_lambda  # noqa: E402
from ui.components import kpi_card, section_divider, show_plotly  # noqa: E402
from ui.components_regime import regime_commentary  # noqa: E402
from ui.tabs import dataqa as dataqa_tab  # noqa: E402
from ui.tabs import diagnostics as diagnostics_tab  # noqa: E402
from ui.tabs import simulator as simulator_tab  # noqa: E402
from ui.theming import NAVY_DARK, inject_global_styles  # noqa: E402

inject_global_styles()

MACRO_FEATURE_CANDIDATES = ["CPI", "UNRATE", "DGS10", "DGS1", "TERM_SPREAD"]
MONTH_ORDER = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
REPORT_PATH = Path("data/processed/report.html")


@lru_cache(maxsize=1)
def _cached_report_bytes() -> bytes | None:
    if REPORT_PATH.exists():
        return REPORT_PATH.read_bytes()
    return None


def render_report_download(
    label: str = "Download HTML Report",
    key: str | None = None,
) -> None:
    report_bytes = _cached_report_bytes()
    if not report_bytes:
        st.caption("HTML report not found. Run `make report` to generate artifacts.")
        return
    st.download_button(
        label,
        data=report_bytes,
        file_name=f"macrotones_report_{pd.Timestamp.today():%Y%m%d}.html",
        mime="text/html",
        key=key,
    )


def render_png_download(fig: go.Figure, filename: str, label: str = "📸 Export PNG") -> None:
    try:
        png_bytes = fig.to_image(format="png")
    except Exception as exc:  # pragma: no cover - depends on kaleido
        st.warning("PNG export requires the `kaleido` package.")
        return
    st.download_button(
        label,
        data=png_bytes,
        file_name=filename,
        mime="image/png",
    )


def _config_picker() -> Path:
    configs = loader.list_configurations()
    default_path = loader.DEFAULT_CONFIG
    options = list(configs.keys()) if configs else []
    if default_path.exists() and default_path.stem not in configs:
        configs[default_path.stem] = default_path
        options.append(default_path.stem)
    if not options:
        st.error("No configuration YAML files were found under ./config.")
        st.stop()
    is_default = default_path.stem in options
    default_index = options.index(default_path.stem) if is_default else 0
    selected = st.sidebar.selectbox("Configuration", options, index=default_index)
    return configs[selected]


def _sidebar_controls(cfg: ConfigModel | None = None) -> dict[str, object]:
    with st.sidebar:
        # Visual settings removed for formal presentation
        sharpe_target = 1.0
        
        lambda_cfg = getattr(cfg.portfolio, "smooth_lambda", None) if cfg else None
        lambda_prior = float(lambda_cfg) if lambda_cfg is not None else 0.5
        
        alpha_cfg = getattr(cfg.model, "ridge_alpha", None) if cfg else None
        alpha_override = float(alpha_cfg) if alpha_cfg is not None else 1.0
        
        cost_bps = float(cfg.portfolio.cost_bps) if cfg else 10.0
        benchmark = cfg.project.benchmark if cfg else "SPY"

        with st.expander("Advanced Settings"):
            if st.button("Clear Cache & Reload"):
                import shutil
                cache_dir = Path(".cache")
                shutil.rmtree(cache_dir, ignore_errors=True)
                cache_dir.mkdir(parents=True, exist_ok=True)
                st.cache_data.clear()
                st.toast("Data cache cleared. Please refresh.")

    return {
        "fast_mode": False,
        "sharpe_target": sharpe_target,
        "lambda_prior": lambda_prior,
        "alpha_override": alpha_override,
        "cost_bps": cost_bps,
        "benchmark": benchmark,
    }

with st.sidebar:
    st.caption(f"MacroTone v{__version__} • {__release_date__}")


def strategy_docs_page() -> None:
    st.markdown("# 📘 Project Documentation: MacroTone Strategy")
    st.markdown("### 1. Executive Summary")
    st.write(
        """
        **MacroTone** is a next-generation asset allocation framework developed for the Advanced AI in Finance Capstone. 
        It integrates **Macroeconomic Regime Detection** with **Natural Language Processing (NLP)** to predict 
        factor performance and dynamically rotate a portfolio of ETFs.
        """
    )
    
    with st.expander("Explore Technical Architecture", expanded=True):
        st.markdown(
            """
            | Component | Specification | Description |
            | :--- | :--- | :--- |
            | **Macro Engine** | **PCA (Principal Component Analysis)** | Extracts a single "Regime Index" from GDP, CPI, and Yield Curve data. |
            | **NLP Engine** | **FinBERT (HuggingFace)** | Analyzes **FOMC Minutes & Press Conferences** to quantify Hawkish/Dovish sentiment. |
            | **Predictive Model** | **Ensemble (Ridge + XGBoost)** | Blends linear trends with non-linear regime interactions to forecast returns. |
            | **Validation** | **Walk-Forward (Expanding Window)** | Retrains monthly to prevent look-ahead bias. Min training window: **120 months**. |
            """
        )

    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 2. Strategy Specifications")
        st.info("These parameters define the current live configuration (Top-K).")
        st.markdown(
            """
            *   **Allocation Logic**: `Top-K (K=2)`
                *   The model selects the **2 highest-conviction factors** each month and equal-weights them.
                *   *Why?* Concentration maximizes the impact of accurate predictions compared to a diluted Mean-Variance portfolio.
            *   **Volatility Target**: `15% Annualized`
                *   Matches the risk profile of the S&P 500 (SPY).
                *   *Mechanism:* If portfolio volatility is predicted to exceed 15%, the strategy moves a portion of capital to **Cash** (De-leveraging).
            *   **Rebalancing**: `Monthly`
                *   Trades are executed at month-end closing prices.
            *   **Transaction Costs**: `10 bps` (Modeling Estimates)
                *   Accounts for slippage and commission drag.
            """
        )

        st.markdown("### 3. Investment Universe (ETFs)")
        st.markdown("The strategy trades liquid US Equity Factors. We map theoretical factors to specific BlackRock/Vanguard ETFs:")
        
        etf_data = {
            "Factor": ["Value", "Size", "Momentum", "Quality", "Low Volatility"],
            "Ticker": ["VTV", "IJR", "MTUM", "QUAL", "USMV"],
            "Description": ["Vanguard Value", "iShares Core S&P Small-Cap", "iShares MSCI USA Momentum", "iShares MSCI USA Quality", "iShares Min Volatility"]
        }
        st.table(pd.DataFrame(etf_data))
        
        st.markdown("### 4. Blending Strategy (Ensemble)")
        st.markdown(
            """
            The final signal is constructed using a **Weight-Space Ensemble**:
            1.  **Ridge Regression** predicts returns based on linear factor momentum.
            2.  **XGBoost** predicts returns based on non-linear macro interactions (e.g., *Value works best when Rates are rising*).
            3.  The final score is a 50/50 blend of these two models.
            """
        )

    st.divider()


def load_llm_policy_log(config_path: Path) -> pd.DataFrame:
    path = loader.processed_path(config_path) / "llm_policy_log.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def load_macro_panel(config_path: Path) -> pd.DataFrame:
    path = loader.processed_path(config_path) / "panel.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_nlp_scores() -> pd.DataFrame:
    path = Path("data/interim/nlp_regime_scores.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "doc_month" in df.columns:
        df = df.rename_axis("doc_month").reset_index().set_index("doc_month")
    df.index = pd.to_datetime(df.index)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_etf_backtest(config_path: Path) -> pd.DataFrame:
    path = loader.processed_path(config_path) / "backtest_etfs.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()





def _load_all(config_path: Path) -> dict[str, pd.DataFrame | Path | dict | ConfigModel]:
    cfg = loader._load_config(config_path)
    backtest = loader.load_backtest(config_path)
    ic_summary = loader.load_ic_summary(config_path)
    ic_rolling = loader.load_ic_rolling(config_path)
    ablation = loader.load_ablation(config_path)
    sweep = loader.load_sweep(config_path)
    turnover = loader.load_turnover(config_path)
    alpha_decomp = loader.load_alpha_decomposition(config_path)
    tracking_error = loader.load_tracking_error(config_path)
    excess_market = loader.load_excess_market(config_path)
    drawdown_attr = loader.load_drawdown_attribution(config_path)
    bootstrap_summary = loader.load_bootstrap_sharpe(config_path)
    regime_info = loader.load_regime_information(config_path)
    regime_coverage = loader.load_regime_coverage(config_path)
    images = loader.get_report_images(config_path)
    llm_log = load_llm_policy_log(config_path)
    macro_panel = load_macro_panel(config_path)
    nlp_scores = load_nlp_scores()
    backtest_etf = load_etf_backtest(config_path)
    return {
        "cfg": cfg,
        "backtest": backtest,
        "backtest_etf": backtest_etf,
        "ic_summary": ic_summary,
        "ic_rolling": ic_rolling,
        "ablation": ablation,
        "sweep": sweep,
        "turnover": turnover,
        "alpha_decomp": alpha_decomp,
        "tracking_error": tracking_error,
        "excess_market": excess_market,
        "drawdown_attr": drawdown_attr,
        "bootstrap_sharpe": bootstrap_summary,
        "regime_info": regime_info,
        "regime_coverage": regime_coverage,
        "images": images,
        "llm_log": llm_log,
        "macro_panel": macro_panel,
        "nlp_scores": nlp_scores,
    }


def _latest_macro_snapshot(bt: pd.DataFrame) -> dict[str, float] | None:
    if bt.empty:
        return None
    date = pd.Timestamp(bt.index[-1])
    try:
        return get_macro_features(date)
    except Exception:
        return None


def _latest_nlp_snapshot(bt: pd.DataFrame) -> dict[str, float] | None:
    if bt.empty:
        return None
    date = pd.Timestamp(bt.index[-1])
    try:
        return get_nlp_regime(date)
    except Exception:
        return None


def _latest_llm_weights(
    llm_log: pd.DataFrame,
    prefix: str = "w_llm_",
) -> dict[str, float] | None:
    if llm_log.empty:
        return None
    latest = llm_log.iloc[-1]
    cols = [col for col in latest.index if col.startswith(prefix)]
    if not cols:
        return None
    return {col.replace(prefix, ""): float(latest[col]) for col in cols}


@st.cache_data(show_spinner=False)
def get_data(config_path: str) -> dict:
    # Adding comment to force cache invalidation
    return _load_all(Path(config_path))


def compute_metrics(bt: pd.DataFrame, cfg: ConfigModel) -> dict[str, float]:
    if bt.empty:
        return {}
    net, rf = loader.get_return_series(bt, cfg)
    cum = (1 + net).cumprod()
    drawdown = cum / cum.cummax() - 1
    ann_ret = (1 + net).prod() ** (12 / len(net)) - 1 if len(net) else float("nan")
    ann_vol = net.std(ddof=0) * np.sqrt(12)
    excess = net - rf
    sharpe = float("nan")
    if excess.std(ddof=0) not in (0, float("nan")) and excess.std(ddof=0) != 0:
        sharpe = (excess.mean() / excess.std(ddof=0)) * np.sqrt(12)
    return {
        "Annual Return": ann_ret,
        "Annual Volatility": ann_vol,
        "Sharpe (Excess)": sharpe,
        "Max Drawdown": drawdown.min(),
        "Cumulative Return": cum.iloc[-1] - 1,
    }


def _fmt_value(value: float | None, pct: bool = False, digits: int = 2) -> str:
    invalid = isinstance(value, float) and (np.isnan(value) or np.isinf(value))
    if value is None or invalid:
        return "n/a"
    if pct:
        return f"{value:.1%}"
    return f"{value:.{digits}f}"


def rolling_sharpe_excess(
    ret: pd.Series, window: int = 12, min_window: int = 3
) -> pd.Series:
    r = ret.dropna()
    if r.empty:
        return r
    roll_mean = r.rolling(window, min_periods=min_window).mean()
    roll_std = r.rolling(window, min_periods=min_window).std(ddof=0)
    sharpe = (roll_mean / roll_std) * np.sqrt(12)
    return sharpe.replace([np.inf, -np.inf], np.nan)


def overview_page(
    bt: pd.DataFrame,
    cfg: ConfigModel,
    config_path: Path,
    settings: dict[str, object],
    bt_etf: pd.DataFrame | None = None,
) -> None:
    if bt.empty:
        st.warning("Backtest results are not available for this configuration.")
        return
    bt_cfg = BacktestConfig.from_model(cfg)
    metrics = compute_metrics(bt, cfg)
    net, rf = loader.get_return_series(bt, cfg)
    cum = (1 + net).cumprod()
    drawdown = cum / cum.cummax() - 1
    roll_sharpe = rolling_sharpe_excess(net - rf)
    market_curve = loader.load_market_series(cum.index, config_path)
    market_equity = (1 + market_curve.fillna(0.0)).cumprod()
    alpha = float(cum.iloc[-1] - market_equity.reindex(cum.index).iloc[-1])
    turnover = bt.get("turnover")
    avg_turnover = float(turnover.mean()) if turnover is not None else float("nan")
    sharpe_text = _fmt_value(metrics.get("Sharpe (Excess)"))
    ann_ret_text = _fmt_value(metrics.get("Annual Return"), pct=True)
    ann_vol_text = _fmt_value(metrics.get("Annual Volatility"), pct=True)
    max_dd_text = _fmt_value(metrics.get("Max Drawdown"), pct=True)

    summary_text = build_report_summary(cfg, bt_cfg, bt.index)
    st.caption(summary_text.header)
    st.markdown(
        f"**Performance Summary** ({summary_text.subtitle})\n\n"
        f"Sharpe {sharpe_text} with annual return {ann_ret_text}, "
        f"vol {ann_vol_text}, and drawdown {max_dd_text}. "
        f"Strategy outperformed market by {_fmt_value(alpha, pct=True)} "
        f"with turnover ≈ {_fmt_value(avg_turnover)}."
    )

    kpi_defs = [
        ("Annual Return", metrics.get("Annual Return"), "%"),
        ("Annual Volatility", metrics.get("Annual Volatility"), "%"),
        ("Sharpe", metrics.get("Sharpe (Excess)"), ""),
        ("Max Drawdown", metrics.get("Max Drawdown"), "%"),
        ("Cumulative Return", metrics.get("Cumulative Return"), "%"),
    ]
    cols = st.columns(len(kpi_defs), gap="large")
    for col, (label, value, suffix) in zip(cols, kpi_defs, strict=False):
        with col:
            kpi_card(label, value, suffix=suffix)

    if bt_etf is not None and not bt_etf.empty:
        st.markdown("### ETF Strategy Performance")
        metrics_etf = compute_metrics(bt_etf, cfg)
        net_etf, rf_etf = loader.get_return_series(bt_etf, cfg)
        # Recalculate alpha for ETF vs Market
        cum_etf = (1 + net_etf.reindex(cum.index).fillna(0.0)).cumprod()
        alpha_etf = float(cum_etf.iloc[-1] - market_equity.reindex(cum.index).iloc[-1])
        etf_turnover = bt_etf.get("turnover")
        avg_etf_turnover = float(etf_turnover.mean()) if etf_turnover is not None else float("nan")

        st.markdown(
            f"Sharpe {_fmt_value(metrics_etf.get('Sharpe (Excess)'))} with annual return "
            f"{_fmt_value(metrics_etf.get('Annual Return'), pct=True)}. "
            f"Alpha vs Market: {_fmt_value(alpha_etf, pct=True)}."
        )

        kpi_defs_etf = [
            ("Annual Return", metrics_etf.get("Annual Return"), "%"),
            ("Annual Volatility", metrics_etf.get("Annual Volatility"), "%"),
            ("Sharpe", metrics_etf.get("Sharpe (Excess)"), ""),
            ("Max Drawdown", metrics_etf.get("Max Drawdown"), "%"),
            ("Cumulative Return", metrics_etf.get("Cumulative Return"), "%"),
        ]
        cols_etf = st.columns(len(kpi_defs_etf), gap="large")
        for col, (label, value, suffix) in zip(cols_etf, kpi_defs_etf, strict=False):
            with col:
                kpi_card(label, value, suffix=suffix)

    eq_fig = go.Figure()
    eq_fig.add_trace(
        go.Scatter(
            x=cum.index,
            y=cum.values,
            name="Strategy",
            line={"color": NAVY_DARK["accent"], "width": 3},
        )
    )
    eq_fig.add_trace(
        go.Scatter(
            x=market_equity.index,
            y=market_equity.values,
            name="Market",
            line={"color": "#fbbf24", "width": 2, "dash": "dash"},
        )
    )
    
    # Optional ETF Comparison
    if bt_etf is not None and not bt_etf.empty:
        net_etf, rf_etf = loader.get_return_series(bt_etf, cfg)
        cum_etf = (1 + net_etf.reindex(cum.index).fillna(0.0)).cumprod()
        eq_fig.add_trace(
            go.Scatter(
                x=cum_etf.index,
                y=cum_etf.values,
                name="ETF Strategy (Tradeable)",
                line={"color": "#10b981", "width": 2, "dash": "dot"},
            )
        )

    eq_fig.update_layout(title="Equity Curve")
    show_plotly(eq_fig)
    render_png_download(eq_fig, "macrotones_equity.png")

    dd_fig = go.Figure()
    dd_fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            line={"color": "#f97316"},
            name="Drawdown",
        )
    )
    dd_fig.update_layout(title="Drawdown")
    show_plotly(dd_fig)

    # --- NEW: Advanced Metrics Row ---
    st.subheader("Risk & Return Analysis")
    
    # 1. Rolling Volatility
    vol_fig = go.Figure()
    rolling_vol = (net - rf).rolling(12).std() * np.sqrt(12)
    vol_fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            line={"color": "#f472b6"},
            name="Rolling Vol (12M)",
        )
    )
    # Add Target Line
    vol_fig.add_hline(
        y=0.15, 
        line_dash="dash", 
        line_color="white", 
        annotation_text="Target (15%)",
        annotation_position="top left"
    )
    vol_fig.update_layout(title="Rolling Volatility (Annualized)")

    # 2. Returns Distribution
    dist_fig = go.Figure()
    dist_fig.add_trace(go.Histogram(x=net, name="Strategy", opacity=0.75, marker_color=NAVY_DARK["accent"]))
    if not market_curve.empty:
        market_rets = market_curve.diff() # curve is cumprod ?? No, load_market_series returns returns or price?
        # loader.load_market_series returns a Series of returns usually. 
        # Checking usage: market_equity = (1+market_curve).cumprod() -> So market_curve IS returns.
        dist_fig.add_trace(go.Histogram(x=market_curve, name="Benchmark", opacity=0.75, marker_color="#fbbf24"))
    dist_fig.update_layout(title="Monthly Returns Distribution", barmode="overlay")

    row2_cols = st.columns(2, gap="large")
    with row2_cols[0]:
        show_plotly(vol_fig)
    with row2_cols[1]:
        show_plotly(dist_fig)

    # --- NEW: Monthly Heatmap ---
    st.subheader("Monthly Returns Heatmap")
    
    # Prepare data for heatmap
    ret_df = net.to_frame("return")
    ret_df["Year"] = ret_df.index.year
    ret_df["Month"] = ret_df.index.strftime("%b")
    
    # Pivot for heatmap: Index=Year, Columns=Month
    # We need to ensure month order
    heatmap_data = ret_df.pivot_table(index="Year", columns="Month", values="return", aggfunc="mean")
    # Reindex columns to ensure Jan-Dec order
    heatmap_data = heatmap_data.reindex(columns=MONTH_ORDER)

    # Visualizing
    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Month", y="Year", color="Return"),
        x=MONTH_ORDER,
        y=heatmap_data.index,
        color_continuous_scale="RdBu",
        range_color=[-0.1, 0.1], # Cap color scale for readability
        text_auto=".1%",
        aspect="auto"
    )
    fig_heat.update_layout(title="Monthly Returns Performance")
    show_plotly(fig_heat)

    # --- MERGED: Portfolio Weights (Area Chart) ---
    st.subheader("Portfolio Allocation")
    weights = bt.filter(like="w_")

    if not weights.empty:
        weights_reset = weights.reset_index()
        date_col = weights_reset.columns[0]
        weights_long = weights_reset.rename(columns={date_col: "Date"}).melt(
            "Date", var_name="Factor", value_name="Weight"
        )
        weights_long["Date"] = pd.to_datetime(weights_long["Date"])
        
        # Clean up factor names
        weights_long["Factor"] = weights_long["Factor"].str.replace("w_", "")
        
        fig = px.area(
            weights_long,
            x="Date",
            y="Weight",
            color="Factor",
            color_discrete_sequence=px.colors.qualitative.Antique,
            title="Portfolio Weights (Stacked Area)",
        )
        fig.update_layout(
            yaxis_title="Allocation %",
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            }
        )
        show_plotly(fig)
    else:
        st.info("Weight columns not available (expected prefix w_).")

    render_report_download(key="report_overview")





def _macro_heatmap_animation(macro_panel: pd.DataFrame) -> go.Figure | None:
    if macro_panel.empty:
        return None
    macro_cols = [col for col in MACRO_FEATURE_CANDIDATES if col in macro_panel.columns]
    if not macro_cols:
        return None
    recent = macro_panel[macro_cols].dropna().tail(120)
    if recent.empty:
        return None
    df = recent.reset_index().rename(columns={"index": "date"})
    df["year"] = df["date"].dt.year.astype(str)
    df["month"] = pd.Categorical(
        df["date"].dt.strftime("%b"),
        categories=MONTH_ORDER,
        ordered=True,
    )
    long_df = df.melt(
        id_vars=["year", "month"],
        value_vars=macro_cols,
        var_name="indicator",
        value_name="value",
    )
    fig = px.density_heatmap(
        long_df,
        x="month",
        y="indicator",
        z="value",
        animation_frame="year",
        color_continuous_scale="RdBu_r",
        title="Macro Heatmap (animated by year)",
    )
    fig.update_yaxes(categoryorder="array", categoryarray=macro_cols)
    return fig


def _lambda_histogram(
    llm_log: pd.DataFrame,
    target: float | None = None,
) -> go.Figure | None:
    if llm_log.empty or "lambda" not in llm_log.columns:
        return None
    fig = px.histogram(
        llm_log,
        x="lambda",
        nbins=20,
        color_discrete_sequence=["#f472b6"],
        title="λ Blend Distribution",
    )
    if target is not None:
        fig.add_vline(
            x=target,
            line_color="#facc15",
            line_dash="dash",
            annotation_text=f"λ={target:.2f}",
            annotation_position="top left",
        )
    return fig


def _macro_nlp_correlation(
    macro_panel: pd.DataFrame,
    llm_log: pd.DataFrame,
) -> go.Figure | None:
    if macro_panel.empty or llm_log.empty or "nlp_regime" not in llm_log.columns:
        return None
    macro_cols = [col for col in MACRO_FEATURE_CANDIDATES if col in macro_panel.columns]
    if not macro_cols:
        return None
    merged = macro_panel[macro_cols].join(
        llm_log.set_index("date")[["nlp_regime"]],
        how="inner",
    )
    if merged.shape[0] < 12:
        return None
    corr = merged.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Macro ↔ NLP Regime Correlation",
    )
    return fig


def macro_nlp_page(
    _config_path: Path,
    macro_panel: pd.DataFrame,
    llm_log: pd.DataFrame,
    nlp_scores: pd.DataFrame,
    fast_mode: bool = False,
    lambda_target: float | None = None,
) -> None:
    st.markdown(
        "**Macro & NLP Narrative** - Track regime shifts by blending macro PCA, "
        "FinBERT tone, and LLM policy signals."
    )
    if fast_mode and not nlp_scores.empty:
        nlp_scores = nlp_scores.tail(24)
        st.warning("Fast mode trims NLP history to the most recent 24 months.")

    macro_col, nlp_col = st.columns(2, gap="large")

    with macro_col:
        section_divider("Macro Regimes")
        
        if not llm_log.empty and "mri" in llm_log.columns:
            mri_fig = go.Figure()
            mri_fig.add_trace(
                go.Scatter(
                    x=llm_log["date"],
                    y=llm_log["mri"],
                    line={"color": "#c084fc"},
                    name="MRI",
                )
            )
            mri_fig.update_layout(title="Macro Regime Index (MRI)")
            show_plotly(mri_fig)
        else:
            st.info("MRI timeline unavailable - missing LLM policy log.")

    with nlp_col:
        section_divider("NLP Signals")
        
        if not llm_log.empty and "nlp_regime" in llm_log.columns:
            bar_df = llm_log.tail(60).copy()
            colors = np.where(bar_df["nlp_regime"] >= 0, "#22d3ee", "#f43f5e")
            regime_fig = go.Figure()
            regime_fig.add_trace(
                go.Bar(
                    x=bar_df["date"],
                    y=bar_df["nlp_regime"],
                    marker_color=colors,
                    name="NLP Regime",
                )
            )
            regime_fig.update_layout(title="LLM NLP Regime (Recent 5y)")
            show_plotly(regime_fig)
        else:
            st.info("Regime bar chart unavailable - LLM log missing nlp_regime.")


    latest = llm_log.iloc[-1] if not llm_log.empty else None
    if latest is not None:
        st.markdown(f"### Regime Insight ({pd.Timestamp(latest['date']):%b %Y})")
        st.info(regime_commentary(latest))
    elif not nlp_scores.empty:
        last_date = nlp_scores.index[-1]
        fallback_row = pd.Series(
            {
                "mri": np.nan,
                "nlp_regime": float(
                    nlp_scores.get("pos_minus_neg_filled", nlp_scores.iloc[:, 0]).iloc[
                        -1
                    ]
                ),
            }
        )
        st.markdown(f"### Regime Insight ({pd.Timestamp(last_date):%b %Y})")
        st.info(regime_commentary(fallback_row))

    st.divider()
    col_policy, col_download = st.columns([3, 2])
    if latest is not None:
        weights_cols = [col for col in latest.index if col.startswith("w_llm_")]
        if weights_cols:
            policy_fig = go.Figure()
            policy_fig.add_trace(
                go.Bar(
                    x=[col.replace("w_llm_", "") for col in weights_cols],
                    y=[latest[col] for col in weights_cols],
                    marker_color="#fb7185",
                    name="LLM weights",
                )
            )
            policy_fig.update_layout(title="Latest LLM Policy Weights")
            with col_policy:
                show_plotly(policy_fig)
        rationale = latest.get("rationale", "")
        with st.expander("Latest LLM Rationale", expanded=False):
            if isinstance(rationale, str) and rationale.strip():
                st.write(rationale)
            else:
                st.write("No rationale captured for the latest policy.")
    else:
        st.info(
            "LLM policy log is empty. Run the engine to populate "
            "data/processed/llm_policy_log.csv."
        )

    if not llm_log.empty:
        csv_buf = StringIO()
        llm_log.to_csv(csv_buf, index=False)
        col_download.download_button(
            "Download LLM Policy Log",
            csv_buf.getvalue(),
            file_name="llm_policy_log.csv",
            mime="text/csv",
        )
    else:
        col_download.info("LLM policy log not found.")

    render_report_download(key="report_macro")


def tearsheet_page(
    bt: pd.DataFrame,
    cfg: ConfigModel,
    images: dict[str, Path | None],
    sharpe_target: float | None = None,
) -> None:
    if bt.empty:
        st.info("Backtest results unavailable for tearsheet visuals.")
        return
    net, rf = loader.get_return_series(bt, cfg)
    cum = (1 + net).cumprod()
    drawdown = cum / cum.cummax() - 1
    roll_sharpe = rolling_sharpe_excess(net - rf)

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        dd_fig = go.Figure()
        dd_fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill="tozeroy",
                line={"color": "#fb7185"},
                name="Drawdown",
            )
        )
        dd_fig.update_layout(title="Drawdown (Enlarged)")
        show_plotly(dd_fig)
    with col_right:
        rs_fig = go.Figure()
        rs_fig.add_trace(
            go.Scatter(
                x=roll_sharpe.index,
                y=roll_sharpe.values,
                line={"color": "#2dd4bf"},
                name="Rolling Sharpe",
            )
        )
        if sharpe_target and sharpe_target > 0:
            rs_fig.add_hline(
                y=sharpe_target,
                line_color="#facc15",
                line_dash="dash",
                annotation_text=f"Target {sharpe_target:.1f}",
                annotation_position="top left",
            )
        rs_fig.update_layout(title="Rolling Sharpe (Enlarged)")
        show_plotly(rs_fig)

    st.markdown(
        "**Tearsheet Context** - Larger drawdown and rolling performance visuals "
        "for presentation-ready screenshots."
    )

    if images.get("tearsheet"):
        st.image(str(images["tearsheet"]), caption="MacroTone Tearsheet")
    elif images.get("rolling"):
        st.image(str(images["rolling"]), caption="Rolling Sharpe Snapshot")

    render_report_download(key="report_tearsheet")


def main() -> None:
    config_path = _config_picker()
    cfg_for_sidebar = loader._load_config(config_path)
    settings = _sidebar_controls(cfg_for_sidebar)
    with st.spinner("Loading analytics..."):
        data = get_data(str(config_path.resolve()))

    bt = data["backtest"]
    cfg = data["cfg"]
    baseline_macro = _latest_macro_snapshot(bt)
    baseline_nlp = _latest_nlp_snapshot(bt)
    baseline_weights = _latest_llm_weights(data["llm_log"])
    latest_returns = None
    ret_cols = bt.filter(like="ret_")
    if not ret_cols.empty:
        latest_returns = ret_cols.iloc[-1].rename(
            lambda col: col.replace("ret_", "")
        )

    tabs = st.tabs(
        [
            "📘 Project Documentation",
            "Strategy Analysis",
            "Macro & NLP Intelligence",
            "Model Diagnostics",
            "Scenario Simulator",
        ]
    )

    with tabs[0]:
        strategy_docs_page()

    with tabs[1]:
        overview_page(bt, cfg, config_path, settings, data.get("backtest_etf"))


    with tabs[2]:
        macro_nlp_page(
            config_path,
            data["macro_panel"],
            data["llm_log"],
            data["nlp_scores"],
            fast_mode=settings["fast_mode"],
            lambda_target=data["cfg"].portfolio.smooth_lambda,
        )

    with tabs[3]:
        diagnostics_tab.render(
            data["ic_summary"],
            data["ic_rolling"],
            data["ablation"],
            data["sweep"],
            bt,
            data["macro_panel"],
            data["llm_log"],
            alpha_hint=settings["alpha_override"],
            alpha_decomp=data["alpha_decomp"],
            tracking_error=data["tracking_error"],
            drawdown_attr=data["drawdown_attr"],
            bootstrap_summary=data["bootstrap_sharpe"],
            regime_info=data["regime_info"],
            excess_market=data["excess_market"],
        )
        render_report_download(key="report_diagnostics")

    with tabs[4]:
        simulator_tab.render(
            baseline_macro,
            baseline_nlp,
            baseline_weights,
            latest_returns=latest_returns,
            cost_bps=float(cfg.portfolio.cost_bps),
        )
        render_report_download(key="report_simulator")


if __name__ == "__main__":
    main()
