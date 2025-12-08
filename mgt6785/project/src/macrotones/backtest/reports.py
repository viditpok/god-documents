from __future__ import annotations

import base64
import textwrap
from datetime import datetime
from numbers import Real
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from macrotones import __release_date__, __version__
from macrotones.backtest.metrics import (
    annualize_monthly,
    max_drawdown,
    sharpe_excess,
)
from macrotones.config.schema import ConfigModel, load_config
from macrotones.utils.seed import DEFAULT_SEED, set_global_seed

CONFIG_PATH = Path("config/project.yaml")
CFG: ConfigModel = load_config(CONFIG_PATH)
PRO = Path(CFG.data.out_processed)


def _pick_series(
    bt: pd.DataFrame,
    candidates: list[str],
    default: float = 0.0,
) -> pd.Series:
    for col in candidates:
        if col in bt.columns:
            return bt[col]
    if bt.filter(like="ret").shape[1] > 0:
        return bt.filter(like="ret").iloc[:, 0]
    return pd.Series(default, index=bt.index)


def get_market_series(bt_index: pd.Index) -> pd.Series:
    bt_path = PRO / "backtest.parquet"
    if bt_path.exists():
        bt = pd.read_parquet(bt_path)
        for col in ["benchmark_ret", "Mkt_RF", "market", "mkt_rf", "market_excess"]:
            if col in bt.columns:
                return bt[col].reindex(bt_index)
    ff = pd.read_parquet(Path(CFG.data.out_raw) / "ff" / "ff_monthly.parquet")
    ff.index = pd.to_datetime(ff.index)
    return ff["Mkt_RF"].reindex(bt_index)


def compute_metrics(bt: pd.DataFrame) -> pd.Series:
    net = _pick_series(bt, ["net_ret", "port_ret", "strategy"], default=0.0)
    rf = _pick_series(bt, ["rf", CFG.project.rf_col], default=0.0)
    ann_ret, ann_vol = annualize_monthly(net)
    sharpe = sharpe_excess(net - rf)
    metrics = {
        "Annual Return": ann_ret,
        "Annual Volatility": ann_vol,
        "Sharpe (excess)": sharpe,
        "Max Drawdown": max_drawdown(net),
        "Average Turnover": bt["turnover"].mean() if "turnover" in bt else float("nan"),
    }
    return pd.Series(metrics)


def _rolling_sharpe_excess(
    ret: pd.Series, window: int = 12, min_window: int = 3
) -> pd.Series:
    # monthly -> annualized using sqrt(12)
    r = ret.dropna()
    if r.empty:
        return r
    roll_mean = r.rolling(window, min_periods=min_window).mean()
    roll_std = r.rolling(window, min_periods=min_window).std(ddof=0)
    sharpe = (roll_mean / roll_std) * np.sqrt(12)
    return sharpe.replace([np.inf, -np.inf], np.nan)


def _data_qa_summary(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    stats = panel.isna().mean().to_frame("Missing%")
    stats["Missing%"] = stats["Missing%"].mul(100)
    stats["Last Refresh"] = pd.to_datetime(panel.index.max())
    today = pd.Timestamp.today().normalize()
    stats["Vintage Lag (days)"] = (today - stats["Last Refresh"]).dt.days
    rolling_std = panel.std(ddof=0).replace(0.0, pd.NA)
    outliers = (panel - panel.mean()).abs() > (3 * rolling_std.fillna(0))
    stats["Outlier%"] = outliers.sum() / len(panel) * 100
    stats["QA Signal"] = stats["Missing%"].apply(
        lambda pct: "🟢" if pct == 0 else ("🟡" if pct <= 10 else "🔴")
    )
    return stats.round({"Missing%": 2, "Outlier%": 2})


def draw_tearsheet(
    bt: pd.DataFrame,
    tearsheet_path: Path,
    market_curve: pd.Series,
) -> None:
    bt = bt.copy()
    net = _pick_series(bt, ["net_ret", "port_ret", "strategy"], default=0.0).fillna(0.0)
    rf = _pick_series(bt, ["rf", CFG.project.rf_col], default=0.0).fillna(0.0)
    equity = (1 + net).cumprod()
    mkt_equity = (1 + market_curve.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1
    roll_sharpe = _rolling_sharpe_excess(net - rf)
    weights = bt.filter(like="w_")
    turnover = bt.get("turnover", pd.Series(index=bt.index, dtype=float)).fillna(0.0)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    axes[0].plot(equity.index, equity.values, label="Strategy")
    axes[0].plot(mkt_equity.index, mkt_equity.values, label="Market")
    axes[0].set_title("Equity Curve")
    axes[0].legend()

    axes[1].plot(drawdown.index, drawdown.values, color="tab:red")
    axes[1].set_title("Drawdown")
    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="tab:red", alpha=0.3)

    axes[2].plot(roll_sharpe.index, roll_sharpe.values, color="tab:green")
    axes[2].set_title("Rolling Sharpe (min window 3, target 12)")

    if not weights.empty:
        heatmap = weights.fillna(0.0)
        heatmap.index = pd.to_datetime(heatmap.index)
        dates = mdates.date2num(heatmap.index.to_pydatetime())
        if len(dates) > 1:
            spacing = np.diff(dates).mean()
        else:
            spacing = 1.0
        extent = [
            dates[0] - spacing / 2,
            dates[-1] + spacing / 2,
            -0.5,
            len(heatmap.columns) - 0.5,
        ]
        im = axes[3].imshow(
            heatmap.values.T,
            aspect="auto",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            origin="lower",
            extent=extent,
        )
        axes[3].set_title("Weights Heatmap")
        axes[3].set_yticks(range(len(heatmap.columns)))
        axes[3].set_yticklabels([col.replace("w_", "") for col in heatmap.columns])
        axes[3].xaxis_date()
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    else:
        axes[3].set_title("Weights Heatmap")
        axes[3].text(0.5, 0.5, "No weight data", ha="center", va="center")
        axes[3].axis("off")

    axes[4].plot(turnover.index, turnover.values, color="tab:purple")
    axes[4].set_title("Turnover")

    axes[5].axis("off")

    for ax in axes[:5]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(tearsheet_path, dpi=150)
    plt.close(fig)


def encode_image(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def df_to_html(df: pd.DataFrame, caption: str | None = None) -> str:
    if df.empty:
        return ""
    html = df.to_html(
        classes="table",
        border=0,
        float_format=lambda x: f"{x:0.4f}" if isinstance(x, Real) else x,
    )
    if caption:
        html = f"<h3>{caption}</h3>\n{html}"
    return html


def build_report(
    metrics: pd.Series,
    tearsheet_img: str,
    additional_images: dict[str, Path],
    tables: dict[str, pd.DataFrame],
    config_dump: str,
    output_path: Path,
) -> None:
    css = textwrap.dedent(
        """
        <style>
        body { font-family: Arial, sans-serif; margin: 20px; color: #222; }
        h1 { color: #2c3e50; }
        .table { border-collapse: collapse; margin-bottom: 20px; }
        .table th,
        .table td {
            border: 1px solid #ddd;
            padding: 6px 10px;
            text-align: right;
        }
        .table th { background-color: #f8f9fa; }
        .images { display: flex; flex-wrap: wrap; gap: 20px; }
        .images img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ccc;
            padding: 4px;
            background: #fafafa;
        }
        pre { background: #f0f0f0; padding: 10px; overflow-x: auto; }
        </style>
        """
    )

    metrics_df = metrics.to_frame("value")
    metrics_html = df_to_html(metrics_df, "Performance Summary")
    tables_html = "".join(
        df_to_html(df, title) for title, df in tables.items() if not df.empty
    )

    image_blocks = []
    if tearsheet_img:
        image_blocks.append(
            '<div><h3>Tearsheet</h3>'
            f'<img src="{tearsheet_img}" alt="Tearsheet"></div>'
        )
    for title, path in additional_images.items():
        encoded = encode_image(path)
        if encoded:
            image_blocks.append(
                f'<div><h3>{title}</h3><img src="{encoded}" alt="{title}"></div>'
            )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    footer = (
        "<footer style='margin-top:30px;font-size:0.85rem;'>"
        f"MacroTone v{__version__} • Released {__release_date__} "
        f"• Generated {timestamp}"
        "</footer>"
    )

    html = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <title>MacroTone Backtest Report</title>
    {css}
    </head>
    <body>
    <h1>MacroTone Backtest Report</h1>
    {metrics_html}
    {tables_html}
    <div class="images">
    {''.join(image_blocks)}
    </div>
    <h3>Configuration</h3>
    <pre>{config_dump}</pre>
    {footer}
    </body>
    </html>
    """
    output_path.write_text(html, encoding="utf-8")
    print(f"Saved HTML report -> {output_path}")


def optional_table(path: Path, index_col: str | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if index_col and index_col in df.columns:
        df = df.set_index(index_col)
    return df


def main() -> None:
    set_global_seed(DEFAULT_SEED)
    backtest_path = PRO / "backtest.parquet"
    if not backtest_path.exists():
        raise FileNotFoundError(f"Backtest results not found: {backtest_path}")

    bt = pd.read_parquet(backtest_path).sort_index()
    bt.index = pd.to_datetime(bt.index)
    metrics = compute_metrics(bt)

    market_series = get_market_series(bt.index)
    tearsheet_path = PRO / "tearsheet.png"
    draw_tearsheet(bt, tearsheet_path, market_series)

    # Preserve legacy plots for compatibility
    net_series = _pick_series(
        bt,
        ["net_ret", "port_ret", "strategy"],
        default=0.0,
    ).fillna(0.0)
    rf_series = _pick_series(
        bt,
        ["rf", CFG.project.rf_col],
        default=0.0,
    ).fillna(0.0)
    equity_curve = (1 + net_series).cumprod()
    market_equity = (1 + market_series.fillna(0.0)).cumprod()
    plt.figure()
    plt.plot(equity_curve.index, equity_curve.values, label="Strategy")
    plt.plot(market_equity.index, market_equity.values, label="Market")
    plt.legend()
    plt.title("Equity Curve")
    plt.tight_layout()
    plt.savefig(PRO / "equity_curve.png", dpi=120)
    plt.close()

    roll_sharpe = _rolling_sharpe_excess(net_series - rf_series)
    plt.figure()
    plt.plot(roll_sharpe.index, roll_sharpe.values)
    plt.title("Rolling Sharpe (min window 3, target 12)")
    plt.tight_layout()
    plt.savefig(PRO / "rolling_sharpe.png", dpi=120)
    plt.close()

    ablation = optional_table(PRO / "ablation_summary.csv", index_col="factor")
    sweep = optional_table(PRO / "sweep_summary.csv")
    regime_summary = optional_table(PRO / "regime_summary.csv", index_col="regime")
    regime_coverage = optional_table(PRO / "regime_coverage.csv")

    tables = {
        "Factor Ablation (Δ metrics)": ablation,
        "Hyperparameter Sweep": sweep,
        "Regime Summary": regime_summary,
        "Regime Coverage": regime_coverage,
    }
    panel_path = PRO / "panel.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        qa_summary = _data_qa_summary(panel)
        if not qa_summary.empty:
            tables["Data QA Summary"] = qa_summary

    additional_images = {
        "Ridge Coefficients": PRO / "ridge_coeffs.png",
        "Regime Boxplot": PRO / "regime_boxplot.png",
    }

    config_dump = yaml.safe_dump(
        CFG.model_dump(by_alias=True, exclude_none=True, mode="json"),
        sort_keys=False,
    )

    report_html_path = PRO / "report.html"
    build_report(
        metrics=metrics,
        tearsheet_img=encode_image(tearsheet_path),
        additional_images=additional_images,
        tables=tables,
        config_dump=config_dump,
        output_path=report_html_path,
    )

    print(f"Saved tearsheet -> {tearsheet_path}")


if __name__ == "__main__":
    main()
