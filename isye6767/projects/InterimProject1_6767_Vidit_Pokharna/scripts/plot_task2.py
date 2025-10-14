import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def plot_pnl_comparison(df: pd.DataFrame, output: Path):
    plt.figure(figsize=(11, 6))
    plt.plot(df["date"], df["pnl_call"], label="PNL (no hedge)", linewidth=1.6)
    plt.plot(
        df["date"],
        df["pnl_with_hedge"],
        label="PNL (with hedge)",
        linestyle="--",
        linewidth=1.6,
    )
    plt.axhline(0.0, color="grey", linestyle=":", linewidth=1.0)
    plt.title("PNL vs PNL with Hedge")
    plt.xlabel("Date")
    plt.ylabel("PNL")
    plt.legend(loc="lower left")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_implied_vol_delta(df: pd.DataFrame, output: Path):
    fig, ax_left = plt.subplots(figsize=(11, 6))
    ax_right = ax_left.twinx()

    ax_left.plot(df["date"], df["implied_vol"], label="Implied Vol (annual)", linewidth=1.6)
    ax_right.plot(
        df["date"],
        df["delta"],
        label="Delta",
        linestyle="--",
        linewidth=1.6,
        color="tab:orange",
    )

    ax_left.set_title("Implied Vol & Delta")
    ax_left.set_xlabel("Date")
    ax_left.set_ylabel("Implied Vol (annual)")
    ax_right.set_ylabel("Delta")

    ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left")

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Task 2 CSV outputs.")
    parser.add_argument("--outputs", default="outputs", help="Directory containing result.csv.")
    parser.add_argument("--plots", default="plots", help="Output directory for PNG figures.")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs)
    plots_dir = Path(args.plots)
    ensure_dir(plots_dir)

    df = load_results(outputs_dir / "result.csv")
    plot_pnl_comparison(df, plots_dir / "pnl_vs_hedge.png")
    plot_implied_vol_delta(df, plots_dir / "implied_vol_delta.png")
    print(f"Task 2 plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
