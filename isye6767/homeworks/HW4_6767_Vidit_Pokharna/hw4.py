import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def strategy_core(
    prices: np.ndarray, initial_cash: float = 10000.0, lot_size: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prices = np.asarray(prices, dtype=float)
    n = prices.size
    signals = np.zeros(n, dtype=int)
    positions = np.zeros(n, dtype=int)
    account_values = np.zeros(n, dtype=float)
    shares = 0
    cash = initial_cash
    max_shares = 2 * lot_size
    for i in range(n):
        action = 0
        if shares > 0 and i == n - 1:
            action = -1
        else:
            if shares > 0 and i >= 2 and np.all(np.diff(prices[i - 2 : i + 1]) < 0):
                action = -1
            elif (
                shares < max_shares
                and i >= 3
                and np.all(np.diff(prices[i - 3 : i + 1]) > 0)
            ):
                action = 1
        if action == 1:
            buy_shares = min(lot_size, max_shares - shares)
            cost = buy_shares * prices[i]
            if buy_shares > 0 and cash >= cost:
                shares += buy_shares
                cash -= cost
                signals[i] = 1
        elif action == -1 and shares > 0:
            cash += shares * prices[i]
            shares = 0
            signals[i] = -1
        positions[i] = shares
        account_values[i] = cash + shares * prices[i]
    return signals, positions, account_values


def _plot_cumulative_pnl(
    dates: np.ndarray,
    account_values: np.ndarray,
    output_dir: Path,
    initial_cash: float = 10000.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pnl = account_values - initial_cash
    try:
        x = np.array(dates, dtype="datetime64[D]")
    except (TypeError, ValueError):
        x = np.arange(account_values.size)
    fig, ax = plt.subplots()
    ax.plot(x, pnl, linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L")
    ax.set_title("Trading Strategy Cumulative P&L")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_pnl.png")
    plt.close(fig)


def trading_strategy(csv_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_file, delimiter=",", names=True, dtype=None, encoding=None)
    dates = np.asarray(data["Date"], dtype=str)
    closes = np.asarray(data["Close"], dtype=float)
    signals, positions, account_values = strategy_core(closes)
    pnl = account_values[-1] - 10000.0 if account_values.size else 0.0
    print(f"Cumulative P&L: {pnl:.2f}")
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trading_results.csv"
    if closes.size:
        stacked = np.column_stack(
            (
                dates.astype(str),
                signals.astype(str),
                positions.astype(str),
                account_values.astype(str),
            )
        )
        np.savetxt(
            output_path,
            stacked,
            fmt="%s",
            delimiter=",",
            header="Date,Signal,Position,AccountValue",
            comments="",
        )
    else:
        np.savetxt(
            output_path,
            np.empty((0, 4), dtype=str),
            fmt="%s",
            delimiter=",",
            header="Date,Signal,Position,AccountValue",
            comments="",
        )
    _plot_cumulative_pnl(dates, account_values, output_dir)
    print(f"Saved trading_results.csv to {output_path}")
    print(f"Saved cumulative P&L chart to {output_dir / 'cumulative_pnl.png'}")
    return signals, positions, account_values


def min_initial_energy(maze: np.ndarray) -> int:
    grid = np.asarray(maze, dtype=int)
    if grid.ndim != 2:
        raise ValueError("maze must be 2D")
    m, n = grid.shape
    dp = np.zeros((m, n), dtype=int)
    dp[-1, -1] = max(1, 1 - grid[-1, -1])
    for i in range(m - 2, -1, -1):
        required = dp[i + 1, -1] - grid[i, -1]
        dp[i, -1] = max(1, required)
    for j in range(n - 2, -1, -1):
        required = dp[-1, j + 1] - grid[-1, j]
        dp[-1, j] = max(1, required)
    for i in range(m - 2, -1, -1):
        for j in range(n - 2, -1, -1):
            required = min(dp[i + 1, j], dp[i, j + 1]) - grid[i, j]
            dp[i, j] = max(1, required)
    return int(dp[0, 0])


def trading_strategy_example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prices = np.array(
        [98, 100, 102, 104, 103, 101, 99, 100, 102, 104, 106, 107, 105], dtype=float
    )
    return strategy_core(prices)


def trading_strategy_test_case() -> dict[str, np.ndarray]:
    prices = np.array(
        [50, 52, 54, 56, 55, 53, 51, 49, 47, 48, 50, 52, 54, 53], dtype=float
    )
    signals, positions, account_values = strategy_core(prices)
    return {
        "prices": prices,
        "signals": signals,
        "positions": positions,
        "account_values": account_values,
    }


def min_initial_energy_example() -> int:
    maze = np.array([[-2, -3, 3], [-5, -10, 1], [10, 30, -5]])
    return min_initial_energy(maze)


def min_initial_energy_test_case() -> dict[str, np.ndarray | int]:
    maze = np.array([[1, -4, 3], [-2, -2, -2], [5, -3, -1]])
    return {
        "maze": maze,
        "initial_energy": min_initial_energy(maze),
    }


if __name__ == "__main__":
    print("=== Trading Strategy: CSV Data ===")
    signals, positions, account_values = trading_strategy("aapl.csv")
    print("Signals:", signals)
    print("Positions:", positions)
    print("Account values:", account_values)
    print()
    print("=== Trading Strategy (Task 1) ===")
    example_signals, example_positions, example_values = trading_strategy_example()
    print(
        "Example prices:",
        np.array(
            [98, 100, 102, 104, 103, 101, 99, 100, 102, 104, 106, 107, 105], dtype=float
        ),
    )
    print("Example signals:", example_signals)
    print("Example positions:", example_positions)
    print("Example account values:", example_values)
    custom_trade = trading_strategy_test_case()
    print(
        "\nTest prices:",
        np.array([50, 52, 54, 56, 55, 53, 51, 49, 47, 48, 50, 52, 54, 53], dtype=float),
    )
    print("Test signals:", custom_trade["signals"])
    print("Test positions:", custom_trade["positions"])
    print("Test account values:", custom_trade["account_values"])
    print()
    print("=== Energy Maze (Task 2) ===")
    print("Example maze:\n", np.array([[-2, -3, 3], [-5, -10, 1], [10, 30, -5]]))
    print("Example maze energy:", min_initial_energy_example())
    custom_maze = min_initial_energy_test_case()
    print("\nTest maze:\n", custom_maze["maze"])
    print("Test maze initial energy:", custom_maze["initial_energy"])
