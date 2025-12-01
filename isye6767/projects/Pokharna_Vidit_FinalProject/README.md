# ISyE 6767 Statistical Arbitrage Project

This repository implements the Avellaneda & Lee (2010) PCA-based statistical arbitrage strategy
for the ISyE 6767 final project. The code:

1. Loads the hourly FTX token data (`data/coin_all_prices_full.csv`) and the evolving top-40 universe
2. Runs rolling 240-hour PCA to estimate the top-two eigen-portfolios each hour
3. Regresses token returns on the factor returns, fits OU processes to residuals, and computes s-scores
4. Generates trading signals via the four-threshold rule, simulates 1-share trades with $100k capital,
   and records hourly equity/returns
5. Produces every deliverable in the brief (CSV outputs, plots, Sharpe ratio, and MDD)

## Environment

Install dependencies with `pip install -r requirements.txt` using your preferred Python environment

## Running the pipeline

Run `python main.py` from the project root after dependencies are installed

Outputs are written to:

- `outputs/csv/task1a_1.csv` & `task1a_2.csv` – leading eigen-portfolio weights per hour
- `outputs/csv/trading_signal.csv` – action label per token/hour
- `outputs/figures/` – required plots (Task 1b, Task 2 snapshots, Task 3 s-scores, Task 4 performance)

The console will also display the Sharpe ratio and maximum drawdown computed from the hourly strategy returns
