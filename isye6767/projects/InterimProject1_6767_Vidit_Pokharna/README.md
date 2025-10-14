# Interim Project 1 – ISyE 6767 Dynamic Delta Hedging

## Files

* `include/delta_hedging/*.h`, `src/*.cpp` – Black–Scholes analytics, GBM simulation, hedging engine, data I/O, statistics, and CLI implementation
* `tests/test_black_scholes.cpp` – unit tests validating call price, delta, and implied-volatility inversion
* `scripts/plot_task1.py`, `scripts/plot_task2.py` – optional matplotlib plot generators that consume CSV outputs
* `data/interest.csv`, `data/sec_GOOG.csv`, `data/op_GOOG.csv` – provided risk-free, stock, and option datasets
* `outputs/` – auto-created directory containing Task 1 and Task 2 CSV exports
* `plots/` – destination for PNG plots produced by the helper scripts (created when scripts run)
* `report/report.pdf` – written project report ready for submission (generated from the Markdown source if desired)

## Build & Test

```bash
# build the CLI
g++ -std=c++17 -O2 \
    -Iinclude \
    src/main.cpp \
    src/black_scholes.cpp \
    src/data_io.cpp \
    src/hedging.cpp \
    src/simulation.cpp \
    src/stats.cpp \
    -o delta_hedging

# build the unit tests
g++ -std=c++17 -O2 \
    -Iinclude \
    tests/test_black_scholes.cpp \
    src/black_scholes.cpp \
    src/data_io.cpp \
    src/hedging.cpp \
    src/simulation.cpp \
    src/stats.cpp \
    -o test_black_scholes

# run the unit tests
./test_black_scholes
```

## Running the Workflows

```bash
# task 1 only (Monte Carlo validation)
./delta_hedging task1 --output outputs

# task 2 only (historical GOOG hedge)
./delta_hedging task2 --output outputs

# both tasks sequentially
./delta_hedging all --output outputs
```

Additional CLI options:

* `--paths <N>` – Task 1 sample path count (default 1000)
* `--seed <value>` – Task 1 RNG seed
* `--data-dir <dir>` – alternate data folder for Task 2 (default `data`)
* `--start-date`, `--end-date`, `--strike` – Task 2 overrides if exploring alternative windows

Each run writes the required CSV outputs into `outputs/`, including:

* `stock_paths.csv`, `option_prices.csv`, `deltas.csv`, `hedging_errors.csv`, `time_grid.csv`, `hedging_summary.csv`
* `result.csv` (date, stock, option, implied vol, delta, hedging error, PnL, PnL with hedge, tau, rate)

## Plot Generation

Plotting is handled outside the C++ binary; data is produced entirely in C++. The helper scripts require Python with `matplotlib` and `pandas`:

```bash
python scripts/plot_task1.py --outputs outputs --plots plots --paths 100
python scripts/plot_task2.py --outputs outputs --plots plots
```