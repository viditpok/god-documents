# MacroTone: ML + NLP for Factor Allocation

MacroTone is a professional-grade factor allocation system that blends distinct macroeconomic signals (derived regimes) with NLP insights (FinBERT + LLM analysis of FOMC minutes) to dynamically rotate between investment factors.

## Project Summary

**Factor Timing Using Macro-NLP Regimes**  
Contributors: Vidit Pokharna, Devang Ajmera, Osho Sharma

### Strategy Overview
Static factor investing often underperforms during regime shifts (e.g., Value crashes in recessions, Momentum fails in reversals). MacroTone addresses this by:
1.  **Identifying Regimes:** Using `Macro Regime Index (MRI)` from economic data and `NLP Sentiment` from Central Bank communications.
2.  **Predicting Returns:** Using an ensemble of **Ridge Regression** (linear trend) and **XGBoost** (non-linear interaction) to forecast factor premiums.
3.  **Dynamic Allocation:** Rotating capital into the **Top-2** highest conviction factors (Top-K) while managing volatility (Target Vol: 15%).

### Investment Universe (Tradeable ETFs)

| Factor | ETF Ticker | Name |
| :--- | :--- | :--- |
| **Value** | **VTV** | Vanguard Value ETF |
| **Size** | **IJR** | iShares Core S&P Small-Cap ETF |
| **Momentum** | **MTUM** | iShares MSCI USA Momentum Factor ETF |
| **Quality** | **QUAL** | iShares MSCI USA Quality Factor ETF |
| **Low Vol** | **USMV** | iShares MSCI USA Min Vol Factor ETF |
| *Cash* | *BIL/SHV* | SPDR/iShares Short Treasury (Risk-free proxy) |

---

## Key Features

### 1. Advanced Signal Processing
*   **Macro Regime Index (MRI):** A PCA-based composite of Inflation (CPI), Growth (GDP, IndPro), and Liquidity (Term Spreads, High Yield).
*   **NLP Intelligence:** 
    *   **FinBERT** for sentiment polarity scoring of FOMC minutes.
    *   **LLM Blending:** A dynamic "Rationale" engine that explains *why* the strategy is shifting (e.g., "Hawkish fed tone suggests defensive rotation").

### 2. Robust Allocation Engine
*   **Method:** **Top-K (K=2)**. Concentration creates higher conviction than softmax smearing.
*   **Risk Control:** **15% Annualized Volatility** Target using a cash toggle.
*   **Execution:** Monthly rebalancing with realistic **10bps transaction costs**.

### 3. Professional Dashboard (v3.2)
The Streamlit UI has been overhauled for a narrative-driven presentation:
*   **Project Documentation:** Built-in "Whitepaper" tab detailing architecture and math.
*   **Executive Summary:** Clean KPIs, Equity Curve, and Drawdown analysis (No redundant metrics).
*   **Strategy Analysis:** **Stacked Area Charts** for allocation history and clear "Top Holdings" visualization.
*   **Macro & NLP Intelligence:** Focused timeline of the MRI and NLP Regime scores.
*   **Scenario Simulator:** Stress-test the strategy against CPI/Yield shocks.

---

## Quickstart

Run the full pipeline (Data -> Model -> Backtest -> UI) with a single command:

```bash
uv run make quickstart
```

### Manual Commands

```bash
# 1. Fetch Data (FRED + Yahoo Finance ETFs)
uv run make data

# 2. Run NLP Pipeline (FOMC Crawl + FinBERT)
uv run make nlp

# 3. Train Models & Run Backtest
uv run make backtest

# 4. Launch Dashboard
uv run make ui
```

## Results Highlights (v3.2)

| Metric | Performance |
| :--- | :--- |
| **Strategy** | **Top-2 Dynamic Rotation** |
| **Benchmark** | SPY (S&P 500) |
| **Sharpe Ratio** | **> 1.0** (Targeted) |
| **Win Rate** | **~58%** (Monthly) |
| **Turnover** | **< 20%** (Low impact) |

---

## Technical Stack

*   **Language:** Python 3.11+
*   **Orchestration:** `Makefile`, `uv` (dependency management)
*   **ML Modules:** `scikit-learn` (Ridge), `xgboost` (GradBoost), `transformers` (FinBERT)
*   **Data:** `pandas`, `polars` (fast IO), `fredapi`, `yfinance`
*   **UI:** `streamlit`, `plotly` (interactive viz)
*   **Quality:** `ruff` (linting), `pytest` (testing), `mypy` (typing)

## Reproducibility
*   **Seed:** 42 (Global determinism for ML & Backtests)
*   **Config:** `config/project.yaml` controls all hyperparameters (lookback windows, regularization, allocation logic).

---

**© 2025 MacroTone Project Team**
