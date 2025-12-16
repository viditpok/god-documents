export PYTHONPATH := src

.PHONY: all data nlp nlp_full features train backtest report clean fetch_fred fetch_ff fetch_fomc diag diagnostics_alpha walkforward costs sample_data ui quickstart run_all test lint fmt type ci audit

all:
	@$(MAKE) data
	@$(MAKE) nlp
	@$(MAKE) features
	@$(MAKE) train
	@$(MAKE) backtest
	@$(MAKE) report

data: fetch_fred fetch_ff fetch_fomc

fetch_fred:
	@if [ "$(FAST)" = "1" ]; then \
		echo "FAST=1 -> skipping FRED fetch (using cached parquet)."; \
	elif [ ! -f data/raw/fred_monthly.parquet ]; then \
		echo "Fetching FRED data..."; \
		uv run python -m macrotones.data.fetch_fred; \
	else \
		echo "FRED data already exists, skipping..."; \
	fi

fetch_ff:
	@if [ "$(FAST)" = "1" ]; then \
		echo "FAST=1 -> skipping FF download (using cached parquet)."; \
	else \
		uv run python -m macrotones.data.fetch_ff; \
	fi

fetch_fomc:
	@if [ "$(FAST)" = "1" ]; then \
		echo "FAST=1 -> skipping FOMC crawl."; \
	else \
		uv run python -m macrotones.data.fetch_fomc; \
	fi

nlp:
	@if [ "$(RUN_NLP)" = "0" ]; then \
		echo "RUN_NLP=0 -> skipping FinBERT scoring."; \
	else \
		uv run python -m macrotones.nlp.finbert_scoring; \
	fi

nlp_full:
	uv run python -m macrotones.data.fetch_fomc
	uv run python -m macrotones.nlp.finbert_scoring

features:
	uv run python -m macrotones.features.dataset

train:
	uv run python -m macrotones.models.train

backtest:
	uv run python -m macrotones.backtest.engine

report:
	uv run python -m macrotones.backtest.reports
	uv run python -m macrotones.backtest.report_regime || true
	uv run python -m macrotones.backtest.diagnostics

ppt:
	uv run python scripts/make_tearsheet_pptx.py

pdf:
	uv run python -m macrotones.utils.html_to_pdf

release:
	@echo "MacroTone v3.0.0 — $(shell date)"
	uv run pytest -q
	uv run make quickstart
	uv run make ppt
	uv run make pdf
	@if [ -f ui/screenshots/dataqa.png ]; then cp ui/screenshots/dataqa.png docs/ui/v3_dataqa.png; fi
	@echo "Artifacts ready under docs/ui and data/processed/"

diag:
	uv run python -m macrotones.backtest.diagnostics

diagnostics_alpha:
	uv run python -m macrotones.backtest.diagnostics_alpha

walkforward:
	uv run python -m macrotones.backtest.walkforward --train_years 10 --test_years 3 --step_years 3

costs:
	uv run python -m macrotones.backtest.sweep_costs --fixed-bps "0,5,10,25" --vol-k "0.5,1.0" --spread-bps "5,10"

sample_data:
	uv run python -m macrotones.tools.sample_data

ui:
	uv run streamlit run ui/app.py

quickstart:
	uv run python -m macrotones.quickstart

run_all: ## Full quickstart + UI (same as 'quickstart')
	$(MAKE) quickstart

test:
	uv run pytest -q

lint:
	uv run ruff check .

fmt:
	uv run black .

type:
	uv run mypy src

ci:
	$(MAKE) lint
	$(MAKE) type
	$(MAKE) test

audit:
	uv run python -m macrotones.tools.audit_data

clean:
	rm -rf data/interim/* data/processed/*
