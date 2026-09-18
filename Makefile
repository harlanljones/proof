.PHONY: backtest backtest-hitters backtest-pitchers ingest tune lint test dashboard

# The reproducible artifact: runs both walk-forwards from pinned snapshots.
backtest: backtest-hitters backtest-pitchers

backtest-hitters:
	uv run python -m backtest.run

backtest-pitchers:
	uv run python -m backtest.run --pitchers

# Pulls any missing snapshots politely (30 s spacing; safe to re-run).
ingest:
	uv run python -m ingest.prefetch

# Hitter shrinkage tuning: grid on 2023-24, validate on 2025 holdout.
tune:
	uv run python -m backtest.run --tune

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run python -m pytest tests/ -q

# Live dashboard: as-of-yesterday projections (data pinned per day).
dashboard:
	uv run streamlit run dashboard.py
