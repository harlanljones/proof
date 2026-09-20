# PROOF — Development Agent Guide

## Project intent

PROOF (Probabilistic Rest-Of-season Objective Forecasts) is an open,
reproducible MLB rest-of-season projection system: empirical-Bayes shrinkage
of year-to-date performance toward a player prior, with a walk-forward
backtest harness that treats honest evaluation as the main artifact. It is
a mini Steamer/ZiPS, not a black box — every number in the README/writeup
must be reproducible from pinned snapshots.

## Where to work

| Path | Purpose |
|---|---|
| `ingest/` | Snapshot pulls from pybaseball (`data/snapshots/`, pinned) |
| `features/` | YTD feature construction from snapshots |
| `models/` | Shrinkage / projection core (per-component k, uncertainty model) |
| `backtest/` | Walk-forward harness, tuning loop, scoring vs holdouts |
| `app/` | Projection output assembly consumed by the dashboard/site |
| `dashboard.py` | Streamlit live dashboard (as-of-yesterday, local TZ) |
| `prerender.py` | Static site regeneration → `dist-site/` (Cloudflare Workers assets) |
| `data/snapshots/` | TRACKED pinned parquet — the reproducibility contract |
| `data/live/` | Regenerable daily pulls — gitignored, never commit |
| `tests/` | pytest suite |
| `writeup/` | Methodology prose |

## Non-negotiable boundaries

- **`data/snapshots/` is pinned and tracked.** Never mutate a snapshot in
  place; reproducibility of the headline backtest depends on it.
- **No test-set tuning.** Shrinkage constants are selected on 2023–24 and
  validated on the 2025 holdout the tuning loop never saw. Never touch the
  tuning/validation split without an explicit user decision.
- **The README/writeup numbers come from the code.** If a change moves a
  headline result, re-run the backtest and update the numbers — never edit
  the prose to match a hoped-for result.
- **Never commit** `data/live/`, `dist-site/`, `.env`,
  `.streamlit/secrets.toml`, or any API token.

## Commands

```bash
make test              # pytest (uv run) — must be green before declaring done
make lint              # ruff check + ruff format --check
make backtest          # full walk-forward (hitters + pitchers), pinned snapshots
make backtest-hitters  # hitters walk-forward only
make backtest-pitchers # pitchers walk-forward only
make tune              # shrinkage grid on 2023–24, validate on 2025 holdout
make ingest            # pull any missing snapshots (30 s spacing, safe to re-run)
make dashboard         # streamlit live dashboard
make prerender         # regenerate static site into dist-site/
make deploy            # prerender + npx wrangler deploy (Cloudflare Workers)
```

Modules run from the repo root via `uv run python -m ...` — this is not an
installable package.

## Quality gate

`make test && make lint` before declaring any change done. Re-run the
relevant backtest after any change to `models/`, `features/`, or
`backtest/` — if the headline numbers move, say so explicitly.

## Code style

Python 3.12+, numpy/pandas/scipy. Match the surrounding code:

- `ruff format` formatting; ruff lint is the gate (per-file ignores in
  pyproject are deliberate — e.g. `DTZ011` on the live dashboard).
- Comments explain *why*, especially around statistical choices
  (why shrink toward a prior, why a particular k).
- Pure logic in small modules with pytest tests beside them in `tests/`.

## Commits

Conventional Commits are enforced by the repo's commit-msg hook:
`feat: ...`, `fix: ...`, `chore: ...`. Plain imperative subject.

## Data safety

- The test suite needs no network credentials; a test that does belongs
  behind a snapshot fixture, never in `make test`.
- Timezone-aware by design: the dashboard is "as of yesterday, local".
  Don't "fix" the deliberate `DTZ011` ignores.
