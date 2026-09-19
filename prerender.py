"""Pre-render the PROOF dashboard into a static site for Cloudflare Workers assets.

Runs the live projections once (as-of yesterday), writes the leaderboards and
riser/faller chart data as JSON, and copies the static frontend into `dist-site`.
The interactive behaviour (tabs, sorting, filtering, search) happens client-side.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import pandas as pd

from app.projections import build_all

ROOT = Path(__file__).resolve().parent
SITE_SRC = ROOT / "site"
OUT_DIR = ROOT / "dist-site"


def _records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].round(4)
    return json.loads(out.to_json(orient="records", date_format="iso"))


def _risers(df: pd.DataFrame) -> list[dict]:
    top = pd.concat([df.nlargest(8, "delta"), df.nsmallest(8, "delta")])
    top = top.assign(direction=["riser"] * 8 + ["faller"] * 8)
    keep = [c for c in top.columns if c not in {"lo80", "hi80"}]
    return _records(top[keep])


def build_site(out_dir: Path) -> dict:
    as_of = dt.date.today() - dt.timedelta(days=1)
    hitters, pitchers = build_all(as_of)

    data_dir = out_dir / "proof" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    hitter_meta = {
        "as_of": as_of.isoformat(),
        "column": "wOBA",
        "playing_label": "PA",
    }
    pitcher_meta = {
        "as_of": as_of.isoformat(),
        "column": "FIP",
        "playing_label": "BF",
    }

    (data_dir / "hitters.json").write_text(
        json.dumps({"meta": hitter_meta, "rows": _records(hitters)}), encoding="utf-8"
    )
    (data_dir / "pitchers.json").write_text(
        json.dumps({"meta": pitcher_meta, "rows": _records(pitchers)}), encoding="utf-8"
    )
    (data_dir / "risers_hitters.json").write_text(
        json.dumps(_risers(hitters)), encoding="utf-8"
    )
    (data_dir / "risers_pitchers.json").write_text(
        json.dumps(_risers(pitchers)), encoding="utf-8"
    )
    return {"hitters": len(hitters), "pitchers": len(pitchers), "as_of": as_of.isoformat()}


def main() -> None:
    parser = argparse.ArgumentParser(prog="proof-prerender")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    proof_dir = out / "proof"
    proof_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SITE_SRC / "hub.html", out / "index.html")
    shutil.copy2(SITE_SRC / "proof" / "index.html", proof_dir / "index.html")
    shutil.copy2(SITE_SRC / "proof" / "style.css", proof_dir / "style.css")
    shutil.copy2(SITE_SRC / "proof" / "app.js", proof_dir / "app.js")
    shutil.copy2(SITE_SRC / "404.html", out / "404.html")

    summary = build_site(out)
    print(
        f"site ready at {out} — hitters={summary['hitters']} "
        f"pitchers={summary['pitchers']} as-of={summary['as_of']}"
    )


if __name__ == "__main__":
    main()
