#!/usr/bin/env python3
"""
compute_hitting_plus.py — Publish the Hitting+ (swingplus) engine's grades to Supabase.

Unlike CoLev, nothing is computed here: the swingplus engine already produces the
grades and publishes them as static JSON on the Hitting+ site. This script just joins
those grades to an MLBAM id (so fg-scatter can match them to FanGraphs rows by
xMLBAMID) and upserts every player-season into the `hitting_plus` table.

Run via GitHub Actions on a schedule, and any time the Hitting+ engine re-publishes.
Idempotent: upserts all seasons found, keyed on (mlbamid, season).

Required env vars:
  SUPABASE_URL         — e.g. https://xxxx.supabase.co
  SUPABASE_SERVICE_KEY — service role key (for upsert)

Optional env vars (default to the live Hitting+ deployment):
  HITTING_PLUS_DATA_URL — swingplus_latest.json
  HITTING_PLUS_INFO_URL — player_info.json (name -> MLBAM id)
"""

import os

import requests
from supabase import create_client

# ── Config ─────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

DATA_URL = os.environ.get(
    "HITTING_PLUS_DATA_URL", "https://hitting-plus.vercel.app/data/swingplus_latest.json"
)
INFO_URL = os.environ.get(
    "HITTING_PLUS_INFO_URL", "https://hitting-plus.vercel.app/data/player_info.json"
)

# swingplus grade key -> Supabase column
GRADE_COLUMNS = {
    "Hitting+": "hitting_plus",
    "Decision+": "decision",
    "Timing+": "timing",
    "Contact+": "contact",
    "Power+": "power",
}


def clean(v):
    """Coerce to float, turning NaN/Inf/missing into None for JSON serialisation."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def main():
    # ── 1. Load the published Hitting+ grades and the name -> MLBAM id map ─────────
    print(f"[1/3] Fetching Hitting+ data...\n      {DATA_URL}\n      {INFO_URL}")
    data = requests.get(DATA_URL, timeout=30).json()
    info = requests.get(INFO_URL, timeout=30).json()
    players = data["players"]
    print(f"      {len(players)} player-seasons, {len(info)} resolved names")

    # ── 2. Join each player-season to an MLBAM id ─────────────────────────────────
    print("[2/3] Joining to MLBAM ids...")
    records = []
    missing = set()
    for p in players:
        name = p.get("player_name")
        entry = info.get(name)
        mlbamid = entry.get("id") if entry else None
        if not mlbamid:
            missing.add(name)
            continue
        row = {
            "name": name,
            "mlbamid": int(mlbamid),
            "season": int(p["game_year"]),
        }
        for grade, col in GRADE_COLUMNS.items():
            row[col] = clean(p.get(grade))
        records.append(row)
    print(f"      {len(records)} rows to upsert ({len(missing)} names without an MLBAM id)")

    # ── 3. Upsert to Supabase ─────────────────────────────────────────────────────
    print(f"[3/3] Upserting {len(records)} rows to Supabase...")
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    # Chunk to keep request bodies reasonable.
    CHUNK = 500
    for i in range(0, len(records), CHUNK):
        sb.table("hitting_plus").upsert(
            records[i : i + CHUNK], on_conflict="mlbamid,season"
        ).execute()
    seasons = sorted({r["season"] for r in records})
    print(f"      Done — {len(records)} rows upserted for seasons {seasons}")


if __name__ == "__main__":
    main()
