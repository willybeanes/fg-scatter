#!/usr/bin/env python3
"""
compute_colev.py — Compute CoLev for the current MLB season and upsert to Supabase.

Run via GitHub Actions on a weekly schedule during the season.
Only processes the current season; historical data is already in Supabase.

Required env vars:
  SUPABASE_URL         — e.g. https://xxxx.supabase.co
  SUPABASE_SERVICE_KEY — service role key (for upsert)
"""

import os
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd
from pybaseball import pitching_stats, statcast
from sklearn.preprocessing import MinMaxScaler
from supabase import create_client

warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────────────
SEASON       = 2026
SEASON_START = '2026-03-27'
SEASON_END   = date.today().strftime('%Y-%m-%d')
MIN_IP       = 50
MIN_GS_SHARE = 0.50

RUN_VALUE_WEIGHTS = {
    (0,0): 0.00, (0,1):-0.06, (0,2):-0.09,
    (1,0): 0.07, (1,1):-0.02, (1,2):-0.07,
    (2,0): 0.14, (2,1): 0.06, (2,2):-0.01,
    (3,0): 0.24, (3,1): 0.15, (3,2): 0.05,
}
POLARITY_WEIGHTS = {
    (0,0):0, (0,1):-1, (0,2):-1,
    (1,0):1, (1,1): 0, (1,2):-1,
    (2,0):1, (2,1): 1, (2,2): 0,
    (3,0):1, (3,1): 1, (3,2): 1,
}

SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_SERVICE_KEY']

# ── 1. Load FanGraphs qualifying starters via pybaseball ───────────────────────
print(f'[1/5] Fetching FanGraphs pitching stats for {SEASON}...')
fg = pitching_stats(SEASON, qual=1)
print(f'      {len(fg)} pitchers, columns: {list(fg.columns)[:15]}')

# pybaseball returns xMLBAMID from the FanGraphs API — find it under various names
mlbam_col = next((c for c in ['xMLBAMID', 'MLBAMID', 'mlbamid'] if c in fg.columns), None)
if mlbam_col is None:
    raise RuntimeError(f'No MLBAM ID column found in FanGraphs data. Columns: {list(fg.columns)}')
fg = fg.rename(columns={mlbam_col: 'MLBAMID'})
fg['MLBAMID'] = pd.to_numeric(fg['MLBAMID'], errors='coerce')

name_col = next((c for c in ['Name', 'PlayerName', 'name'] if c in fg.columns), None)
if name_col and name_col != 'Name':
    fg = fg.rename(columns={name_col: 'Name'})

# Filter to qualifying starters (min IP + majority of appearances as starter)
if 'IP' in fg.columns:
    fg = fg[pd.to_numeric(fg['IP'], errors='coerce') >= MIN_IP]
if 'GS' in fg.columns and 'G' in fg.columns:
    fg['gs_share'] = (pd.to_numeric(fg['GS'], errors='coerce') /
                      pd.to_numeric(fg['G'],  errors='coerce').replace(0, np.nan))
    fg = fg[fg['gs_share'] >= MIN_GS_SHARE]

fg['season'] = SEASON
sp_ids       = set(fg['MLBAMID'].dropna().astype(int).tolist())
id_to_name   = dict(zip(fg['MLBAMID'].astype(int), fg['Name']))
print(f'      {len(sp_ids)} qualifying starters after filters')

# ── 2. Pull Statcast data month-by-month ───────────────────────────────────────
print(f'\n[2/5] Pulling Statcast data {SEASON_START} → {SEASON_END}...')

# Build month-by-month date ranges up to today
month_ranges = []
current = pd.Timestamp(SEASON_START)
end     = pd.Timestamp(SEASON_END)
while current <= end:
    month_end = min(current + pd.offsets.MonthEnd(0), end)
    month_ranges.append((current.strftime('%Y-%m-%d'), month_end.strftime('%Y-%m-%d')))
    current = month_end + pd.Timedelta(days=1)

KEEP_COLS = ['pitcher', 'player_name', 'balls', 'strikes', 'delta_run_exp']
frames = []

for s, e in month_ranges:
    t0 = time.time()
    try:
        df = statcast(start_dt=s, end_dt=e, verbose=False)
        if df is None or len(df) == 0:
            print(f'      {s}: no data')
            continue
        avail = [c for c in KEEP_COLS if c in df.columns]
        df = df[df['pitcher'].isin(sp_ids)][avail].copy()
        df['balls']         = pd.to_numeric(df['balls'],         errors='coerce')
        df['strikes']       = pd.to_numeric(df['strikes'],       errors='coerce')
        df['delta_run_exp'] = pd.to_numeric(df['delta_run_exp'], errors='coerce')
        df = df.dropna(subset=['balls', 'strikes', 'pitcher'])
        df['balls']   = df['balls'].astype(int)
        df['strikes'] = df['strikes'].astype(int)
        df['MLBAMID'] = df['pitcher'].astype(int)
        frames.append(df)
        print(f'      {s} → {e}: {len(df):,} pitches ({time.time()-t0:.0f}s)')
    except Exception as ex:
        print(f'      {s}: ERROR — {ex}')

if not frames:
    raise RuntimeError('No Statcast data returned — aborting')

pitches = pd.concat(frames, ignore_index=True)
print(f'      Total: {len(pitches):,} pitches from {pitches["MLBAMID"].nunique()} pitchers')

# ── 3. Compute CoLev ───────────────────────────────────────────────────────────
print(f'\n[3/5] Computing CoLev...')

def compute_colev(pdf):
    total = len(pdf)
    if total == 0:
        return None
    cf = pdf.groupby(['balls', 'strikes']).size().reset_index(name='n')
    cf['share'] = cf['n'] / total
    cf['rv']    = cf.apply(lambda r: RUN_VALUE_WEIGHTS.get((int(r.balls), int(r.strikes)), 0.0), axis=1)
    cf['pol']   = cf.apply(lambda r: POLARITY_WEIGHTS.get( (int(r.balls), int(r.strikes)), 0),   axis=1)
    colev       = -(cf['share'] * cf['rv']).sum()
    colev_pol   = -(cf['share'] * cf['pol']).sum()
    dre = (-pdf['delta_run_exp'].mean(skipna=True)
           if 'delta_run_exp' in pdf.columns else np.nan)
    cfd = cf.set_index(['balls', 'strikes'])['share'].to_dict()
    return {
        'total_pitches':  total,
        'CoLev':          round(colev,     5),
        'CoLev_polarity': round(colev_pol, 5),
        'CoLev_dRE':      round(dre, 5) if not np.isnan(dre) else None,
    }

colev_rows = []
for mlbam_id, pdf in pitches.groupby('MLBAMID'):
    result = compute_colev(pdf)
    if result is None:
        continue
    colev_rows.append({
        'MLBAMID': mlbam_id,
        'Name':    id_to_name.get(mlbam_id, str(mlbam_id)),
        'season':  SEASON,
        **result,
    })

colev_df = pd.DataFrame(colev_rows)
print(f'      {len(colev_df)} pitcher-seasons computed')

# ── 4. Merge with FanGraphs and scale ─────────────────────────────────────────
print(f'\n[4/5] Merging with FanGraphs stats and scaling...')

FG_KEEP = [c for c in ['Name', 'MLBAMID', 'season', 'IP', 'ERA', 'GS', 'G', 'K-BB%', 'FIP', 'xFIP', 'SIERA'] if c in fg.columns]
merged = colev_df.merge(fg[FG_KEEP], on=['MLBAMID', 'season'], how='inner', suffixes=('', '_fg'))
if 'Name_fg' in merged.columns:
    merged['Name'] = merged['Name_fg']
    merged = merged.drop(columns=['Name_fg'])
print(f'      {len(merged)} qualifying pitcher-seasons after merge')

# Fit scaler on historical + new data so the 1–5 scale stays consistent
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
hist = sb.table('colev').select('colev').neq('season', SEASON).execute()
hist_vals = [r['colev'] for r in hist.data if r['colev'] is not None]
all_vals  = hist_vals + merged['CoLev'].tolist()

scaler = MinMaxScaler(feature_range=(1.0, 5.0))
scaler.fit(np.array(all_vals).reshape(-1, 1) * -1)
merged['CoLev_scaled'] = scaler.transform(
    -merged['CoLev'].values.reshape(-1, 1)
).round(2).flatten()

print(f'      Scaler fit on {len(hist_vals)} historical + {len(merged)} new values')

# ── 5. Upsert to Supabase ──────────────────────────────────────────────────────
print(f'\n[5/5] Upserting {len(merged)} rows to Supabase...')

rows = merged[['Name', 'MLBAMID', 'season', 'CoLev', 'CoLev_scaled', 'CoLev_dRE']].copy()
rows.columns = ['name', 'mlbamid', 'season', 'colev', 'colev_scaled', 'colev_dre']
rows['mlbamid'] = rows['mlbamid'].astype(int)

# Replace NaN with None for JSON serialisation
records = rows.to_dict('records')
for r in records:
    for k, v in r.items():
        if isinstance(v, float) and np.isnan(v):
            r[k] = None

sb.table('colev').upsert(records, on_conflict='mlbamid,season').execute()
print(f'      Done — {len(records)} rows upserted for season {SEASON}')
