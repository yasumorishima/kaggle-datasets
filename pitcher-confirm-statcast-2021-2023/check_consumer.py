"""Read the built files the way the frozen pitcher notebook does (run with Kaggle's pandas, 2.3.3):
the 23 columns, game_date to datetime, game_type == "R", no duplicate pitch keys, pitch_type >= 90%
filled per season, and (with --full) the season's last date in [Sep 15, Oct 6]. Prints each column's
pandas dtype beside a reference file's (a dataset4 statcast_2024.parquet) and fails if any differs.

    python check_consumer.py <folder with statcast_2021..2023.parquet> <reference statcast_2024.parquet> [--full]
"""

import sys

import pandas as pd
import pyarrow.parquet as pq

COLS = ["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number", "pitch_type", "release_speed",
        "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z", "arm_angle", "pfx_x", "pfx_z",
        "zone", "description", "events", "type", "launch_speed", "woba_denom", "inning", "game_type", "spin_axis"]
folder, ref = sys.argv[1], sys.argv[2]
full = "--full" in sys.argv[3:]
print("pandas", pd.__version__)
r = pd.read_parquet(ref, columns=COLS)
rs = pq.read_schema(ref)
bad = []
frames = []
for y in (2021, 2022, 2023):
    p = f"{folder}/statcast_{y}.parquet"
    s = pq.read_schema(p)
    if s.names != COLS:
        bad.append(f"{y}: columns {s.names}")
    d = pd.read_parquet(p, columns=COLS)
    t = pd.DataFrame({"arrow": [str(s.field(c).type) for c in COLS], "arrow_ref": [str(rs.field(c).type) for c in COLS],
                      "pandas": d.dtypes.astype(str).to_numpy(), "pandas_ref": r.dtypes.astype(str).to_numpy(),
                      "nulls": d.isna().sum().to_numpy(), "nulls_ref": r.isna().sum().to_numpy()}, index=COLS)
    t["same"] = t["pandas"] == t["pandas_ref"]
    print(f"== {y}: {len(d):,} rows (reference {len(r):,})\n{t.to_string()}")
    bad += [f"{y}: {c} {t.at[c, 'pandas']} != {t.at[c, 'pandas_ref']}" for c in COLS if not t.at[c, "same"]]
    d["season"] = y
    frames.append(d)
out = pd.concat(frames, ignore_index=True)
out["game_date"] = pd.to_datetime(out["game_date"])
n0 = len(out)
out = out[out.game_type == "R"]
print(f"game_type == 'R': {len(out):,} of {n0:,} rows kept")
if out.duplicated(["game_pk", "at_bat_number", "pitch_number"]).any():
    bad.append("duplicate pitches")
fill = out.groupby("season")[["pitch_type", "spin_axis", "arm_angle", "pfx_x", "pfx_z"]].apply(lambda g: g.notna().mean())
print(fill.round(4).to_string())
if not (fill.pitch_type >= 0.90).all():
    bad.append("pitch_type under 90% filled")
last = out.groupby("season").game_date.max()
print("last game_date per season:", {y: str(d.date()) for y, d in last.items()})
if full:
    for y, d in last.items():
        if not pd.Timestamp(y, 9, 15) <= d <= pd.Timestamp(y, 10, 6):
            bad.append(f"{y}: last date {d.date()}")
print("FAIL: " + "; ".join(bad) if bad else "OK: the frozen loader's checks pass and every dtype matches the reference")
sys.exit(1 if bad else 0)
