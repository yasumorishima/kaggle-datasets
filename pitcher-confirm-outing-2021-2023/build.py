"""Build the private Kaggle dataset "MLB Statcast 2021-2023 With Outing Columns"
(yasunorim/mlb-statcast-2021-2023-outing-columns): the 23 columns of
yasunorim/mlb-statcast-2021-2023-pitch-subset plus inning_topbot, bat_score and post_bat_score, which the frozen
last-outing study (pitcher-after-il, commit 5bbe6ed) reads to find how a pitcher left his outing. It is the input of
a pre-registered confirmation of that study on MLB 2021-2023. Nothing about injured-list placements or transactions
is read here.

The subset's builder (../pitcher-confirm-statcast-2021-2023/build.py) is loaded unchanged and run with three more
kept columns: the same Savant request per day, the same parse, and every gate of that builder (season totals equal
Savant's, every Final game present, no duplicate key, schemas, tracking gaps). Only these module globals are set:
COLUMNS (the 23 + 3), FROZEN_TYPES and SCHEMA (the three new types are dataset4's), GAME_CHECKED (the three new
columns join the per-day and per-game "never empty" checks). The subset's own files are not rebuilt or changed.

Gates added here, on the written files (the build fails and nothing is kept if any is violated):
- inning_topbot is Top or Bot on every row; bat_score and post_bat_score are filled on every row, are >= 0 and
  post_bat_score >= bat_score;
- one inning_topbot per (game_pk, at_bat_number), and one per (game_pk, inning, pitcher) except where a pitcher
  appears in both halves (counted and printed: two-way players);
- within a (game_pk, inning, inning_topbot), bat_score never decreases in at_bat_number / pitch_number order, and
  post_bat_score of a pitch equals bat_score of the next pitch in the same half except at a half's last pitch (counted;
  the share of exceptions is printed and must be at most MAX_SCORE_BREAKS).
The notebook that reads these files checks that their 23 shared columns equal the pinned subset's, key by key.
Files are named outing_{2021,2022,2023}.parquet (so a notebook with both datasets attached finds each by name).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_SUB = Path(__file__).resolve().parent.parent / "pitcher-confirm-statcast-2021-2023" / "build.py"
_spec = importlib.util.spec_from_file_location("subset_build", _SUB)
sub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sub)
d4 = sub.d4
# the fetch/parse code and the subset builder that built the pinned subset (GHA 38483678623 at 492e865)
assert sub.md5(_SUB) == "f5f7de1ee64e5ccb76ae7c9232eeff58", "the subset builder changed"
assert sub.md5(sub._D4) == "593784729763616cdb2519cd9ff7ee5a", "dataset4's fetch/parse code changed"

NEW = ["inning_topbot", "bat_score", "post_bat_score"]
assert set(NEW) <= set(d4.SAVANT_COLUMNS) and not set(NEW) & set(sub.COLUMNS)
COLUMNS = list(sub.COLUMNS) + NEW
FROZEN_TYPES = dict(sub.FROZEN_TYPES) | {c: d4.SCHEMA.field(c).type for c in NEW}
assert FROZEN_TYPES["inning_topbot"] == pa.string() and FROZEN_TYPES["bat_score"] == pa.int64() \
    and FROZEN_TYPES["post_bat_score"] == pa.int64(), {c: FROZEN_TYPES[c] for c in NEW}
SCHEMA = pa.schema([pa.field(c, FROZEN_TYPES[c]) for c in COLUMNS])
sub.COLUMNS, sub.FROZEN_TYPES, sub.SCHEMA = COLUMNS, FROZEN_TYPES, SCHEMA
sub.GAME_CHECKED = [c for c in COLUMNS if c not in sub.NOT_GATED]
assert sub.TRACKING <= set(sub.GAME_CHECKED)
MAX_SCORE_BREAKS = 0.001  # set before the first build; the printed share is reported in the freeze file


def check_outing_columns(df: pd.DataFrame, year: int) -> list[str]:
    errs: list[str] = []
    tb = df["inning_topbot"]
    if not tb.isin(["Top", "Bot"]).all():
        errs.append(f"{year}: inning_topbot values {sorted(tb.dropna().unique().tolist())[:5]}, null {int(tb.isna().sum())}")
    for c in ("bat_score", "post_bat_score"):
        if df[c].isna().any() or (df[c] < 0).any():
            errs.append(f"{year}: {c} null {int(df[c].isna().sum())}, negative {int((df[c] < 0).sum())}")
    if (df["post_bat_score"] < df["bat_score"]).any():
        errs.append(f"{year}: post_bat_score < bat_score on {int((df['post_bat_score'] < df['bat_score']).sum())} rows")
    n_ab = df.groupby(["game_pk", "at_bat_number"])["inning_topbot"].nunique()
    if (n_ab > 1).any():
        errs.append(f"{year}: {int((n_ab > 1).sum())} at-bats with two inning_topbot values")
    n_pi = df.groupby(["game_pk", "inning", "pitcher"])["inning_topbot"].nunique()
    print(f"  {year}: (game, inning, pitcher) in both halves: {int((n_pi > 1).sum())}", flush=True)
    s = df.sort_values(["game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number"], kind="stable")
    half = ["game_pk", "inning", "inning_topbot"]
    same = (s[half].shift(-1) == s[half]).all(axis=1).to_numpy()
    b, pb = s["bat_score"].to_numpy(), s["post_bat_score"].to_numpy()
    dec = int((same[:-1] & (b[1:] < b[:-1])).sum())
    if dec:
        errs.append(f"{year}: bat_score decreases {dec} times within a half-inning")
    brk = same[:-1] & (pb[:-1] != b[1:])
    share = float(brk.sum()) / max(int(same[:-1].sum()), 1)
    print(f"  {year}: post_bat_score != next pitch's bat_score within a half on {int(brk.sum())} pitches "
          f"({share:.5f})", flush=True)
    if share > MAX_SCORE_BREAKS:
        errs.append(f"{year}: score breaks {share:.5f} > {MAX_SCORE_BREAKS}")
    return errs


def main() -> int:
    out = None
    for i, a in enumerate(sys.argv):
        if a == "--out":
            out = Path(sys.argv[i + 1])
    assert out is not None, "--out is required"
    print(f"subset builder {_SUB.parent.name}/build.py md5 {sub.md5(_SUB)}; kept columns {len(COLUMNS)}", flush=True)
    rc = sub.main()
    if rc != 0:
        return rc
    errs: list[str] = []
    for y in sub.YEARS:
        p = out / f"statcast_{y}.parquet"
        t = pq.read_table(p)
        if not t.schema.remove_metadata().equals(SCHEMA.remove_metadata()):
            errs.append(f"{p.name}: schema differs")
        errs += check_outing_columns(t.select(["game_pk", "at_bat_number", "pitch_number", "inning", "pitcher"] + NEW)
                                     .to_pandas(), y)
    if errs:
        print("\n".join(["", "Outing-column gates failed, nothing kept:"] + errs))
        for p in out.iterdir():
            p.unlink()
        out.rmdir()
        return 1
    print("md5 manifest (for the freeze file):")
    for y in sub.YEARS:
        p = out / f"statcast_{y}.parquet"
        q = out / f"outing_{y}.parquet"
        p.rename(q)
        print(f"  {sub.md5(q)}  {q.name}  {q.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
