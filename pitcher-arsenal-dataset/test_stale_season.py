"""The gates must fail when one season's table is served as another season.

Savant answers a request it does not understand with the current season, so the failure this
build guards against is a season whose table is really another season's. Here one season of a
real fetch is relabelled as another (exactly, and with a few cells changed, as when Savant has
revised the older season a little) and the gates must reject both, while the real fetch passes.

Needs the raw tables of a real run: `python build.py ... --cache DIR` stores them in DIR/raw.pkl.
Usage: python test_stale_season.py DIR/raw.pkl
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build  # noqa: E402


def run(raw):
    long = build.build_long(raw)
    return build.gates(raw, long, build.build_wide(long, raw), build.build_changes(long), None)


def relabel(raw, table, src, dst, perturb=0.0):
    out = dict(raw)
    df = raw[table]
    col = "year"
    as_str = df[col].dtype == object
    moved = df[df[col].astype(int) == src].copy()
    moved[col] = str(dst) if as_str else dst
    if perturb:
        num = [c for c in ("pitch_usage", "whiff_percent", "avg_speed", "diff_z") if c in moved.columns]
        rng = np.random.default_rng(0)
        rows = rng.random(len(moved)) < perturb
        moved.loc[rows, num[0]] = moved.loc[rows, num[0]].astype(float) + 0.1
    out[table] = pd.concat([df[df[col].astype(int) != dst], moved], ignore_index=True)
    return out


def main() -> int:
    raw = pickle.loads(Path(sys.argv[1]).read_bytes())
    failures = []
    base = run(raw)
    if base:
        failures.append(f"the real fetch fails the gates: {base}")
    cases = [("stats", 2025, 2026, 0.0), ("stats", 2024, 2025, 0.05), ("arsenal", 2023, 2022, 0.0),
             ("movement", 2021, 2020, 0.05), ("arm_angle", 2025, 2026, 0.0)]
    for table, src, dst, perturb in cases:
        errs = run(relabel(raw, table, src, dst, perturb))
        hit = [e for e in errs if f"{min(src, dst)} and {max(src, dst)}" in e]
        print(f"{'ok  ' if hit else 'MISS'} {table} {src} served as {dst} (perturbed {perturb:.0%}): {hit[:2]}")
        if not hit:
            failures.append(f"{table}: {src} relabelled {dst} passed the gates")
    print("\n".join(failures) if failures else "all stale-season cases rejected; the real fetch passes")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
