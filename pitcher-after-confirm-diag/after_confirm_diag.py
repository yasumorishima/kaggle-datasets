# %% [markdown]
# # Pre-freeze diagnostic for the 2021-2023 confirmation of the last-outing signal (no 2021-2023 label is built)
#
# 1. **2024-2026 (labels already read):** the regression run of the confirmation notebook, then (a) the position
#    template of the real P1 positives (where the row sits in the pitcher-season, and the days to his next
#    appearance), and (b) a descriptive rival: the same statistic for the last outing before a non-arm placement.
# 2. **2021-2023 (random placements only):** the confirmation notebook's code with the frozen random-placement
#    generator, then 1,000 redraws of random positives placed like the real 2024-2026 positives (the joint bins of
#    the template), each tested with C2's rule (200 whole-group rotations, p over 201). The rejection rate at
#    p <= 0.025 and the share whose 95% lower bound (200 pitcher bootstrap draws) is above 0.5 are printed. The real
#    2021-2023 placements are never joined: the notebook runs with SMOKE_FAKE_LABELS=1 for that part.

# %%
import hashlib
import os

import numpy as np
import pandas as pd
import requests

CONF_URL = "https://raw.githubusercontent.com/yasumorishima/kaggle-datasets/b13220283bb0eada01dd766e88ed19a1f8347477/pitcher-after-confirm/pitcher_after_confirm.py"
CONF_MD5 = "bd0c206ba8873f8aa056fb21e727458d"
src = requests.get(CONF_URL, timeout=60).text if CONF_URL.startswith("http") else open(CONF_URL).read()
assert hashlib.md5(src.encode()).hexdigest() == CONF_MD5, "not the confirmation notebook under test"
# only the definitions and the replacement machinery (stop before section 1)
DEFS = src[: src.index("# %% [markdown]\n# ## 1. Regression gate")]
g = {"__name__": "conf_defs"}
exec(compile(DEFS, "conf_defs", "exec"), g)

# %% [markdown]
# ## 1. 2024-2026: template and the non-arm rival (real labels, already read)

# %%
os.environ.pop("SMOKE_FAKE_LABELS", None)
FIXED, SEEN = {}, {}
ns = {"__name__": "diag_regression", "MODE": "regression", "FIXED": FIXED, "SEEN": SEEN, "patch_base": g["patch_base"],
      "FROZEN": g["FROZEN"], "KEY": g["KEY"], "OUT_COLS": g["OUT_COLS"], "SUBSET_COLUMNS": g["SUBSET_COLUMNS"],
      "OUTING_MD5": g["OUTING_MD5"]}
exec(compile(g["make"]("regression"), "diag_regression", "exec"), ns)
R = ns["rows_m"].copy()
assert not ns["FAKE"]


def positions(r):
    """Share of the pitcher-season's rows before the row (0 = first) and days to his next row (inf = last)."""
    r = r.sort_values(["pitcher", "season", "day"])
    n = r.groupby(["pitcher", "season"]).day.transform("size")
    k = r.groupby(["pitcher", "season"]).cumcount()
    nxt = r.groupby(["pitcher", "season"]).day.shift(-1)
    gap = ((nxt - r.day).dt.days).fillna(np.inf)
    return pd.DataFrame({"q": k / n, "gap": gap}, index=r.index).reindex(r.index)


def bins(p):
    qb = np.minimum((p.q * 5).astype(int), 4)  # fifths of the pitcher-season
    gb = np.select([p.gap <= 7, p.gap <= 30, np.isfinite(p.gap)], [0, 1, 2], 3)  # next row within 7, 30 days, later, none
    return qb * 4 + gb


Rp = positions(R)
R["bin"] = bins(Rp).reindex(R.index)
no_hook = R.early_hook.eq(0.0)
P1rows = R[no_hook & R.y_last.notna() & np.isfinite(R.A_rank)]
tmpl = P1rows[P1rows.y_last.eq(1.0)].bin.value_counts(normalize=True).sort_index()
base_ = P1rows[P1rows.y_last.eq(0.0)].bin.value_counts(normalize=True).sort_index()
print("template of the real 2024-2026 P1 positives (bin = fifth of season x next-row gap class):")
print(pd.DataFrame({"positives": tmpl, "negatives": base_}).fillna(0).round(4).to_string())
dtl = P1rows[P1rows.y_last.eq(1.0)].days_to_il.value_counts(normalize=True).sort_index()

# the non-arm rival, descriptive: the last row before a non-arm placement 1-14 days later (same rules otherwise)
il = R[R.il14].copy()
il["pl"] = il.day + pd.to_timedelta(il.days_to_il, unit="D")
last = il.sort_values("day").groupby(["pitcher", "pl"]).tail(1)
keep = last[~last.next_arm & last.days_to_il.ge(1) & last.use14]
yr = np.where(R.use14 & ~R.il14, 0.0, np.nan)
yr[R.index.get_indexer(keep.index)] = 1.0
Rr = R.assign(y_r=yr)
D = Rr[no_hook & Rr.y_r.notna() & np.isfinite(Rr.A_rank)]
auc = ns["auc"]
print(f"descriptive rival, 2024-2026: last outing before a NON-arm placement 1-14 days later, no early hook: positives "
      f"{int(D.y_r.eq(1.0).sum())}, AUC {auc(D.y_r.to_numpy() == 1.0, D.A_rank.to_numpy()):.4f} (arm P1: 0.6059)")
LEAGUE_2025 = {"lsc": dict(SEEN["lsc"]), "lg": SEEN["lg"].copy()}
print("league constants from MLB 2025 (scale, median pitch count by role):", LEAGUE_2025["lsc"], LEAGUE_2025["lg"].to_dict()
      if hasattr(LEAGUE_2025["lg"], "to_dict") else LEAGUE_2025["lg"])
del ns, R, Rr, D, P1rows
import gc
gc.collect()

# %% [markdown]
# ## 2. 2021-2023: random positives placed like the real ones

# %%
os.environ["SMOKE_FAKE_LABELS"] = "1"
FIXED.update(LEAGUE_2025)
SEEN.clear()
ns = {"__name__": "diag_confirm", "MODE": "confirm", "FIXED": FIXED, "SEEN": SEEN, "patch_base": g["patch_base"],
      "FROZEN": g["FROZEN"], "KEY": g["KEY"], "OUT_COLS": g["OUT_COLS"], "SUBSET_COLUMNS": g["SUBSET_COLUMNS"],
      "OUTING_MD5": g["OUTING_MD5"]}
exec(compile(g["make"]("confirm"), "diag_confirm", "exec"), ns)
assert ns["FAKE"] and ns["ROW_SEASONS"] == [2021, 2022, 2023]
C = ns["rows_m"].copy()
Cp = positions(C)
C["bin"] = bins(Cp).reindex(C.index)
rotated_group = ns["rotated_group"]
use = C.use14.to_numpy()
ok_row = (C.early_hook.eq(0.0) & np.isfinite(C.A_rank)).to_numpy()
by_bin = {b: np.flatnonzero(use & ok_row & C.bin.eq(b).to_numpy()) for b in range(20)}
n_pos = int(round(314 * 1.17))
days = C.day.to_numpy()
pit = C.pitcher.to_numpy()
by_pit = {b: np.asarray(ix) for b, ix in pd.Series(np.arange(len(C))).groupby(pit).groups.items()}
assert C.index.equals(pd.RangeIndex(len(C))), "rows are expected in a fresh index"
rng = np.random.default_rng(31)
res = []
for r_ in range(1000):
    nb = rng.multinomial(n_pos, tmpl.reindex(range(20)).fillna(0).to_numpy())
    pos = np.concatenate([rng.choice(by_bin[b], min(k, len(by_bin[b])), replace=False) for b, k in enumerate(nb) if k])
    y = np.where(use, 0.0, np.nan)
    # rows of the same pitcher in the 14 days before each positive's (drawn) placement are left out, as the target does
    k_ = rng.choice(dtl.index.to_numpy(), len(pos), p=dtl.to_numpy())
    for i, kk in zip(pos, k_):
        rows_p = by_pit[pit[i]]
        dp = days[rows_p]
        y[rows_p[(dp > days[i]) & (dp <= days[i] + np.timedelta64(int(kk), "D"))]] = np.nan
        y[rows_p[(dp < days[i]) & (dp >= days[i] - np.timedelta64(14 - int(kk), "D"))]] = np.nan
    y[pos] = 1.0
    sel = ok_row & np.isfinite(y)
    df = C[sel].reset_index(drop=True)
    yy = y[sel] == 1.0
    s = df.A_rank.to_numpy()
    gix = list(df.groupby(["pitcher", "season"]).indices.values())
    a = auc(yy, s)
    fl = np.array([auc(yy, rotated_group(s, gix, sd)) for sd in range(7000, 7200)])
    p = (1 + int((fl >= a).sum())) / 201
    idx_by = list(df.groupby("pitcher").indices.values())
    rb = np.random.default_rng(2)
    bs = [auc(yy[ix], s[ix]) for ix in (np.concatenate([idx_by[j] for j in rb.integers(0, len(idx_by), len(idx_by))]) for _ in range(200))]
    lo = np.nanpercentile(bs, 2.5)
    res.append({"auc": a, "p": p, "lo": lo, "pos": int(yy.sum()), "fmean": fl.mean()})
    if (r_ + 1) % 100 == 0:
        d = pd.DataFrame(res)
        print(f"{r_ + 1}: positives mean {d.pos.mean():.0f}; AUC mean {d.auc.mean():.4f}; floor mean {d.fmean.mean():.4f}; "
              f"p<=0.025 {(d.p <= 0.025).mean():.3f}; lower>0.5 {(d.lo > 0.5).mean():.3f}; both {((d.p <= 0.025) & (d.lo > 0.5)).mean():.3f}",
              flush=True)
pd.DataFrame(res).to_csv("after_confirm_diag.csv", index=False)
