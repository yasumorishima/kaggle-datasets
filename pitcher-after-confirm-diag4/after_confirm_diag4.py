# %% [markdown]
# # Pre-freeze diagnostic 4 for the 2021-2023 confirmation of the last-outing signal: sizing for v8 (counts only)
#
# Diagnostic 3 stopped the v7.2 freeze (item 5 added the exact gap) and showed the 4x merge left one stratum per role.
# This run prints counts only, so v8's strata and merge rule are chosen before any A statistic is computed under them:
# 2024-2026 (labels already read): arm positives, pool zeros, no-placement zeros, non-arm last outings and demotion last
# outings by role x gap class x calendar fifth, gap quantiles, and the merged strata each candidate rule would produce.
# 2021-2023: the build runs with real placements and every output is discarded; only rows with no placement of any kind
# within 14 days are counted (asserted). No A statistic, no AUC and no p is computed anywhere.

# %%
import contextlib
import gc
import hashlib
import io
import os

import numpy as np
import pandas as pd
import requests
from scipy.stats import rankdata, spearmanr

CONF_URL = "https://raw.githubusercontent.com/yasumorishima/kaggle-datasets/d1289f4ac28ffc81d23a657f23034147c68fc12e/pitcher-after-confirm/pitcher_after_confirm.py"
CONF_MD5 = "10199285ff0beda02d0cd712d94670a2"
src = requests.get(CONF_URL, timeout=60).text if CONF_URL.startswith("http") else open(CONF_URL).read()
assert hashlib.md5(src.encode()).hexdigest() == CONF_MD5, "not the confirmation notebook under test"
DEFS = src[: src.index("# %% [markdown]\n# ## 1. Regression gate")]  # definitions and the replacement machinery only
g = {"__name__": "conf_defs"}
exec(compile(DEFS, "conf_defs", "exec"), g)
N_DRAWS = int(os.environ.get("DIAG_DRAWS", "1000"))
N_FL = int(os.environ.get("DIAG_FLOOR", "200"))
N_POWER = int(os.environ.get("DIAG_POWER", "200"))
SMOKE = os.environ.get("SMOKE_FAKE_LABELS") == "1"
COVS = ["rest", "p7", "p14", "n_prev_il", "n_prev_arm", "days_since_act", "own_scale", "_gap"]
SHIFTS = [0.02, 0.05, 0.1]


def run(mode, fake, quiet=False):
    if fake:
        os.environ["SMOKE_FAKE_LABELS"] = "1"
    else:
        os.environ.pop("SMOKE_FAKE_LABELS", None)
    ns = {"__name__": f"diag_{mode}", "MODE": mode, "FIXED": FIXED, "SEEN": SEEN, "patch_base": g["patch_base"],
          "FROZEN": g["FROZEN"], "KEY": g["KEY"], "OUT_COLS": g["OUT_COLS"], "SUBSET_COLUMNS": g["SUBSET_COLUMNS"],
          "OUTING_MD5": g["OUTING_MD5"]}
    code = compile(g["make"](mode), f"diag_{mode}", "exec")
    if quiet:  # the 2021-2023 build with real placements: nothing it prints is read
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, ns)
    else:
        exec(code, ns)
    assert ns["FAKE"] == fake
    return ns


def fast_auc(y, r):  # r: average ranks of the scores
    n1 = y.sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def gkey(gap):
    return np.where(np.isnan(gap), -1, np.minimum(np.nan_to_num(gap), 120))



# %% [markdown]
# ## Sizing for v8 (counts only: no A statistic is computed under any new rule)

# %%
RULES = {
    "4x (frozen v7.1)": lambda n, k: k > 0 and n < 4 * k,
    "2x": lambda n, k: k > 0 and n < 2 * k,
    "zeros >= positives": lambda n, k: k > 0 and n - k < k,
    "zeros >= 5": lambda n, k: k > 0 and n - k < 5,
    "zeros >= 1": lambda n, k: k > 0 and n - k < 1,
}


def xtab(P, mask, name):
    t = pd.DataFrame({"role": P.role.to_numpy()[mask], "class": pd.Categorical(np.array(g["GAP_CLASSES"])[P.cls.to_numpy()[mask]], categories=g["GAP_CLASSES"]),
                      "fifth": P.fifth.to_numpy()[mask]})
    tab = pd.crosstab([t.role, t["class"]], t.fifth, margins=True)
    print(f"--- {name}: {int(mask.sum()):,} rows (role 0 reliever / 1 starter; columns calendar fifth)")
    print(tab.to_string())


def gap_q(P, mask, name):
    gp = np.where(np.isnan(P.gap.to_numpy()), 121.0, np.minimum(P.gap.to_numpy(), 120.0))
    for r in (0.0, 1.0):
        m = mask & (P.role.to_numpy() == r)
        q = np.percentile(gp[m], [10, 25, 50, 75, 90]) if m.any() else []
        print(f"gap quantiles 10/25/50/75/90 (none = 121) {name} role {r:.0f}: n {int(m.sum())}, {np.round(q, 1).tolist()}")


def merged_summary(P, sel, yy, label):
    role, cls, fif = P.role.to_numpy()[sel], P.cls.to_numpy()[sel], P.fifth.to_numpy()[sel]
    for nm, rule in RULES.items():
        key = g["merge_strata"](role, cls, fif, yy, rule)
        d = pd.DataFrame({"key": key, "role": role, "cls": cls, "fifth": fif, "y": yy})
        agg = d.groupby("key").agg(role=("role", "first"), classes=("cls", lambda s: tuple(sorted(set(s)))),
                                   fifths=("fifth", lambda s: tuple(sorted(set(s)))), rows=("y", "size"),
                                   positives=("y", "sum"))
        span = d.key.map(agg.classes.map(len))
        one = float((span[d.y] == 1).mean())
        print(f"{label} | rule {nm}: strata {len(agg)} (reliever {int((agg.role == 0).sum())}, starter "
              f"{int((agg.role == 1).sum())}); share of positives in a stratum of one gap class {one:.3f}")
        for c in range(5):
            m = d.y & (d.cls == c)
            if m.any():
                print(f"    positives of class {g['GAP_CLASSES'][c]}: {int(m.sum())}, their stratum spans classes "
                      f"{sorted(set(span[m]))} (median zeros in their stratum "
                      f"{float(np.median((d.key[m].map(agg.rows - agg.positives)).to_numpy())):.0f})")
        if nm != "4x (frozen v7.1)":
            print(agg.assign(classes=agg.classes.map(lambda t: [g["GAP_CLASSES"][i] for i in t])).to_string())


# %% [markdown]
# ## 1. 2024-2026 (labels already read in diagnostic 3)

# %%
FIXED, SEEN = {}, {}
ns = run("regression", SMOKE)
R = ns["rows_m"]
P = g["pool_frame"](R, ns["season_first"]["mlb"], ns["season_last"]["mlb"])
y_pool, _ = g["pool_labels"](R, P, ns["il_by"])
fin = np.isfinite(R.A_rank).to_numpy() & np.isfinite(R.A_rank_pos).to_numpy()
inpool = np.isfinite(y_pool) & fin
pos = inpool & (y_pool == 1.0)
zer = inpool & (y_pool == 0.0)
nopl = (R.use14 & R.early_hook.eq(0.0) & ~R.il14).to_numpy() & fin & P.cls.ge(0).to_numpy()
assert not (pos & nopl).any(), "a positive counted as a no-placement row"
il = R[R.il14 & ~R.next_arm & R.next_text].copy()
il["pl"] = il.day + pd.to_timedelta(il.days_to_il, unit="D")
_last = il.sort_values("day").groupby(["pitcher", "pl"]).tail(1)
nonarm = np.zeros(len(R), bool)
nonarm[R.index.get_indexer(_last[_last.days_to_il.ge(1)].index)] = True
nonarm &= zer
dem = (R.ok14 & R.other14 & ~R.il14 & R.early_hook.eq(0.0)).to_numpy() & fin & P.cls.ge(0).to_numpy()
xtab(P, pos, "2024-2026 arm positives in the pool")
xtab(P, zer, "2024-2026 pool zeros (all)")
xtab(P, nopl & zer, "2024-2026 pool zeros with no placement of any kind within 14 days")
xtab(P, nonarm, "2024-2026 pool zeros that are the last outing before a non-arm text placement")
xtab(P, dem, "2024-2026 last outings before a Triple-A outing within 14 days (outside the pool)")
for m_, nm_ in ((pos, "arm positives"), (zer, "pool zeros"), (nopl & zer, "no-placement zeros"), (nonarm, "non-arm last"),
                (dem, "demotion last")):
    gap_q(P, m_, nm_)
merged_summary(P, inpool, y_pool[inpool] == 1.0, "2024-2026 arm vs pool")
LEAGUE_2025 = {"lsc": dict(SEEN["lsc"]), "lg": SEEN["lg"].copy()}
del ns, R, P
gc.collect()

# %% [markdown]
# ## 2. 2021-2023: counts of rows with no placement within 14 days only (the build's outputs are discarded)

# %%
FIXED.update(LEAGUE_2025)
SEEN.clear()
ns = run("confirm", SMOKE, quiet=True)
assert ns["ROW_SEASONS"] == [2021, 2022, 2023]
R = ns["rows_m"]
P = g["pool_frame"](R, ns["season_first"]["mlb"], ns["season_last"]["mlb"])
fin = np.isfinite(R.A_rank).to_numpy() & np.isfinite(R.A_rank_pos).to_numpy()
nopl = (R.use14 & R.early_hook.eq(0.0) & ~R.il14).to_numpy() & fin & P.cls.ge(0).to_numpy()
assert not R.il14.to_numpy()[nopl].any(), "a row with a placement entered the count"
xtab(P, nopl, "2021-2023 pool rows with no placement of any kind within 14 days")
gap_q(P, nopl, "2021-2023 no-placement")
print("DIAGNOSTIC 4 (sizing) done")
