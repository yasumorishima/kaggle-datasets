# %% [markdown]
# # Pre-freeze diagnostic 5 for the 2021-2023 confirmation of the last-outing signal (PLAN v8.2)
#
# 2024-2026 (labels already read): the v8.2 deciding contrast (arm positives vs clean long breaks, role x none strata,
# floor P) beside the frozen rotation, the reported items of v8.1 item 6, and power at the 2021-2023 sizes.
# 2021-2023: the confirmation build runs with real placements and everything it prints is discarded (the frozen code
# inside it computes its own quantities, e.g. B0 for 2023, which are not read); this notebook prints only counts for
# 2021-2023 (clean long breaks by season x role x none; arm positives by season x gap class before any A-derived filter)
# and computes no statistic of A on 2021-2023.

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

CONF_URL = "https://raw.githubusercontent.com/yasumorishima/kaggle-datasets/e82f5e702455359e60ac2ec81e092dda09e3b33b/pitcher-after-confirm/pitcher_after_confirm.py"
CONF_MD5 = "0cd200dbe4763671b124a1c7fee95006"
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
# ## 1. 2024-2026 (labels already read): the v8.2 contrast, the reported items, power

# %%
FIXED, SEEN = {}, {}
ns = run("regression", SMOKE)
R = ns["rows_m"]
r1, r1p = g["p1_stat"](ns, "A_rank"), g["p1_stat"](ns, "A_rank_pos")
if not SMOKE:
    assert abs(r1["auc"] - g["FROZEN"]["auc"]) < 1e-12, "the regression run does not reproduce P1"
Q, QP = g["clean_test"](ns, "A_rank"), g["clean_test"](ns, "A_rank_pos")
print("2024-2026 strata of the clean-break contrast (role * 10 + none):\n" + Q["table"].T.to_string())
print("balance (False = clean long breaks, True = arm positives):\n" + Q["balance"].round(3).to_string())
for nm, q_, r_ in (("A_rank", Q, r1), ("A_rank_pos", QP, r1p)):
    print(f"2024-2026 [{nm}] arm positives vs clean long breaks: stratified AUC {q_['auc']:.4f}, floor mean "
          f"{q_['floor'].mean():.4f}, 97.5th {np.percentile(q_['floor'], 97.5):.4f}, p {q_['p']:.4f} ({q_['pos']} vs "
          f"{q_['rows'] - q_['pos']}); fifth-reweighted {q_['reweighted']:.4f}; rotation p {r_['p']:.4f} -> "
          f"{'both floors pass' if q_['p'] <= 0.025 and r_['p'] <= 0.025 else 'not both'}")
E = Q["auc"] - Q["floor"].mean()
print(f"E (2024-2026 excess of the contrast over floor P, A_rank) {E:+.4f}")
g["reported_v82"](ns, "2024-2026 (reported)")

# power material: the 2024-2026 contrast rows
P = g["pool_frame"](R, ns["season_first"]["mlb"], ns["season_last"]["mlb"])
sel = np.flatnonzero(np.isin(R.index, Q["index"]))
y_all = Q["y"]
none_all = P.gap.isna().to_numpy()[sel]
role_all = P.role.to_numpy()[sel]
sA = R.A_rank.to_numpy()[sel]
LEAGUE_2025 = {"lsc": dict(SEEN["lsc"]), "lg": SEEN["lg"].copy()}
rot_material = r1["df"][["pitcher", "season", "y_last", "A_rank"]].copy()
rotated_group, auc_f = ns["rotated_group"], ns["auc"]
del ns
gc.collect()

# %% [markdown]
# ## 2. 2021-2023: counts only (no A statistic; the build's printed output is discarded)

# %%
FIXED.update(LEAGUE_2025)
SEEN.clear()
ns = run("confirm", SMOKE, quiet=True)
assert ns["ROW_SEASONS"] == [2021, 2022, 2023]
R2 = ns["rows_m"]
P2 = g["pool_frame"](R2, ns["season_first"]["mlb"], ns["season_last"]["mlb"])
c2 = g["clean_parts"](R2, P2, ns["il_by"], ns["season_last"]["mlb"])
clean2 = c2["base"] & c2["quiet"] & c2["long"] & ~c2["between"]
t = pd.DataFrame({"season": R2.season.to_numpy(), "role": P2.role.to_numpy(), "none": c2["none"]})
print("2021-2023 clean long breaks by season x role x none (counts):\n"
      + t[clean2].groupby(["season", "role", "none"]).size().unstack(["role", "none"]).fillna(0).astype(int).to_string())
print(f"2021-2023 long breaks removed by the cleaning rule: {int((c2['base'] & c2['quiet'] & c2['long'] & c2['between']).sum())}")
posall = R2.y_last.eq(1.0).to_numpy()  # before the finite-A and early-hook filters
cls_name = np.array(["<=7"] + g["GAP_CLASSES"])[P2.cls.to_numpy() + 1]
print("2021-2023 arm positives (y_last = 1, before the finite-A and early-hook filters) by season x gap class (counts):\n"
      + pd.crosstab(R2.season.to_numpy()[posall], cls_name[posall]).to_string())
g21 = posall & (R2.season.to_numpy() == 2021) & (np.nan_to_num(P2.gap.to_numpy(), nan=999) < 11)
print(f"2021 positives with a gap below 11 days (kept): {int(g21.sum())}")
N0 = {(r, n): int((clean2 & (P2.role.to_numpy() == r) & (c2["none"] == n)).sum()) for r in (0.0, 1.0) for n in (False, True)}
N1 = int((posall & R2.use14.to_numpy() & R2.early_hook.eq(0.0).to_numpy() & (P2.cls.to_numpy() >= 0)).sum())
print(f"power sizes: zeros by (role, none) {N0}; positives with use14, no early hook and a gap of 8+ days or none {N1} "
      f"(no A-derived filter)")
del ns, R2, P2
gc.collect()

# %% [markdown]
# ## 3. Power at the 2021-2023 sizes, from 2024-2026 (A_rank line; N1 counts positives before the finite-A filter, so it
# ## is an upper bound and the power below is optimistic)

# %%
N_PW = int(os.environ.get("DIAG_POWER", "200"))
rng = np.random.default_rng(55)
cells = {(r, n): np.flatnonzero(~y_all & (role_all == r) & (none_all == n)) for r in (0.0, 1.0) for n in (False, True)}
for cell, k in N0.items():
    if k and not len(cells[cell]):
        print(f"power: 2021-2023 has {k} zeros in (role, none) {cell} but 2024-2026 has none there: left out of the draws")


def contrast_draw(pos_ix, cell_ix, shift=0.0, null=False):
    """One contrast at the 2021-2023 sizes from the given 2024-2026 positives and zeros (indices into the contrast rows)."""
    take = [rng.choice(pos_ix, N1, replace=N1 > len(pos_ix))]
    for cell, k in N0.items():
        z = cell_ix[cell]
        if k and len(z):
            take.append(rng.choice(z, k, replace=k > len(z)))
    ix = np.concatenate(take)
    yy = np.zeros(len(ix), bool)
    yy[: len(take[0])] = True
    key = g["merge_kind"](role_all[ix], none_all[ix], yy)
    if null:  # labels permuted within strata
        yy = next(g["_draw_masks"](yy, key, [int(rng.integers(1 << 30))]))
    st = g["Strat"](key, sA[ix] + shift * yy)
    a = st.auc(yy)
    fl = np.array([st.auc(m) for m in g["_draw_masks"](yy, key, range(7000, 7000 + N_FL))])
    return (1 + int((fl >= a).sum())) / (N_FL + 1) <= 0.025


def rotation_draw(df, shift=0.0):
    yr = df.y_last.to_numpy() == 1.0
    sr = df.A_rank.to_numpy() + shift * yr
    gix = list(df.groupby(["pitcher", "season"]).indices.values())
    ar = auc_f(yr, sr)
    fr = np.array([auc_f(yr, rotated_group(sr, gix, sd)) for sd in range(7000, 7000 + N_FL)])
    return (1 + int((fr >= ar).sum())) / (N_FL + 1) <= 0.025


# (i) planted, the conjunction: the contrast under its null with +sh on the drawn positives, AND the frozen rotation on
# the 2024-2026 P1 rows with the arm labels permuted across rows within each pitcher-season (its null) and +sh planted
rot_null = rot_material.copy()
rot_gid = rot_null.groupby(["pitcher", "season"]).ngroup().to_numpy()
rot_sorted = np.argsort(rot_gid, kind="stable")
res = {}
for sh in (0.0, E / 2, E):
    ok = []
    for _ in range(N_PW):
        c_ok = contrast_draw(np.flatnonzero(y_all), cells, sh, null=True)
        rn = rot_null.copy()
        o = np.lexsort((rng.random(len(rn)), rot_gid))  # a random order within each pitcher-season
        yv = rn.y_last.to_numpy().copy()
        yv[rot_sorted] = yv[o]
        rn["y_last"] = yv
        assert np.array_equal(np.bincount(rot_gid, weights=(yv == 1.0)), np.bincount(rot_gid, weights=(rot_null.y_last.to_numpy() == 1.0))),             "the rotation null moved a positive across pitcher-seasons"
        ok.append((c_ok, rotation_draw(rn, sh)))
    ok = np.array(ok)
    res[sh] = ok
    print(f"power (planted {sh:+.4f}, A_rank line, both floors under their nulls): contrast {ok[:, 0].mean():.3f}, rotation "
          f"{ok[:, 1].mean():.3f}, both {np.mean(ok[:, 0] & ok[:, 1]):.3f} ({len(ok)} draws)")

# (ii) unplanted: pitcher bootstrap over the union of P1 and contrast pitchers; the contrast is subsampled to the
# 2021-2023 sizes inside each draw and the rotation uses the drawn copies' P1 rows
pit = R.pitcher.to_numpy()[sel]
U = np.union1d(np.unique(pit), rot_material.pitcher.unique())
rot_by = {b: d for b, d in rot_material.groupby("pitcher")}
both = []
for _ in range(min(N_PW, 100)):
    cnt = pd.Series(rng.choice(U, len(U))).value_counts()
    mult = pd.Series(pit).map(cnt).fillna(0).astype(int).to_numpy()
    rep_ix = np.repeat(np.arange(len(pit)), mult)
    pos_b = rep_ix[y_all[rep_ix]]
    cells_b = {c: rep_ix[np.isin(rep_ix, ix_)] for c, ix_ in cells.items()}
    c_ok = contrast_draw(pos_b, cells_b) if len(pos_b) else False
    parts = [rot_by[b].assign(pitcher=f"{b}_{c_}") for b, k in cnt.items() if b in rot_by for c_ in range(k)]
    both.append((c_ok, rotation_draw(pd.concat(parts, ignore_index=True))))
both = np.array(both)
print(f"unplanted pitcher bootstrap of 2024-2026 at the 2021-2023 contrast sizes ({len(both)} draws, A_rank line): contrast "
      f"passes {both[:, 0].mean():.3f}, rotation passes {both[:, 1].mean():.3f}, both {np.mean(both[:, 0] & both[:, 1]):.3f}")
print("DIAGNOSTIC 5 done")
