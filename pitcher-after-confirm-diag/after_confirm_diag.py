# %% [markdown]
# # Pre-freeze diagnostic 2 for the 2021-2023 confirmation of the last-outing signal (no 2021-2023 label is built)
#
# Fixed in PLAN_after_confirm.md v5 before it runs:
# 1. **2024-2026 (labels already read):** the regression run of the confirmation notebook; the real P1 positives'
#    positions (template for 2a); **(b)** the last outing before a NON-arm placement under the redesigned rule (both
#    floors, both lines) -- if it passes, the position-matched floor does not control "the last outing before a long
#    absence" and C2 is not frozen.
# 2. **2021-2023 (random placements only, SMOKE_FAKE_LABELS=1):** **(a)** 1,000 draws of pseudo-positives placed finer
#    than the floor's bins (each of 367 targets drawn from the real positives' (position, days to next row, season end)
#    takes a random one of the 5 eligible rows with the same days-to-next key nearest in position), left-out windows as
#    the target does; the share passing the redesigned rule (200 rotations, 200 position-matched draws, p over 201)
#    must be at most 0.05. **(c)** the same draws with +0.05 on the pseudo-positives' scores: AUC excess and share passing.

# %%
import gc
import hashlib
import os

import numpy as np
import pandas as pd
import requests
from scipy.stats import rankdata

CONF_URL = "https://raw.githubusercontent.com/yasumorishima/kaggle-datasets/824d9f6b695580a4054d0c98688a6ab65a55a132/pitcher-after-confirm/pitcher_after_confirm.py"
CONF_MD5 = "0b79718aeaf4638a4ca9b4da80489ac6"
src = requests.get(CONF_URL, timeout=60).text if CONF_URL.startswith("http") else open(CONF_URL).read()
assert hashlib.md5(src.encode()).hexdigest() == CONF_MD5, "not the confirmation notebook under test"
DEFS = src[: src.index("# %% [markdown]\n# ## 1. Regression gate")]  # definitions and the replacement machinery only
g = {"__name__": "conf_defs"}
exec(compile(DEFS, "conf_defs", "exec"), g)
N_DRAWS = int(os.environ.get("DIAG_DRAWS", "1000"))
N_FL = int(os.environ.get("DIAG_FLOOR", "200"))


def run(mode, fake):
    if fake:
        os.environ["SMOKE_FAKE_LABELS"] = "1"
    else:
        os.environ.pop("SMOKE_FAKE_LABELS", None)
    ns = {"__name__": f"diag_{mode}", "MODE": mode, "FIXED": FIXED, "SEEN": SEEN, "patch_base": g["patch_base"],
          "FROZEN": g["FROZEN"], "KEY": g["KEY"], "OUT_COLS": g["OUT_COLS"], "SUBSET_COLUMNS": g["SUBSET_COLUMNS"],
          "OUTING_MD5": g["OUTING_MD5"]}
    exec(compile(g["make"](mode), f"diag_{mode}", "exec"), ns)
    assert ns["FAKE"] == fake
    return ns


def passes(r):
    return r["p"] <= 0.025 and r["p_bin"] <= 0.025


# %% [markdown]
# ## 1. 2024-2026: regression, template, and (b) the non-arm last outing under the redesigned rule

# %%
FIXED, SEEN = {}, {}
ns = run("regression", False)
R = ns["rows_m"]
POS = g["position_frame"](R, ns["season_last"]["mlb"])
r1, r1p = g["p1_stat"](ns, "A_rank"), g["p1_stat"](ns, "A_rank_pos")
assert abs(r1["auc"] - g["FROZEN"]["auc"]) < 1e-12 and (r1["pos"], r1["rows"]) == (g["FROZEN"]["positives"], g["FROZEN"]["rows_p1"]), \
    "the regression run does not reproduce P1"
print(f"arm P1 2024-2026 (frozen AUC reproduced {r1['auc']:.16f}) under the redesigned rule: rotation p {r1['p']:.4f} / "
      f"{r1p['p']:.4f}; position-matched floor mean {r1['floor_bin'].mean():.4f} / {r1p['floor_bin'].mean():.4f}, 97.5th "
      f"{np.percentile(r1['floor_bin'], 97.5):.4f} / {np.percentile(r1p['floor_bin'], 97.5):.4f}, p {r1['p_bin']:.4f} / "
      f"{r1p['p_bin']:.4f} -> {'passes' if passes(r1) and passes(r1p) else 'does not pass'}")
print(r1["bins"].T.to_string())
tm = POS.loc[r1["index"][r1["df"].y_last.to_numpy() == 1.0]]
template = pd.DataFrame({"q": tm.q.to_numpy(), "gap": tm.gap.to_numpy(), "late": tm.late.to_numpy()})
dtl = r1["df"].days_to_il[r1["df"].y_last.eq(1.0)].value_counts(normalize=True).sort_index()
print(f"template: {len(template)} positives; days to next row: none-late {(template.gap.isna() & template.late).mean():.3f}, "
      f"none-early {(template.gap.isna() & ~template.late).mean():.3f}, median gap {template.gap.median():.0f}")

# (b) the non-arm rival under the same code path: y_last replaced by the non-arm target, then restored
il = R[R.il14].copy()
il["pl"] = il.day + pd.to_timedelta(il.days_to_il, unit="D")
last = il.sort_values("day").groupby(["pitcher", "pl"]).tail(1)
keep = last[~last.next_arm & last.days_to_il.ge(1) & last.use14]
yr = np.where(R.use14 & ~R.il14, 0.0, np.nan)
yr[R.index.get_indexer(keep.index)] = 1.0
y_save = R["y_last"].copy()
R["y_last"] = yr
nb, nbp = g["p1_stat"](ns, "A_rank"), g["p1_stat"](ns, "A_rank_pos")
R["y_last"] = y_save
NONARM_PASS = passes(nb) and passes(nbp)
print(f"(b) NON-arm last outing 2024-2026: positives {nb['pos']}, AUC {nb['auc']:.4f} / {nbp['auc']:.4f}; rotation p "
      f"{nb['p']:.4f} / {nbp['p']:.4f}; position-matched floor mean {nb['floor_bin'].mean():.4f} / {nbp['floor_bin'].mean():.4f}, "
      f"p {nb['p_bin']:.4f} / {nbp['p_bin']:.4f} -> {'PASSES (C2 is not frozen)' if NONARM_PASS else 'does not pass'}")
LEAGUE_2025 = {"lsc": dict(SEEN["lsc"]), "lg": SEEN["lg"].copy()}
print("league constants from MLB 2025 (scale, median pitch count by role):", LEAGUE_2025["lsc"], LEAGUE_2025["lg"].to_dict())
del ns, R, POS, r1, r1p, nb, nbp, il, last, keep
gc.collect()

# %% [markdown]
# ## 2. 2021-2023 (random placements only): (a) the redesigned rule under the null, (c) power

# %%
FIXED.update(LEAGUE_2025)
SEEN.clear()
ns = run("confirm", True)
assert ns["ROW_SEASONS"] == [2021, 2022, 2023]
C = ns["rows_m"].reset_index(drop=True)
P = g["position_frame"](C, ns["season_last"]["mlb"])
rotated_group, auc = ns["rotated_group"], ns["auc"]
use = C.use14.to_numpy()
ok_row = (C.early_hook.eq(0.0) & np.isfinite(C.A_rank) & np.isfinite(C.A_rank_pos)).to_numpy()
elig = np.flatnonzero(use & ok_row)


def gkey(gap, late):
    return np.where(np.isnan(gap), np.where(late, -2, -1), np.minimum(np.nan_to_num(gap), 120))


ek = gkey(P.gap.to_numpy()[elig], P.late.to_numpy()[elig])
eq = P.q.to_numpy()[elig]
groups = {}
for k in np.unique(ek):
    ix = elig[ek == k]
    o = np.argsort(eq[ek == k], kind="stable")
    groups[k] = (ix[o], eq[ek == k][o])
gkeys = np.array(sorted(groups))
tk = gkey(template.gap.to_numpy(), template.late.to_numpy())
tq = template.q.to_numpy()
days = C.day.to_numpy()
pit = C.pitcher.to_numpy()
by_pit = {b: np.asarray(ix) for b, ix in pd.Series(np.arange(len(C))).groupby(pit).groups.items()}


def draw_pseudo(rng, n=367):
    j = rng.integers(0, len(tk), n)
    out = []
    for kk, qq in zip(tk[j], tq[j]):
        if kk not in groups:  # nearest key of the same kind (a none key only matches none keys)
            cand = gkeys[(gkeys >= 0)] if kk >= 0 else gkeys[gkeys < 0]
            kk = cand[np.argmin(np.abs(cand - kk))]
        ix, qs = groups[kk]
        c = np.searchsorted(qs, qq)
        lo, hi = max(0, c - 5), min(len(ix), c + 5)
        near = np.arange(lo, hi)[np.argsort(np.abs(qs[lo:hi] - qq), kind="stable")[:5]]
        out.append(ix[rng.choice(near)])
    return np.unique(np.array(out))


def fast_auc(y, r):  # r: average ranks of the scores
    n1 = y.sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


rng = np.random.default_rng(31)
res = []
for r_ in range(N_DRAWS):
    pos = draw_pseudo(rng)
    k_ = rng.choice(dtl.index.to_numpy(), len(pos), p=dtl.to_numpy())
    out_ = np.zeros(len(C), dtype=bool)
    for i, kk in zip(pos, k_):
        rows_p = by_pit[pit[i]]
        dp = days[rows_p]
        out_[rows_p[(dp > days[i]) & (dp <= days[i] + np.timedelta64(int(kk), "D"))]] = True
        out_[rows_p[(dp < days[i]) & (dp >= days[i] - np.timedelta64(14 - int(kk), "D"))]] = True
    y = np.where(use, 0.0, np.nan)
    y[out_] = np.nan
    y[pos[~out_[pos]]] = 1.0
    sel = ok_row & np.isfinite(y)
    df = C[sel].reset_index(drop=True)
    yy = y[sel] == 1.0
    key = g["merge_bins"](P.key.to_numpy()[sel], yy)
    gix = list(df.groupby(["pitcher", "season"]).indices.values())
    masks = list(g["_draw_masks"](yy, key, range(7000, 7000 + N_FL)))
    row = {"pos": int(yy.sum()), "bins": len(np.unique(key))}
    for nm, sc in (("", "A_rank"), ("_p", "A_rank_pos")):
        for pl, add in (("", 0.0), ("_pl", 0.05)):
            s = df[sc].to_numpy() + add * yy
            rk = rankdata(s)
            a = fast_auc(yy, rk)
            if r_ == 0:
                assert abs(a - auc(yy, s)) < 1e-12, "fast AUC differs from the notebook's"
            fr = np.array([fast_auc(yy, rotated_group(rk, gix, sd)) for sd in range(7000, 7000 + N_FL)])
            fb = np.array([fast_auc(m, rk) for m in masks])
            row.update({f"auc{nm}{pl}": a, f"p{nm}{pl}": (1 + int((fr >= a).sum())) / (N_FL + 1),
                        f"pb{nm}{pl}": (1 + int((fb >= a).sum())) / (N_FL + 1), f"fb{nm}{pl}": fb.mean()})
    res.append(row)
    if (r_ + 1) % 100 == 0 or r_ + 1 == N_DRAWS:
        d = pd.DataFrame(res)
        cut = 0.025
        null_pass = (d.p <= cut) & (d.pb <= cut) & (d.p_p <= cut) & (d.pb_p <= cut)
        old_pass = (d.p <= cut) & (d.p_p <= cut)
        pw = (d.p_pl <= cut) & (d.pb_pl <= cut) & (d.p_p_pl <= cut) & (d.pb_p_pl <= cut)
        print(f"{r_ + 1}: positives mean {d.pos.mean():.0f}, merged bins {d.bins.mean():.1f}; (a) AUC {d.auc.mean():.4f} / "
              f"{d.auc_p.mean():.4f}, position-matched floor mean {d.fb.mean():.4f} / {d.fb_p.mean():.4f}; passes the "
              f"redesigned rule {null_pass.mean():.3f} (rotation floors only {old_pass.mean():.3f}); (c) +0.05: AUC excess "
              f"{(d.auc_pl - d.auc).mean():+.4f} / {(d.auc_p_pl - d.auc_p).mean():+.4f}, passes {pw.mean():.3f}", flush=True)
pd.DataFrame(res).to_csv("after_confirm_diag2.csv", index=False)
d = pd.DataFrame(res)
A_SHARE = float(((d.p <= 0.025) & (d.pb <= 0.025) & (d.p_p <= 0.025) & (d.pb_p <= 0.025)).mean())
print(f"DIAGNOSTIC 2: (a) null share passing {A_SHARE:.3f} (must be <= 0.05); (b) non-arm passes: {NONARM_PASS}; "
      f"-> {'C2 may be frozen with the redesigned rule' if A_SHARE <= 0.05 and not NONARM_PASS else 'C2 is NOT frozen: redesign again'}")
