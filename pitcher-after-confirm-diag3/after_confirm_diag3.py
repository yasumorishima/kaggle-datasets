# %% [markdown]
# # Pre-freeze diagnostic 3 for the 2021-2023 confirmation of the last-outing signal (no 2021-2023 arm label meets A)
#
# Fixed in PLAN_after_confirm.md v7.1 before it runs:
# 1. **2024-2026 (labels already read):** the regression run; the arm P1 under the pool floor; **item 5** each
#    covariate (rest, p7, p14, n_prev_il, n_prev_arm, days_since_act, own_scale, exact gap) as a score under the pool
#    floor (two-sided) with its within-stratum Spearman with A_rank among 0 rows (p <= 0.025 and |Spearman| >= 0.05 ->
#    its tercile joins the strata before freezing); non-arm placements and demotions as pseudo-positives and the
#    arm-vs-non-arm secondary (reported); **item 4** on 2024-2026: pseudo-positives on rows with no placement of any kind
#    within 14 days, placed by the real positives' template (role, exact gap key, nearest calendar position), tested by
#    the full rule (200 stratified draws, 200 rotations, p over 201); the share passing must be at most 0.05.
# 2. **2021-2023:** the confirmation's rows are built with the real placements and **every output of that build is
#    discarded**; only rows with no placement of any kind within 14 days are used (asserted), for item 4 and the power of
#    item 8 (planted shifts on the pseudo-positives' scores). No pool label of 2021-2023 is built.

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
# ## The null check (PLAN v7.1 item 4): pseudo-positives on rows with no placement within 14 days

# %%


class NullCheck:
    def __init__(self, ns, template):
        R = ns["rows_m"].reset_index(drop=True)
        self.R = R
        self.P = g["pool_frame"](R, ns["season_first"]["mlb"], ns["season_last"]["mlb"])
        fin = np.isfinite(R.A_rank) & np.isfinite(R.A_rank_pos)
        self.base = (R.use14 & R.early_hook.eq(0.0) & fin & ~R.il14).to_numpy()
        self.pool = self.base & self.P.cls.ge(0).to_numpy()
        assert not R.il14.to_numpy()[self.base].any(), "a row with a placement entered the null check"
        el = np.flatnonzero(self.pool)
        ek = gkey(self.P.gap.to_numpy()[el])
        er = self.P.role.to_numpy()[el]
        ef = self.P.frac.to_numpy()[el]
        self.groups = {}
        for key in set(zip(er, ek)):
            m = (er == key[0]) & (ek == key[1])
            o = np.argsort(ef[m], kind="stable")
            self.groups[key] = (el[m][o], ef[m][o])
        self.template = template
        self.fallback = self.short = self.skipped = 0
        self.days = R.day.to_numpy()
        pit = R.pitcher.to_numpy()
        self.by_pit = {b: np.asarray(ix) for b, ix in pd.Series(np.arange(len(R))).groupby(pit).groups.items()}
        self.pit = pit
        print(f"null check: {int(self.base.sum()):,} rows with no placement within 14 days, {len(el):,} of them in the pool; "
              f"template {len(template)} positives")

    def draw(self, rng):
        used = set()
        pos, ks = [], []
        for t in rng.permutation(len(self.template)):
            role, kk, fr, k = self.template.iloc[t][["role", "gkey", "frac", "days_to_il"]]
            key = (role, kk)
            if key not in self.groups:  # nearest key of the same role and kind (a none key only matches none keys)
                cand = [x for x in self.groups if x[0] == role and (x[1] >= 0) == (kk >= 0)] or \
                       [x for x in self.groups if x[0] == role]
                if not cand:
                    self.skipped += 1
                    continue
                key = min(cand, key=lambda x: (abs(x[1] - kk), x[1]))
                self.fallback += 1
            ix, fs = self.groups[key]
            hi = int(np.searchsorted(fs, fr))
            lo = hi - 1
            near = []
            while len(near) < 5 and (lo >= 0 or hi < len(ix)):  # the 5 unused rows nearest in calendar position
                if hi < len(ix) and (lo < 0 or abs(fs[hi] - fr) <= abs(fs[lo] - fr)):
                    j, hi = hi, hi + 1
                else:
                    j, lo = lo, lo - 1
                if ix[j] not in used:
                    near.append(j)
            if len(near) < 5:
                self.short += 1
            if not near:
                self.skipped += 1
                continue
            i = ix[rng.choice(near)]
            used.add(i)
            pos.append(i)
            ks.append(int(k))
        y = np.where(self.base, 0.0, np.nan)
        for i, k in zip(pos, ks):  # the target's left-out window, as for a placement k days after the row
            rp = self.by_pit[self.pit[i]]
            dp = self.days[rp]
            y[rp[(dp > self.days[i]) & (dp <= self.days[i] + np.timedelta64(k, "D"))]] = np.nan
            y[rp[(dp < self.days[i]) & (dp >= self.days[i] - np.timedelta64(14 - k, "D"))]] = np.nan
        y[np.array(pos)] = 1.0
        return y

    def test(self, y, add=0.0, rot_seeds=None):
        """The full rule on one draw: both lines, the pool floor first, the rotation only if the pool passes."""
        R, P = self.R, self.P
        out = {}
        ypool = np.where(self.pool, y, np.nan)
        sel = np.isfinite(ypool)
        yy = ypool[sel] == 1.0
        key = g["merge_strata"](P.role.to_numpy()[sel], P.cls.to_numpy()[sel], P.fifth.to_numpy()[sel], yy)
        masks = list(g["_draw_masks"](yy, key, range(7000, 7000 + N_FL)))
        selr = np.isfinite(y)
        yr = y[selr] == 1.0
        df = R[selr]
        gix = list(df.groupby(["pitcher", "season"]).indices.values())
        ok = True
        for nm in ("A_rank", "A_rank_pos"):
            s = R[nm].to_numpy()[sel] + add * yy
            st = g["Strat"](key, s)
            a = st.auc(yy)
            fb = np.array([st.auc(m) for m in masks])
            pb = (1 + int((fb >= a).sum())) / (N_FL + 1)
            out[f"pool_auc_{nm}"], out[f"pool_floor_{nm}"], out[f"pool_p_{nm}"] = a, fb.mean(), pb
            out[f"rot_p_{nm}"] = np.nan
            if pb <= 0.025 and ok:
                rk = rankdata(df[nm].to_numpy() + add * yr)
                ar = fast_auc(yr, rk)
                fr_ = np.array([fast_auc(yr, self.rot(rk, gix, sd)) for sd in range(7000, 7000 + N_FL)])
                out[f"rot_p_{nm}"] = (1 + int((fr_ >= ar).sum())) / (N_FL + 1)
                ok = out[f"rot_p_{nm}"] <= 0.025
            else:
                ok = False
        out["passes"] = bool(ok)
        out["pos"] = int(yy.sum())
        out["strata"] = len(np.unique(key))
        return out


def null_run(ns, template, label, seed):
    nc = NullCheck(ns, template)
    nc.rot = ns["rotated_group"]
    rng = np.random.default_rng(seed)
    res = []
    for r_ in range(N_DRAWS):
        y = nc.draw(rng)
        row = nc.test(y)
        if r_ < N_POWER:
            for d in SHIFTS:
                o = nc.test(y, add=d)
                row[f"pass_{d}"] = o["passes"]
                row[f"excess_{d}"] = o["pool_auc_A_rank"] - row["pool_auc_A_rank"]
        res.append(row)
        if (r_ + 1) % 100 == 0 or r_ + 1 == N_DRAWS:
            d_ = pd.DataFrame(res)
            print(f"{label} {r_ + 1}: pseudo-positives mean {d_.pos.mean():.0f}, strata {d_.strata.mean():.1f}; pool AUC "
                  f"{d_.pool_auc_A_rank.mean():.4f} / {d_.pool_auc_A_rank_pos.mean():.4f} against floor mean "
                  f"{d_.pool_floor_A_rank.mean():.4f} / {d_.pool_floor_A_rank_pos.mean():.4f}; pool p <= 0.025 "
                  f"{(d_.pool_p_A_rank <= 0.025).mean():.3f} / {(d_.pool_p_A_rank_pos <= 0.025).mean():.3f}; passes the full rule "
                  f"{d_.passes.mean():.3f}", flush=True)
    d_ = pd.DataFrame(res)
    d_.to_csv(f"diag3_null_{label}.csv", index=False)
    print(f"{label}: template placements over {N_DRAWS} draws: fallback key {nc.fallback}, fewer than 5 unused candidates "
          f"{nc.short}, skipped {nc.skipped} (of {N_DRAWS * len(template)})")
    for d in SHIFTS:
        print(f"{label} power: +{d} on the pseudo-positives' scores: pool AUC excess {d_[f'excess_{d}'].mean():+.4f}, passes "
              f"{d_[f'pass_{d}'].mean():.3f} (first {N_POWER} draws)")
    return float(d_.passes.mean())


# %% [markdown]
# ## 1. 2024-2026 (labels already read)

# %%
FIXED, SEEN = {}, {}
ns = run("regression", SMOKE)
R = ns["rows_m"]
r1 = g["p1_stat"](ns, "A_rank")
if not SMOKE:
    assert abs(r1["auc"] - g["FROZEN"]["auc"]) < 1e-12 and (r1["pos"], r1["rows"]) == (g["FROZEN"]["positives"], g["FROZEN"]["rows_p1"]), \
        "the regression run does not reproduce P1"
q1, q1p = g["pool_test"](ns, "A_rank"), g["pool_test"](ns, "A_rank_pos")
print(f"arm P1 2024-2026 under the pool floor: stratified AUC {q1['auc']:.4f} / {q1p['auc']:.4f}, floor mean "
      f"{q1['floor'].mean():.4f} / {q1p['floor'].mean():.4f}, 97.5th {np.percentile(q1['floor'], 97.5):.4f} / "
      f"{np.percentile(q1p['floor'], 97.5):.4f}, p {q1['p']:.4f} / {q1p['p']:.4f}; rotation p {r1['p']:.4f}; positives "
      f"{q1['pos']} in {q1['rows']} pool rows; positives within 7 days of the next row, left out: {q1['near_left_out']}")
print(q1["table"].T.to_string())
EXCESS = q1["auc"] - q1["floor"].mean()
print(f"item 8: the arm excess over the pool floor (A_rank) {EXCESS:+.4f}")

# item 5: covariates and the exact gap as scores under the pool floor (two-sided), and their within-stratum Spearman with A
P = g["pool_frame"](R, ns["season_first"]["mlb"], ns["season_last"]["mlb"])
R["_gap"] = np.where(np.isnan(P.gap), 121.0, np.minimum(P.gap, 120.0))
y_pool, _ = g["pool_labels"](R, P, ns["il_by"])
sel0 = np.isin(R.index, q1["index"])
key0 = pd.Series(q1["key"], index=q1["index"])
ADD = []
for c in COVS:
    if c not in R.columns:
        print(f"item 5: {c}: not a row column, skipped")
        continue
    t = g["pool_test"](ns, c, two_sided=True)
    z = R.loc[q1["index"]].assign(_k=key0.to_numpy(), _y=q1["y"])
    z = z[~z._y & np.isfinite(z[c]) & np.isfinite(z.A_rank)]
    sp = [(len(gz), spearmanr(gz[c], gz.A_rank).correlation) for _, gz in z.groupby("_k") if len(gz) >= 10 and gz[c].nunique() > 1]
    w = np.array([n for n, _ in sp], float)
    v = np.array([s_ for _, s_ in sp], float)
    msp = float(np.nansum(w * v) / w[np.isfinite(v)].sum()) if len(sp) else np.nan
    add = t["p"] <= 0.025 and abs(msp) >= 0.05
    ADD += [c] if add else []
    print(f"item 5: {c}: stratified AUC {t['auc']:.4f}, floor mean {t['floor'].mean():.4f}, two-sided p {t['p']:.4f}; "
          f"within-stratum Spearman with A_rank among 0 rows {msp:+.4f} -> {'ADD ITS TERCILE TO THE STRATA' if add else 'no change'}")
print(f"item 5: covariates whose tercile joins the strata: {ADD if ADD else 'none'}")

# reported rivals: non-arm placements and demotions as pseudo-positives (the arm positives left out)
yna = np.where(np.isfinite(y_pool), 0.0, np.nan)
yna[R.y_last.eq(1.0).to_numpy()] = np.nan
il = R[R.il14 & ~R.next_arm & R.next_text].copy()
il["pl"] = il.day + pd.to_timedelta(il.days_to_il, unit="D")
_last = il.sort_values("day").groupby(["pitcher", "pl"]).tail(1)
last = R.index.get_indexer(_last[_last.days_to_il.ge(1)].index)
base = g["pool_base"](R, P)
y1 = yna.copy()
y1[last[base[last]]] = 1.0
tn, tnp = g["pool_test"](ns, "A_rank", y=y1), g["pool_test"](ns, "A_rank_pos", y=y1)
print(f"reported: non-arm last outings as positives: {tn['pos']}, stratified AUC {tn['auc']:.4f} / {tnp['auc']:.4f}, p {tn['p']:.4f} / {tnp['p']:.4f}")
dem = (R.ok14 & R.other14 & ~R.il14 & R.early_hook.eq(0.0) & np.isfinite(R.A_rank) & np.isfinite(R.A_rank_pos) & P.cls.ge(0)).to_numpy()
y2 = yna.copy()
y2[dem] = 1.0
td, tdp = g["pool_test"](ns, "A_rank", y=y2), g["pool_test"](ns, "A_rank_pos", y=y2)
print(f"reported: last outings before a Triple-A outing within 14 days as positives: {td['pos']}, stratified AUC "
      f"{td['auc']:.4f} / {tdp['auc']:.4f}, p {td['p']:.4f} / {tdp['p']:.4f}")
ya, ncov = g["arm_vs_nonarm_labels"](R, P, ns["il_lab"], ns["il_by"])
sa, sap = g["pool_test"](ns, "A_rank", y=ya, rule=g["violates_both"]), g["pool_test"](ns, "A_rank_pos", y=ya, rule=g["violates_both"])
print(f"reported: secondary arm vs non-arm 2024-2026 (COVID-related rows left out {ncov}): arm {sa['pos']}, non-arm "
      f"{sa['rows'] - sa['pos']}, stratified AUC {sa['auc']:.4f} / {sap['auc']:.4f}, p {sa['p']:.4f} / {sap['p']:.4f}")

if SMOKE:  # code test: fake placements rarely reach the pool, so the template is 30 random pool rows
    _ix = np.random.default_rng(6).choice(q1["index"], min(30, len(q1["index"]) // 5), replace=False)
    tm = P.loc[_ix]
    _dtl = np.random.default_rng(7).integers(1, 15, len(_ix))
else:
    tm = P.loc[q1["index"][q1["y"]]]
    _dtl = R.loc[tm.index, "days_to_il"].to_numpy()
    assert ((_dtl >= 1) & (_dtl <= 14)).all(), "a positive outside 1-14 days"
template = pd.DataFrame({"role": tm.role.to_numpy(), "gkey": gkey(tm.gap.to_numpy()), "frac": tm.frac.to_numpy(),
                         "days_to_il": _dtl})
print(f"template: {len(template)} positives; none {np.mean(template.gkey == -1):.3f}, median gap "
      f"{np.median(template.gkey[template.gkey >= 0]):.0f}, starters {template.role.mean():.3f}")
NULL_2426 = null_run(ns, template, "2024-2026", 31)
LEAGUE_2025 = {"lsc": dict(SEEN["lsc"]), "lg": SEEN["lg"].copy()}
R.drop(columns="_gap", inplace=True)
del ns, R, P, q1, q1p, r1
gc.collect()

# %% [markdown]
# ## 2. 2021-2023: rows with no placement within 14 days only (the build's outputs are discarded)

# %%
FIXED.update(LEAGUE_2025)
SEEN.clear()
ns = run("confirm", SMOKE, quiet=True)
assert ns["ROW_SEASONS"] == [2021, 2022, 2023]
NULL_2123 = null_run(ns, template, "2021-2023", 32)
print(f"DIAGNOSTIC 3: item 4 share passing 2024-2026 {NULL_2426:.3f}, 2021-2023 {NULL_2123:.3f} (each must be <= 0.05); "
      f"item 5 terciles to add: {ADD if ADD else 'none'} -> "
      f"{'C2 may be frozen' if max(NULL_2426, NULL_2123) <= 0.05 and not ADD else 'C2 is NOT frozen: redesign'}")
