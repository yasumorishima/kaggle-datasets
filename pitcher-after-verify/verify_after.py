# %% [markdown]
# # Independent check of the last-outing study's P1 (after the single frozen run)
#
# The frozen notebook (`pitcher-after-il/pitcher_after_il.py`, md5 a96dcbef03d67e4cfc9404279ac2e566, merged at
# 5bbe6ed) printed P1 SUPPORTED. This notebook does not change that verdict and decides nothing. It runs the frozen
# notebook up to its P1 section (rows, labels and A as frozen), then rebuilds, with code written separately from the
# frozen functions: the target from the transactions, the velocity drop A from the raw pitches, the daily rank, the
# AUC (Mann-Whitney), the whole-group rotation floor (other seeds) and the pitcher bootstrap (other seed). Each rebuilt
# value is compared with the frozen one. A last section is descriptive only (not pre-registered).

# %%
import hashlib
import os

import numpy as np
import pandas as pd
import requests

VERIFY_SMOKE = os.environ.get("VERIFY_SMOKE") == "1"  # code test on the sample with random placements only
if not VERIFY_SMOKE:
    os.environ.pop("SMOKE_FAKE_LABELS", None)
NB_URL = ("https://raw.githubusercontent.com/yasumorishima/kaggle-datasets/"
          "5bbe6ed4df1249c8673600367d9c9941a86531a6/pitcher-after-il/pitcher_after_il.py")
nb = requests.get(NB_URL, timeout=60).text
assert hashlib.md5(nb.encode()).hexdigest() == "a96dcbef03d67e4cfc9404279ac2e566", "not the frozen notebook"
CUT = "# %% [markdown]\n# ## P1: does the last outing before an arm placement look slow"
assert nb.count(CUT) == 1
exec(compile(nb[: nb.index(CUT)], "frozen_after_up_to_p1", "exec"))
assert FAKE == VERIFY_SMOKE

# %% [markdown]
# ## 1. The target, rebuilt from the transactions

# %%
R = rows_m.reset_index(drop=True).copy()
R["day"] = pd.to_datetime(R.day)
pl = il_lab[["pitcher", "effective", "arm"]].copy()
pl["effective"] = pd.to_datetime(pl.effective)
pl = pl.sort_values(["pitcher", "effective"]).drop_duplicates(["pitcher", "effective"])
pl_by = {b: (g.effective.to_numpy(), g.arm.to_numpy()) for b, g in pl.groupby("pitcher")}

y_ind = np.full(len(R), np.nan)
gap_ind = np.full(len(R), np.nan)
for b, g in R.groupby("pitcher"):
    eff, arm = pl_by.get(b, (np.array([], dtype="datetime64[ns]"), np.array([], dtype=bool)))
    days = g.day.to_numpy()
    j = np.searchsorted(eff, days, side="left")  # first placement effective on or after the row day
    has = j < len(eff)
    gap = np.full(len(days), np.inf)
    if len(eff):
        gap[has] = (eff[j[has]] - days[has]) / np.timedelta64(1, "D")
    gap_ind[g.index] = gap
    use = g.use14.to_numpy()
    y_ind[g.index[use & (gap > 14)]] = 0.0
    for jj in np.unique(j[has & (gap <= 14)]):
        m = np.flatnonzero((j == jj) & has & (gap <= 14))
        last = m[np.argmax(days[m])]
        if arm[jj] and gap[last] >= 1 and use[last]:
            y_ind[g.index[last]] = 1.0
# the game-log rule, rebuilt: a game-log appearance at this level after the day and on or before the placement
n_contra = 0
for i in np.flatnonzero(y_ind == 1.0):
    b, d, k = R.pitcher.iat[i], R.day.iat[i], gap_ind[i]
    v = pd.to_datetime(app_days["mlb"].get(b, np.array([], dtype="datetime64[ns]")))
    if ((v > d) & (v <= d + pd.Timedelta(days=int(k)))).any():
        y_ind[i] = np.nan
        n_contra += 1
fz = R.y_last.to_numpy()
same_y = np.array_equal(fz, y_ind, equal_nan=True)
print(f"target: rebuilt positives {int(np.nansum(y_ind == 1)):,}, negatives {int(np.nansum(y_ind == 0)):,}, "
      f"game-log left out {n_contra}; frozen positives {int(np.nansum(fz == 1)):,}, negatives {int(np.nansum(fz == 0)):,}; "
      f"identical: {same_y}")
if not same_y:
    dif = np.flatnonzero(~((fz == y_ind) | (np.isnan(fz) & np.isnan(y_ind))))
    print("first differing rows:\n" + R.iloc[dif[:10]][["pitcher", "season", "day", "days_to_il", "use14"]]
          .assign(frozen=fz[dif[:10]], rebuilt=y_ind[dif[:10]]).to_string())

# %% [markdown]
# ## 2. The velocity drop A, rebuilt from the raw pitches

# %%
TYPES = ["FF", "SI", "FC"]
P = ctx["mlb"][["pitcher", "season", "game_date", "game_pk", "at_bat_number", "pitch_number", "pitch_type",
                "release_speed", "inning"]].copy()
P = P.sort_values(["pitcher", "season", "game_date", "game_pk", "at_bat_number", "pitch_number"], kind="stable")
P["pitch_type"] = P.pitch_type.fillna("")


def mad_(v):
    v = np.asarray(v, float)
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


CNT = P[P.pitch_type.isin(TYPES)].groupby(["pitcher", "season"]).pitch_type.value_counts().unstack(fill_value=0)
PREV_MODE = {k: max(TYPES, key=lambda t: (int(row.get(t, 0)), -TYPES.index(t))) for k, row in CNT.iterrows()
             if max(int(row.get(t, 0)) for t in TYPES) > 0}
recs = []
for (b, y), g in P.groupby(["pitcher", "season"], sort=False):
    prev_mode = PREV_MODE.get((b, y - 1))
    by_day = list(g.groupby("game_date", sort=True))
    seen = {t: [] for t in TYPES}  # speeds of each type thrown before the current day, in order
    for d, gd in by_day:
        n_before = {t: len(seen[t]) for t in TYPES}
        if max(n_before.values()) > 0:
            T = max(TYPES, key=lambda t: (n_before[t], -TYPES.index(t)))
        else:
            T = prev_mode
        x = normal = np.nan
        if T is not None:
            v_day = gd.release_speed[gd.pitch_type == T].to_numpy(float)
            v_day = v_day[np.isfinite(v_day)]
            if len(v_day) >= 5:
                x = float(v_day.mean())
            w = np.asarray(seen[T][-60:], float)
            if len(w) == 60 and np.isfinite(w).sum() >= 30:
                normal = float(np.nanmean(w))
        ng = gd.game_pk.nunique()
        recs.append((b, y, d, T, x, normal, ng, float(gd.inning.min() == 1) if ng == 1 else np.nan))
        for t in TYPES:
            seen[t].extend(gd.release_speed[gd.pitch_type == t].to_numpy(float).tolist())
V = pd.DataFrame(recs, columns=["pitcher", "season", "day", "T", "x", "normal", "games", "start"])
V["dev"] = np.where(V.games > 1, np.nan, V.x - V.normal)
own = np.full(len(V), np.nan)
for _, ix in V.groupby(["pitcher", "season"]).indices.items():
    dv = V.dev.to_numpy()[ix]
    for k in range(len(ix)):
        prior = dv[:k][np.isfinite(dv[:k])]
        if len(prior) >= 5:
            own[ix[k]] = mad_(prior)
V["own_ind"] = own
R2 = R.merge(V, on=["pitcher", "season", "day"], how="left", suffixes=("", "_ind"), validate="one_to_one")
assert len(R2) == len(R)
# the league scale, rebuilt: 2025 pitcher-seasons by season role (more than half of his single-game outings start)
V["sr"] = V.groupby(["pitcher", "season"]).start.transform(lambda v: float(np.nanmean(v) > 0.5))
lsc_ind = {}
for role in (0.0, 1.0):
    ms = [mad_(v[np.isfinite(v)]) for (b, y), v in V[V.sr.eq(role) & V.season.eq(2025)].groupby(["pitcher", "season"]).dev.apply(np.asarray).items()
          if np.isfinite(v).sum() >= 20]
    lsc_ind[role] = float(np.median(ms))
print(f"league scale rebuilt: relievers {lsc_ind[0.0]:.4f}, starters {lsc_ind[1.0]:.4f} (frozen printed above as league-typical scale)")
for c in ("x", "normal", "dev"):
    assert f"{c}_ind" in R2.columns, c
ls = np.where(R2.role.eq(1.0), lsc_ind[1.0], lsc_ind[0.0])
sc = np.where(np.isfinite(R2.own_ind), np.maximum(R2.own_ind, 0.5 * ls), ls)
R2["A_ind"] = np.clip(-(R2.dev_ind / sc), -10, 10)
fa, ia = R2.A.to_numpy(), R2.A_ind.to_numpy()
both = np.isfinite(fa) & np.isfinite(ia)
print(f"A: defined frozen {np.isfinite(fa).sum():,}, rebuilt {np.isfinite(ia).sum():,}, both {both.sum():,}; "
      f"max |frozen - rebuilt| on both {np.max(np.abs(fa[both] - ia[both])):.3e}; "
      f"dev max |diff| {np.nanmax(np.abs(R2.dev.to_numpy() - R2.dev_ind.to_numpy())):.3e}")

# %% [markdown]
# ## 3. Daily rank, AUC, floor and bootstrap, rebuilt

# %%


def day_rank(df, col):
    out = np.full(len(df), np.nan)
    for _, ix in df.groupby(["day", "role"]).indices.items():
        v = df[col].to_numpy()[ix]
        f = np.isfinite(v)
        if f.sum():
            r_ = pd.Series(v[f]).rank(method="average").to_numpy()
            out[ix[f]] = (r_ - 0.5) / f.sum()
    return out


def mw_auc(y, s):
    r_ = pd.Series(s).rank(method="average").to_numpy()
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (r_[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


for label, ycol, acol in (("frozen target, rebuilt A", "y_last", "A_ind"), ("rebuilt target, rebuilt A", "y_ind", "A_ind"),
                          ("frozen target, frozen A", "y_last", "A")):
    W = R2.assign(y_ind=y_ind)
    W["rk"] = day_rank(W, acol)
    D = W[W.early_hook.eq(0.0) & W[ycol].notna() & np.isfinite(W.rk)].reset_index(drop=True)
    y = D[ycol].to_numpy() == 1.0
    s = D.rk.to_numpy()
    a = mw_auc(y, s)
    gix = [ix for ix in D.groupby(["pitcher", "season"]).indices.values()]
    rng = np.random.default_rng(424242)
    fl = []
    for _ in range(1000):
        s2 = s.copy()
        for ix in gix:
            if len(ix) >= 7:
                s2[ix] = np.roll(s[ix], int(rng.integers(0, len(ix))))
        fl.append(mw_auc(y, s2))
    fl = np.array(fl)
    pit = D.pitcher.to_numpy()
    upit = np.unique(pit)
    rows_of = {p_: np.flatnonzero(pit == p_) for p_ in upit}
    rb = np.random.default_rng(99)
    bs = []
    for _ in range(2000):
        take = np.concatenate([rows_of[p_] for p_ in rb.choice(upit, len(upit), replace=True)])
        yy = y[take]
        if 0 < yy.sum() < len(yy):
            bs.append(mw_auc(yy, s[take]))
    print(f"{label}: AUC {a:.6f} (frozen printed 0.605933); positives {int(y.sum())}, rows {len(D):,}; "
          f"floor mean {fl.mean():.4f}, max {fl.max():.4f}, p {(1 + (fl >= a).sum()) / 1001:.4f}; "
          f"bootstrap 0.625% {np.percentile(bs, 0.625):.4f}")

# %% [markdown]
# ## 4. Descriptive only (not pre-registered; no verdict)

# %%
W = R2.copy()
D = W[W.early_hook.eq(0.0) & W.y_last.notna() & np.isfinite(W.A_rank)]
pos, neg = D[D.y_last.eq(1.0)], D[D.y_last.eq(0.0)]
print(f"dev (mph), no early hook: positives mean {pos.dev.mean():+.3f} median {pos.dev.median():+.3f}; "
      f"negatives mean {neg.dev.mean():+.3f} median {neg.dev.median():+.3f}")
for nm, m in (("effective 1 day after the outing", pos.days_to_il.eq(1)), ("2 to 14 days after", pos.days_to_il.ge(2))):
    sub = pd.concat([pos[m], neg])
    yy = sub.y_last.to_numpy() == 1.0
    print(f"  {nm}: positives {int(m.sum())}, AUC {mw_auc(yy, sub.A_rank.to_numpy()):.4f}")
for nm, m in (("mid-inning exit", pos.o_mid_exit.eq(1)), ("exit between innings", pos.o_mid_exit.eq(0))):
    sub = pd.concat([pos[m], neg[neg.o_mid_exit.eq(1) if nm == "mid-inning exit" else neg.o_mid_exit.eq(0)]])
    yy = sub.y_last.to_numpy() == 1.0
    print(f"  {nm} (both groups): positives {int(m.sum())}, AUC {mw_auc(yy, sub.A_rank.to_numpy()):.4f}")
