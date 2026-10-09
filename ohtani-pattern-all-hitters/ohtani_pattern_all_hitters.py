# %% [markdown]
# # Does Ohtani's 2026 Pattern Hold for Every MLB Hitter?
#
# The notebook [Ohtani 2026: Could the Slump Be Seen Early?](https://www.kaggle.com/code/yasunorim/ohtani-2026-could-the-slump-be-seen-early) found two things in one player:
#
# - a **bat-speed drop of 3.1 mph over ten game days** in early August 2026, larger than any 10-day change in the healthy seasons it was compared with, a month before his injured list;
# - **process flags** (exit velocity, hard contact, bat speed) that went up about as often as the **results** flag, sometimes earlier, sometimes with nothing following.
#
# One player is not a pattern. This notebook runs the same measurements on **every MLB hitter, 2024-2026**, against every injured-list placement in those seasons. The tests, thresholds and success criteria were committed before any of this data was fetched: [FREEZE_population.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_population.md) (first committed in `6d208bf` before any data was fetched; two amendments `fe1e790` and the one shipped with this notebook, also before the data, are at the end of that file).
#
# Data: **[MLB Statcast + Bat Tracking 2024-2026](https://www.kaggle.com/datasets/yasunorim/mlb-statcast-bat-tracking-2024-2025)** (every regular-season pitch). Injured-list placements and player positions come from MLB StatsAPI (internet on).

# %%
import glob
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 12,
                     "xtick.labelsize": 12, "ytick.labelsize": 12})

INPUT_ROOT = os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input")
YEARS = [int(y) for y in os.environ.get("YEARS", "2024,2025,2026").split(",")]
# Window sizes are the frozen ones; SMOKE shrinks them only for a code test on a few days of data.
SMOKE = os.environ.get("SMOKE") == "1"
W_BS, W_BIP, W_SW, W_PA, LAG = (10, 5, 15, 10, 2) if SMOKE else (100, 40, 150, 100, 10)
DROP, HORIZON, MIN_PREV_PA = -2.0, 30, (20 if SMOKE else 300)
OHTANI = 660271

COLS = ["batter", "game_date", "game_pk", "at_bat_number", "pitch_number", "type", "description",
        "events", "launch_speed", "bat_speed", "woba_value", "woba_denom"]
frames = []
for y in YEARS:
    p = glob.glob(os.path.join(INPUT_ROOT, "**", f"statcast_{y}.parquet"), recursive=True)
    if not p:
        raise FileNotFoundError(f"Attach yasunorim/mlb-statcast-bat-tracking-2024-2025 (statcast_{y}.parquet).")
    d = pd.read_parquet(p[0], columns=COLS)
    d["season"] = y
    frames.append(d)
pitches = pd.concat(frames, ignore_index=True)
del frames
pitches["game_date"] = pd.to_datetime(pitches["game_date"])
pitches = pitches.sort_values(["batter", "game_date", "game_pk", "at_bat_number", "pitch_number"], kind="stable")
print(f"{len(pitches):,} pitches, {pitches.batter.nunique():,} batters, seasons {YEARS}")

# %% [markdown]
# ## Injured-list placements and positions (MLB StatsAPI)

# %%
API = "https://statsapi.mlb.com/api/v1"
UA = {"User-Agent": "kaggle-notebook-research/1.0"}


def get(path, **params):
    for k in range(4):
        try:
            r = requests.get(f"{API}{path}", params=params, headers=UA, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"StatsAPI {path} failed")


rows = []
for y in YEARS:
    for m in range(2, 11):
        s = pd.Timestamp(y, m, 1)
        e = s + pd.offsets.MonthEnd(0)
        for t in get("/transactions", startDate=f"{s:%Y-%m-%d}", endDate=f"{e:%Y-%m-%d}", sportId=1)["transactions"]:
            d = t.get("description", "")
            if "injured list" in d.lower() and t.get("person"):
                kind = "placed" if "placed" in d.lower() else "activated" if "activated" in d.lower() else None
                if kind:
                    rows.append({"batter": t["person"]["id"], "kind": kind, "description": d,
                                 "effective": t.get("effectiveDate", t.get("date"))})
        time.sleep(0.5)
tx = pd.DataFrame(rows)
tx["effective"] = pd.to_datetime(tx["effective"].str[:10])
act = tx[tx.kind == "activated"].drop_duplicates(["batter", "effective"])
il = tx[tx.kind == "placed"].drop_duplicates(["batter", "effective"]).sort_values(["batter", "effective"])
# Placements of the same batter less than 10 days apart are one stint; keep the earliest (AMENDMENTS 2).
il = il[~(il.batter.eq(il.batter.shift()) & (il.effective - il.effective.shift()).dt.days.lt(10))].reset_index(drop=True)
UPPER = ["shoulder", "elbow", "wrist", "hand", "finger", "thumb", "biceps", "triceps", "forearm", "back",
         "oblique", "rib", "intercostal", "neck"]
LOWER = ["hamstring", "quad", "calf", "knee", "ankle", "foot", "hip", "groin", "toe", "heel", "achilles"]
# Region from the injury sentence only (the last sentence), by word start (AMENDMENTS 1 and 2).
last = il.description.str.strip().str.rstrip(".").str.split(". ", regex=False).str[-1].str.lower()
injury = last.where(~last.str.contains("injured list"), "")


def has(words):
    pat = r"\b(?:" + "|".join(w.rstrip("s") for w in words) + r")"
    return injury.str.contains(pat, regex=True)


up_, lo_ = has(UPPER), has(LOWER)
il["region"] = np.select([up_ & ~lo_, lo_ & ~up_], ["upper", "lower"], "other")
# Most placements state an injury; if almost none were classified the parsing is broken.
assert (injury != "").mean() > 0.9 and (il.region != "other").mean() > 0.6, il.region.value_counts()
assert il.loc[il.description.str.contains("Shohei Ohtani") & (il.effective == "2026-09-08"), "region"].tolist() in ([["upper"]] if 2026 in YEARS else [[]])
assert ((il.batter == OHTANI) & (il.effective == "2026-09-08")).sum() == (1 if 2026 in YEARS else 0)
print(len(il), "placements;", il.region.value_counts().to_dict())

ids = pitches.batter.unique()
pos = {}
for i in range(0, len(ids), 150):
    for p in get("/people", personIds=",".join(map(str, ids[i:i + 150])))["people"]:
        pos[p["id"]] = p.get("primaryPosition", {}).get("abbreviation")
assert len(pos) == len(ids), "every batter needs a position"
pitchers = {b for b, a in pos.items() if a == "P"}
print(f"{len(pitchers)} batters with primary position P are left out")

# %% [markdown]
# ## Signals, day by day (only earlier pitches)

# %%
SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt",
         "missed_bunt", "bunt_foul_tip"}
MISS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
LOWER_IS_WORSE = {"EV90": True, "Hard-hit": True, "Bat speed": True, "Whiff": False, "wOBA": True}


def windowed(dates, vals, w, f, days):
    out = np.full(len(days), np.nan)
    ks = np.searchsorted(dates, days, side="left")  # events strictly before each day
    for i, k in enumerate(ks):
        if k >= w:
            out[i] = f(vals[k - w:k])
    return out


def streams(b):
    bip = b[(b.type == "X") & b.launch_speed.notna()]
    sw = b[b.description.isin(SWING)]
    bs = sw[sw.bat_speed >= 50]
    pa = b[b.woba_denom.gt(0)]
    return {
        "EV90": (bip.game_date.to_numpy(), bip.launch_speed.to_numpy(), W_BIP, lambda v: np.percentile(v, 90)),
        "Hard-hit": (bip.game_date.to_numpy(), (bip.launch_speed >= 95).to_numpy(float), W_BIP, np.mean),
        "Bat speed": (bs.game_date.to_numpy(), bs.bat_speed.to_numpy(float), W_BS, np.mean),
        "Whiff": (sw.game_date.to_numpy(), sw.description.isin(MISS).to_numpy(float), W_SW, np.mean),
        "wOBA": (pa.game_date.to_numpy(), np.c_[pa.woba_value.to_numpy(float), pa.woba_denom.to_numpy(float)],
                 W_PA, lambda v: v[:, 0].sum() / v[:, 1].sum()),
    }


def flag_up(x, line, lower_worse, run=3):
    bad = np.where(np.isnan(x), False, (x < line) if lower_worse else (x > line))
    up, streak = np.zeros(len(x), bool), 0
    for i, z in enumerate(bad):
        streak = streak + 1 if z else 0
        up[i] = streak >= run
    return up


groups = {k: g for k, g in pitches[~pitches.batter.isin(pitchers)].groupby(["batter", "season"], sort=False)}
pa_count = {k: int(g.events.notna().sum()) for k, g in groups.items()}
series = {}
for (b, y), g in groups.items():
    days = np.unique(g.game_date.to_numpy())
    st = streams(g)
    sig = {k: windowed(*v[:2], v[2], v[3], days) for k, v in st.items()}
    series[(b, y)] = (days, sig)

day_rows = []
for (b, y), (days, sig) in series.items():
    if (b, y) == (OHTANI, 2026):
        continue
    bs = sig["Bat speed"]
    d10 = np.full(len(days), np.nan)
    d10[LAG:] = bs[LAG:] - bs[:-LAG]
    rec = {"batter": b, "season": y, "day": days, "d10": d10}
    prev = series.get((b, y - 1))
    if prev is not None and pa_count.get((b, y - 1), 0) >= MIN_PREV_PA:
        for k in LOWER_IS_WORSE:
            base = prev[1][k][~np.isnan(prev[1][k])]
            if len(base) >= (2 if SMOKE else 20):
                line = np.percentile(base, 5 if LOWER_IS_WORSE[k] else 95)
                rec[f"flag_{k}"] = flag_up(sig[k], line, LOWER_IS_WORSE[k])
    day_rows.append(pd.DataFrame(rec))
days_df = pd.concat(day_rows, ignore_index=True)
days_df["day"] = pd.to_datetime(days_df["day"])

# Planted check: no day may use its own or later pitches. Rebuild one batter-season with that day's
# own pitches set to absurd values; the value on that day must not move.
(b0, y0), (days0, sig0) = max(series.items(), key=lambda kv: np.isfinite(kv[1][1]["Bat speed"]).sum())
assert np.isfinite(sig0["Bat speed"]).sum() >= 3
g0 = groups[(b0, y0)].copy()
t0 = days0[np.where(np.isfinite(sig0["Bat speed"]))[0][2]]
g0.loc[g0.game_date >= t0, "bat_speed"] = 200.0
again = windowed(*streams(g0)["Bat speed"][:2], W_BS, np.mean, days0)
i0 = list(days0).index(t0)
assert np.array_equal(again[: i0 + 1], sig0["Bat speed"][: i0 + 1], equal_nan=True)

# Outcome: an IL placement with effective date in [t, t + 30 days].
il_by = il.groupby("batter").effective.apply(lambda s: np.sort(s.to_numpy()))


def next_il(row_b, day_arr):
    eff = il_by.get(row_b)
    if eff is None:
        return np.full(len(day_arr), np.nan)
    k = np.searchsorted(eff, day_arr, side="left")
    out = np.full(len(day_arr), np.nan)
    ok = k < len(eff)
    out[ok] = (eff[k[ok]] - day_arr[ok]) / np.timedelta64(1, "D")
    return out


days_df = days_df.sort_values(["batter", "day"]).reset_index(drop=True)
days_df["days_to_il"] = np.concatenate([next_il(b, g.day.to_numpy()) for b, g in days_df.groupby("batter", sort=False)])
days_df["il30"] = days_df.days_to_il <= HORIZON
print(f"{len(days_df):,} batter-days, {days_df.batter.nunique():,} batters, IL within 30 days on {days_df.il30.mean():.2%} of days")

# %% [markdown]
# ## H1: does a sharp bat-speed drop come before the injured list?
#
# Exposure: bat speed (last 100 swings) at least **2.0 mph lower** than ten game days earlier. Outcome: an injured-list placement within the next 30 days. Supported if the relative risk's 95% interval (bootstrap over batters) is above 1.

# %%
rng = np.random.default_rng(0)


def rr_boot(df, expo, out="il30", n=2000):
    """Relative risk of `out` for exposed vs unexposed batter-days, CI by resampling batters."""
    g = pd.DataFrame({"b": df.batter, "e": expo.astype(bool), "o": df[out].astype(bool)})
    if g.e.sum() == 0 or (~g.e).sum() == 0:  # nothing to compare (shown as empty in the table)
        return dict(rr=np.nan, lo=np.nan, hi=np.nan, exposed_days=int(g.e.sum()), exposed_il=0, ppv=np.nan, base=np.nan)
    g["oe"], g["ou"], g["u"] = g.e & g.o, ~g.e & g.o, ~g.e
    per = g.groupby("b")[["e", "oe", "u", "ou"]].sum().astype(float)  # exposed, exposed+IL, unexposed, unexposed+IL
    tot = per.sum()
    rr = (tot["oe"] / tot["e"]) / (tot["ou"] / tot["u"])
    m = per.to_numpy()
    w = rng.multinomial(len(m), np.full(len(m), 1 / len(m)), size=n)
    s = w @ m
    rrs = (s[:, 1] / s[:, 0]) / (s[:, 3] / s[:, 2])
    lo, hi = np.nanpercentile(rrs, [2.5, 97.5])
    return dict(rr=rr, lo=lo, hi=hi, exposed_days=int(tot["e"]), exposed_il=int(tot["oe"]),
                ppv=tot["oe"] / tot["e"], base=tot["ou"] / tot["u"])


h1 = days_df[days_df.d10.notna()].copy()
h1["drop"] = h1.d10 <= DROP
res = {"all seasons": rr_boot(h1, h1["drop"])}
for y in YEARS:
    s = h1[h1.season == y]
    res[str(y)] = rr_boot(s, s["drop"])
# Sensitivity: leave out days within 30 days after an IL activation (returning can itself lower bat speed).
act_by = act.groupby("batter").effective.apply(lambda s: np.sort(s.to_numpy()))


def days_since_act(b, day_arr):
    a = act_by.get(b)
    if a is None:
        return np.full(len(day_arr), np.inf)
    k = np.searchsorted(a, day_arr, side="right") - 1
    out = np.full(len(day_arr), np.inf)
    ok = k >= 0
    out[ok] = (day_arr[ok] - a[k[ok]]) / np.timedelta64(1, "D")
    return out


h1["since_act"] = np.concatenate([days_since_act(b, g.day.to_numpy()) for b, g in h1.groupby("batter", sort=False)])
s = h1[h1.since_act > HORIZON]
res["excl. 30 days after return"] = rr_boot(s, s["drop"])
tab_h1 = pd.DataFrame(res).T
print(tab_h1.round(4).to_string())
auc = roc_auc_score(h1.il30, -h1.d10)
print(f"AUC of -D10 for IL within 30 days: {auc:.4f}")

# The concatenations above align only if each batter's rows are contiguous and in day order.
assert (h1.batter != h1.batter.shift()).sum() == h1.batter.nunique()
assert not h1.duplicated(["batter", "day"]).any() and h1.groupby("batter", sort=False).day.is_monotonic_increasing.all()

# %%
# Sensitivity: of the IL placements whose 30 days before are covered by D10, how many had a drop?
cover = []
for _, r in il.iterrows():
    w = h1[(h1.batter == r.batter) & (h1.day < r.effective) & (h1.day >= r.effective - pd.Timedelta(days=HORIZON))]
    if len(w):
        cover.append({"region": r.region, "had_drop": bool(w["drop"].any())})
cover = pd.DataFrame(cover, columns=["region", "had_drop"])
print(f"IL placements with D10 in the 30 days before: {len(cover)}; with a drop: {cover.had_drop.mean():.1%}")


# False alarms: runs of exposed days with no IL in the 30 days after the run starts.
def runs(x):
    return np.flatnonzero(np.diff(np.r_[0, x.astype(int)]) == 1)


fa, ep = 0, 0
for _, g in h1.groupby(["batter", "season"]):
    st = runs(g["drop"].to_numpy())
    ep += len(st)
    fa += int((~g.il30.to_numpy()[st]).sum())
n_bs = h1.groupby(["batter", "season"]).ngroups
print(f"drop episodes: {ep} in {n_bs} batter-seasons; {fa} ({fa / max(ep, 1):.0%}) with no IL in the next 30 days")

fig, ax = plt.subplots(figsize=(10, 5))
t = tab_h1
ypos = np.arange(len(t))[::-1]
ax.errorbar(t.rr, ypos, xerr=[t.rr - t.lo, t.hi - t.rr], fmt="o", color="#2c3e50", ms=9, capsize=5, lw=2)
ax.axvline(1, color="0.5", lw=1.5)
ax.set_yticks(ypos, t.index)
ax.set_xlabel("Relative risk of an IL placement within 30 days (2+ mph drop vs other days)")
verdict = "above 1" if res["all seasons"]["lo"] > 1 else "not clearly above 1"
ax.set_title(f"A sharp bat-speed drop: relative risk {res['all seasons']['rr']:.2f}, {verdict}",
             fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Upper body vs lower body
#
# The frozen secondary prediction: the drop matters more for upper-body injuries (shoulder, elbow, wrist, hand, back, oblique...) than for leg injuries, which a swing should not see coming.

# %%
by_region = {}
for reg in ["upper", "lower"]:
    eff = il[il.region == reg].groupby("batter").effective.apply(lambda s: np.sort(s.to_numpy()))
    def nxt(b, d, eff=eff):
        e = eff.get(b)
        if e is None:
            return np.full(len(d), np.nan)
        k = np.searchsorted(e, d, side="left")
        o = np.full(len(d), np.nan)
        ok = k < len(e)
        o[ok] = (e[k[ok]] - d[ok]) / np.timedelta64(1, "D")
        return o
    h1[f"il30_{reg}"] = np.concatenate([nxt(b, g.day.to_numpy()) for b, g in h1.groupby("batter", sort=False)]) <= HORIZON
    by_region[reg] = rr_boot(h1, h1["drop"], out=f"il30_{reg}")
print(pd.DataFrame(by_region).T.round(4).to_string())
# Decision rule (AMENDMENTS 2): paired bootstrap over batters of RR_upper / RR_lower, lower bound > 1.
rng1 = np.random.default_rng(1)
g = pd.DataFrame({"b": h1.batter, "e": h1["drop"]})
cols = []
for reg in ["upper", "lower"]:
    o = h1[f"il30_{reg}"]
    g[f"oe_{reg}"], g[f"ou_{reg}"] = g.e & o, ~g.e & o
g["u"] = ~g.e
per = g.groupby("b")[["e", "u", "oe_upper", "ou_upper", "oe_lower", "ou_lower"]].sum().astype(float).to_numpy()
w = rng1.multinomial(len(per), np.full(len(per), 1 / len(per)), size=2000) @ per
rr_u = (w[:, 2] / w[:, 0]) / (w[:, 3] / w[:, 1])
rr_l = (w[:, 4] / w[:, 0]) / (w[:, 5] / w[:, 1])
ratio = rr_u / rr_l
lo_r, hi_r = np.nanpercentile(ratio, [2.5, 97.5])
tot = per.sum(0)
point = ((tot[2] / tot[0]) / (tot[3] / tot[1])) / ((tot[4] / tot[0]) / (tot[5] / tot[1]))
print(f"RR_upper / RR_lower = {point:.3f} (95% {lo_r:.3f}-{hi_r:.3f}); "
      f"{'supported' if lo_r > 1 else 'not supported'}; resamples dropped as undefined: {int(np.isnan(ratio).sum())}")

# %% [markdown]
# ## H2: do process flags warn earlier than the results flag?
#
# The level rule from the Ohtani notebook (worse than the 5th percentile of the batter's previous season, 3 game days in a row), for every batter-season with a previous season of 300+ PA. A relative risk does not measure lead time; what it can show is whether a process flag carries more injury risk than the results flag. Baselines need 20+ days in the previous season, so with data from 2024 this covers 2025 and 2026.

# %%
h2 = {}
for k in LOWER_IS_WORSE:
    c = f"flag_{k}"
    if c in days_df:
        d = days_df[days_df[c].notna()]
        h2[k] = rr_boot(d, d[c].astype(bool))
tab_h2 = pd.DataFrame(h2).T
print(tab_h2.round(4).to_string())
lead = [k for k in tab_h2.index if k != "wOBA" and tab_h2.loc[k, "lo"] > tab_h2.loc["wOBA", "rr"]]
print("process signals whose interval is above the wOBA estimate:", lead or "none")

fig, ax = plt.subplots(figsize=(10, 5))
ypos = np.arange(len(tab_h2))[::-1]
colors = ["#c0392b" if k == "wOBA" else "#2c3e50" for k in tab_h2.index]
for y_, (k, r) in zip(ypos, tab_h2.iterrows()):
    ax.errorbar(r.rr, y_, xerr=[[r.rr - r.lo], [r.hi - r.rr]], fmt="o", color=colors[list(tab_h2.index).index(k)],
                ms=9, capsize=5, lw=2)
ax.axvline(1, color="0.5", lw=1.5)
ax.set_yticks(ypos, tab_h2.index)
ax.set_xlabel("Relative risk of an IL placement within 30 days (flag up vs not)")
ax.set_title("Process flags vs the results flag (red)", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()
