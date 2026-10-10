# %% [markdown]
# # Does a Hitter Drift From His Own Normal Before the Injured List?
#
# [Ohtani 2026 Pattern Across Every MLB Hitter](https://www.kaggle.com/code/yasunorim/ohtani-2026-pattern-across-every-mlb-hitter) tested one hand-picked signal (a 2.0 mph bat-speed drop) and five fixed level flags. None came before the injured list better than chance. This notebook asks the wider question: **does anything about a hitter move away from his own normal before he goes on the injured list?**
#
# - **20 parameters** per hitter and day: bat path (speed, length, attack angle and direction, tilt, contact point), batted balls, plate discipline, plate-appearance results and playing time.
# - Each one is compared with **the same hitter's own past** (a robust z-score), so a slow swinger is not "anomalous" just for being slow.
# - **Anomaly detection without labels** (largest deviation, mean squared deviation, Isolation Forest) and **gradient boosting with labels**, compared with a rival model that knows only age, position, injury history and playing time.
# - Trained on 2024-2025, scored once on 2026. Everything was fixed before any feature was computed: [FREEZE_anomaly.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_anomaly.md).
#
# Data: **[MLB Statcast + Bat Tracking 2024-2026](https://www.kaggle.com/datasets/yasunorim/mlb-statcast-bat-tracking-2024-2025)**. Injured-list moves, positions and birth dates come from MLB StatsAPI (internet on).

# %%
import glob
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.covariance import MinCovDet
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 12,
                     "xtick.labelsize": 12, "ytick.labelsize": 12})

INPUT_ROOT = os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input")
YEARS = [int(y) for y in os.environ.get("YEARS", "2024,2025,2026").split(",")]
TRAIN, TEST = [y for y in YEARS if y < max(YEARS)], max(YEARS)
# SMOKE shrinks windows and resamples only for a code test on a few cached days; SMOKE_FAKE_LABELS replaces
# the injured list with random placements so every path runs without reading the real outcome. Neither is
# set in the real run, and with fake labels no verdict file is written.
SMOKE = os.environ.get("SMOKE") == "1"
FAKE = os.environ.get("SMOKE_FAKE_LABELS") == "1"
# Window sizes (swings, balls in play, pitches seen, chase-zone pitches, swings for whiff, PA): long = H2's
# frozen sizes, short = roughly the last two weeks.
LONG = (8, 4, 12, 6, 8, 6) if SMOKE else (100, 40, 150, 100, 150, 100)
SHORT = (4, 2, 6, 3, 4, 3) if SMOKE else (30, 15, 50, 30, 50, 30)
W_GAME = 2 if SMOKE else 10
BASE_LAG, MIN_BASE, MIN_SCALE, MIN_Z = (0, 2, 3, 3) if SMOKE else (30, 20, 60, 8)
N_BOOT, N_FLOOR, N_FLOOR_P1 = (30, 2, 20) if SMOKE else (2000, 20, 200)
H, H30 = 14, 30
OHTANI = 660271

COLS = ["batter", "game_date", "game_pk", "at_bat_number", "pitch_number", "type", "description", "events",
        "zone", "stand", "hc_x", "hc_y", "launch_speed", "launch_angle", "estimated_woba_using_speedangle",
        "woba_value", "woba_denom", "bat_speed", "swing_length", "attack_angle", "attack_direction",
        "swing_path_tilt", "intercept_ball_minus_batter_pos_x_inches", "intercept_ball_minus_batter_pos_y_inches"]
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
season_last = pitches.groupby("season").game_date.max()
season_first = pitches.groupby("season").game_date.min()
if not SMOKE:  # a truncated file would silently move the censoring line
    for y, d in season_last.items():
        assert pd.Timestamp(y, 9, 20) <= d <= pd.Timestamp(y, 10, 6), (y, d)
print(f"{len(pitches):,} pitches, {pitches.batter.nunique():,} batters, seasons {YEARS}; train {TRAIN}, test {TEST}")

# %% [markdown]
# ## Injured-list moves, positions and birth dates (MLB StatsAPI)
#
# Parsed exactly as in the population notebook: effective date, placements of the same batter less than 10 days apart merged into one stint, body region from the injury sentence.

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


# IL moves from the two seasons before the data too: injury history for the rival model only.
HIST_YEARS = [min(YEARS) - 2, min(YEARS) - 1] + YEARS
rows = []
for y in HIST_YEARS:
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
il = il[~(il.batter.eq(il.batter.shift()) & (il.effective - il.effective.shift()).dt.days.lt(10))].reset_index(drop=True)
UPPER = ["shoulder", "elbow", "wrist", "hand", "finger", "thumb", "biceps", "triceps", "forearm", "back",
         "oblique", "rib", "intercostal", "neck"]
LOWER = ["hamstring", "quad", "calf", "knee", "ankle", "foot", "hip", "groin", "toe", "heel", "achilles"]
last = il.description.str.strip().str.rstrip(".").str.split(". ", regex=False).str[-1].str.lower()
injury = last.where(~last.str.contains("injured list"), "")


def has(words):
    pat = r"\b(?:" + "|".join(w.rstrip("s") for w in words) + r")"
    return injury.str.contains(pat, regex=True)


up_, lo_ = has(UPPER), has(LOWER)
il["region"] = np.select([up_ & ~lo_, lo_ & ~up_], ["upper", "lower"], "other")
assert (injury != "").mean() > 0.9 and (il.region != "other").mean() > 0.6, il.region.value_counts()
# The one placement the project started from must be found and read as upper body (fails if the parse breaks).
assert il.loc[(il.batter == OHTANI) & (il.effective == "2026-09-08"), "region"].tolist() == (["upper"] if 2026 in YEARS else [])
print(len(il), "placements;", il.region.value_counts().to_dict(), f"{len(act)} activations")

ids = pitches.batter.unique()
people = {}
for i in range(0, len(ids), 150):
    for p in get("/people", personIds=",".join(map(str, ids[i:i + 150])))["people"]:
        people[p["id"]] = (p.get("primaryPosition", {}).get("abbreviation"), p.get("birthDate"), p.get("fullName"))
assert len(people) == len(ids), "every batter needs a person record"
pitchers = {b for b, v in people.items() if v[0] == "P"}
# MLB games played in the previous season (rival model); a traded player's team splits are summed.
prev_games = {}
for y in YEARS:
    for i in range(0, len(ids), 150):
        js = get("/people", personIds=",".join(map(str, ids[i:i + 150])),
                 hydrate=f"stats(group=[hitting],type=[season],season={y - 1})")
        for p in js["people"]:
            sp = [x for s_ in p.get("stats", []) for x in s_.get("splits", []) if x.get("sport", {}).get("id", 1) == 1]
            tot = [x for x in sp if "team" not in x]
            prev_games[(p["id"], y)] = float(sum(x["stat"].get("gamesPlayed", 0) for x in (tot or sp)))
        time.sleep(0.3)
assert len(prev_games) == len(ids) * len(YEARS)
assert SMOKE or prev_games[(OHTANI, 2024)] == 135  # his 2023 games, read from StatsAPI by hand
POS_GROUP = {"C": 0, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1, "LF": 2, "CF": 2, "RF": 2, "OF": 2, "DH": 3, "TWP": 3}
print(f"{len(pitchers)} batters with primary position P are left out")

# %% [markdown]
# ## Twenty parameters, day by day (only earlier pitches)
#
# Each value is a rolling summary of the batter's most recent events **before** the day, within the season. A window value needs at least half of its events with the column filled.

# %%
SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt",
         "missed_bunt", "bunt_foul_tip"}
MISS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
SWING_COLS = {"Bat speed": "bat_speed", "Swing length": "swing_length", "Attack angle": "attack_angle",
              "Attack direction": "attack_direction", "Swing path tilt": "swing_path_tilt",
              "Contact depth": "intercept_ball_minus_batter_pos_y_inches",
              "Contact lateral": "intercept_ball_minus_batter_pos_x_inches"}
BASE18 = list(SWING_COLS) + ["EV90", "Launch angle", "Sweet spot", "Pull", "xwOBAcon", "Swing rate", "Chase",
                             "Whiff", "K", "BB", "wOBA"]
P18 = BASE18
P18S = [n + " (short)" for n in BASE18]
PT = ["Days between games", "PA per game"]
PT_RAW = ["Rest gap", "Early exit"]
ZPARAMS = P18 + P18S + PT
PARAMS = ZPARAMS + PT_RAW


def wmean(dates, v, w, days):
    """Mean of the last w events strictly before each day (needs w events and half of them filled)."""
    v = np.asarray(v, float)
    k = np.searchsorted(dates, days, side="left")
    cs = np.r_[0.0, np.cumsum(np.nan_to_num(v))]
    cn = np.r_[0, np.cumsum(~np.isnan(v))]
    out = np.full(len(days), np.nan)
    ok = k >= w
    s = cs[k[ok]] - cs[k[ok] - w]
    n = cn[k[ok]] - cn[k[ok] - w]
    out[ok] = np.where(n >= w / 2, s / np.maximum(n, 1), np.nan)
    return out


def wratio(dates, a, b, w, days):
    k = np.searchsorted(dates, days, side="left")
    ca, cb = np.r_[0.0, np.cumsum(a)], np.r_[0.0, np.cumsum(b)]
    out = np.full(len(days), np.nan)
    ok = k >= w
    den = cb[k[ok]] - cb[k[ok] - w]
    out[ok] = np.where(den > 0, (ca[k[ok]] - ca[k[ok] - w]) / np.where(den > 0, den, 1), np.nan)
    return out


def wpct(dates, v, w, q, days):
    k = np.searchsorted(dates, days, side="left")
    out = np.full(len(days), np.nan)
    for i, kk in enumerate(k):
        if kk >= w:
            out[i] = np.percentile(v[kk - w:kk], q)
    return out


def params_for(g):
    """g: one batter-season, sorted. Returns (game days, {parameter: values on those days})."""
    days = np.unique(g.game_date.to_numpy())
    gd = g.game_date.to_numpy()
    out = {}
    sw = g[g.description.isin(SWING).to_numpy()]
    comp = sw[sw.bat_speed.ge(50).to_numpy()]
    bip = g[(g.type == "X").to_numpy() & g.launch_speed.notna().to_numpy()]
    bd = bip.game_date.to_numpy()
    la = bip.launch_angle.to_numpy(float)
    sweet = np.where(np.isnan(la), np.nan, ((la >= 8) & (la <= 32)).astype(float))
    ang = np.degrees(np.arctan2(bip.hc_x.to_numpy(float) - 125.42, 198.27 - bip.hc_y.to_numpy(float)))
    pull = np.where(np.isnan(ang), np.nan, np.where(bip.stand.to_numpy() == "R", ang < -15, ang > 15).astype(float))
    ch = g[g.zone.isin([11, 12, 13, 14]).to_numpy()]
    pa = g[g.woba_denom.gt(0).to_numpy()]
    pd_ = pa.game_date.to_numpy()
    for sfx, (w_sw, w_bip, w_seen, w_chase, w_whiff, w_pa) in (("", LONG), (" (short)", SHORT)):
        for name, col in SWING_COLS.items():
            out[name + sfx] = wmean(comp.game_date.to_numpy(), comp[col].to_numpy(float), w_sw, days)
        out["EV90" + sfx] = wpct(bd, bip.launch_speed.to_numpy(float), w_bip, 90, days)
        out["Launch angle" + sfx] = wmean(bd, la, w_bip, days)
        out["Sweet spot" + sfx] = wmean(bd, sweet, w_bip, days)
        out["Pull" + sfx] = wmean(bd, pull, w_bip, days)
        out["xwOBAcon" + sfx] = wmean(bd, bip.estimated_woba_using_speedangle.to_numpy(float), w_bip, days)
        out["Swing rate" + sfx] = wmean(gd, g.description.isin(SWING).to_numpy(float), w_seen, days)
        out["Chase" + sfx] = wmean(ch.game_date.to_numpy(), ch.description.isin(SWING).to_numpy(float), w_chase, days)
        out["Whiff" + sfx] = wmean(sw.game_date.to_numpy(), sw.description.isin(MISS).to_numpy(float), w_whiff, days)
        out["K" + sfx] = wmean(pd_, pa.events.isin(["strikeout", "strikeout_double_play"]).to_numpy(float), w_pa, days)
        out["BB" + sfx] = wmean(pd_, pa.events.eq("walk").to_numpy(float), w_pa, days)
        out["wOBA" + sfx] = wratio(pd_, pa.woba_value.to_numpy(float), pa.woba_denom.to_numpy(float), w_pa, days)
    # Playing time before t (t itself is known: he is in the game). Index i is the day t.
    n = len(days)
    gap, pag, rest, early = (np.full(n, np.nan) for _ in range(4))
    pa_day = pd.Series(g.events.notna().to_numpy(int), index=gd).groupby(level=0).sum().reindex(days).to_numpy(float)
    for i in range(1, n):
        rest[i] = (days[i] - days[i - 1]) / np.timedelta64(1, "D")
        if i - 1 >= 3:  # PA in the previous game minus his median per game before it
            early[i] = pa_day[i - 1] - np.median(pa_day[:i - 1])
        if i >= W_GAME:
            gap[i] = (days[i] - days[i - W_GAME]) / np.timedelta64(1, "D") / W_GAME
            pag[i] = pa_day[i - W_GAME:i].mean()
    out["Days between games"], out["PA per game"], out["Rest gap"], out["Early exit"] = gap, pag, rest, early
    return days, out


groups = {k: g for k, g in pitches[~pitches.batter.isin(pitchers)].groupby(["batter", "season"], sort=False)}
series = {k: params_for(g) for k, g in groups.items()}
print(f"{len(series):,} batter-seasons")

# Planted check: no day may use its own or later pitches. Set every pitch from day t0 on to absurd values;
# no parameter on t0 or earlier may move.
k0 = max(series, key=lambda k: len(series[k][0]))
days0, p0 = series[k0]
t0 = days0[len(days0) // 2]
g0 = groups[k0].copy()
late = (g0.game_date >= t0).to_numpy()
for c in list(SWING_COLS.values()) + ["launch_speed", "launch_angle", "hc_x", "hc_y", "estimated_woba_using_speedangle", "woba_value"]:
    g0.loc[late, c] = 999.0
g0.loc[late, "zone"] = 12.0
g0.loc[late, "description"] = "swinging_strike"
g0.loc[late, "events"] = "strikeout"
_, p0b = params_for(g0)
i0 = list(days0).index(t0)
for name in PARAMS:
    assert np.array_equal(p0[name][: i0 + 1], p0b[name][: i0 + 1], equal_nan=True), name
# ... and the planted values do reach later days, so the check can fail (needs days after t0).
if i0 + 1 < len(days0):
    assert any(not np.array_equal(p0[n][i0 + 1:], p0b[n][i0 + 1:], equal_nan=True) for n in P18S)
else:
    assert SMOKE

# %% [markdown]
# ## Each parameter against the same hitter's own normal
#
# Baseline: his values from the start of the previous season up to **30 days before** the day (so a slowly building change is not absorbed into the baseline). z = (value - median) / (1.4826 x MAD), clipped to +-10.

# %%


def mad(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def own_z(days, x, prev, league_scale):
    """Centre: median of this season's values up to t - BASE_LAG days when there are MIN_BASE of them,
    otherwise the previous season's median (MIN_BASE values). Scale: MAD of his whole previous season when it
    has MIN_SCALE values, otherwise the league-typical scale (a short in-season baseline of overlapping
    windows has a too-small MAD). prev = (days, values) or None. Returns z and the baseline kind per day:
    1/2 in-season centre with own/league scale, 3/4 previous-season centre with own/league scale, 0 none."""
    fin = np.isfinite(x)
    cur_x = x[fin]
    cur_d = days[fin]
    px = prev[1][np.isfinite(prev[1])] if prev is not None else np.array([])
    prev_centre = np.median(px) if len(px) >= MIN_BASE else None
    own_scale = mad(px) if len(px) >= MIN_SCALE else None
    scale = own_scale if own_scale is not None else league_scale
    cut = np.searchsorted(cur_d, days - np.timedelta64(BASE_LAG, "D"), side="right")
    z = np.full(len(days), np.nan)
    kind = np.zeros(len(days), int)
    cache = {}
    for i, c in enumerate(cut):
        if not fin[i]:
            continue
        if c >= MIN_BASE:
            if c not in cache:
                cache[c] = np.median(cur_x[:c])
            centre, k = cache[c], 1
        elif prev_centre is not None:
            centre, k = prev_centre, 3
        else:
            continue
        if scale is not None and scale > 0:
            z[i] = np.clip((x[i] - centre) / scale, -10, 10)
            kind[i] = k + (own_scale is None)
    return z, kind


# League-typical scale per parameter: median over hitters of the MAD of a whole season (MIN_SCALE values) in
# the last training season. Label-free; used only when a hitter has no full previous season.
LEAGUE_SEASON = TRAIN[-1]
league_scale = {}
for n in ZPARAMS:
    ms = [mad(v[np.isfinite(v)]) for (b_, y_), (_, pp) in series.items() if y_ == LEAGUE_SEASON
          for v in [pp[n]] if np.isfinite(v).sum() >= MIN_SCALE]
    league_scale[n] = float(np.median(ms)) if ms else None
print("league-typical scales:", {k: round(v, 4) for k, v in league_scale.items() if v is not None and "(short)" in k})


def zs_for(key):
    days, p = series[key]
    prev = series.get((key[0], key[1] - 1))
    return {n: own_z(days, p[n], None if prev is None else (prev[0], prev[1][n]), league_scale[n]) for n in ZPARAMS}


# Planted checks on synthetic days (run in every mode): values inside the last BASE_LAG days before t must
# not enter t's centre while older values must; early in a season the previous season is the centre; the
# scale is the previous season's MAD, or the league scale when that season is short.
syn_d = np.datetime64("2030-04-01") + np.arange(200).astype("timedelta64[D]")
syn_x = np.random.default_rng(5).normal(70, 2, 200)
j = 150
za, _ = own_z(syn_d, syn_x, None, 2.0)
x1 = syn_x.copy()
recent = (syn_d > syn_d[j] - np.timedelta64(BASE_LAG, "D")) & (syn_d < syn_d[j])
x1[recent] = 500.0
assert za[j] == own_z(syn_d, x1, None, 2.0)[0][j] and np.isfinite(za[j])
x1[:40] = 500.0
assert za[j] != own_z(syn_d, x1, None, 2.0)[0][j]
prev_syn = (syn_d - np.timedelta64(365, "D"), np.random.default_rng(6).normal(60, 2, 200))
early_i = int(np.searchsorted(syn_d, syn_d[0] + np.timedelta64(BASE_LAG, "D"), side="left"))
assert np.isnan(own_z(syn_d, syn_x, None, None)[0][early_i]) and own_z(syn_d, syn_x, prev_syn, 2.0)[0][early_i] > 3
prev_wide = (prev_syn[0], 70 + 2 * (syn_x - 70))
z_own, k_own = own_z(syn_d, syn_x, (prev_syn[0], syn_x), 99.0)
z_wide, _ = own_z(syn_d, syn_x, prev_wide, 99.0)
assert np.isfinite(z_own[j]) and np.isclose(z_wide[j], z_own[j] / 2) and k_own[j] == 1
short_prev = (prev_syn[0][:MIN_SCALE - 1], prev_syn[1][:MIN_SCALE - 1])
z_lg, k_lg = own_z(syn_d, syn_x, short_prev, 4.0)
assert np.isclose(z_lg[j], (syn_x[j] - np.median(syn_x[:j - BASE_LAG + 1][syn_d[:j - BASE_LAG + 1] <= syn_d[j] - np.timedelta64(BASE_LAG, "D")])) / 4.0) and k_lg[j] == 2


# %% [markdown]
# ## One row per hitter and game day

# %%
il_by = {b: (g.effective.to_numpy(), g.region.to_numpy()) for b, g in il.groupby("batter")}
act_by = {b: np.sort(g.effective.to_numpy()) for b, g in act.groupby("batter")}
recs = []
for key, (days, p) in series.items():
    b, y = key
    zk = zs_for(key)
    z = {n: v[0] for n, v in zk.items()}
    r = pd.DataFrame({"batter": b, "season": y, "day": days})
    for n in PARAMS:
        r[n] = p[n]
    for n in ZPARAMS:
        r["z_" + n] = z[n]
    r["prev_games"] = prev_games[(b, y)]
    r["base_kind"] = zk["Bat speed (short)"][1]
    pos, birth, _ = people[b]
    r["pos"] = POS_GROUP.get(pos, np.nan)
    r["age"] = (days - np.datetime64(birth)) / np.timedelta64(1, "D") / 365.25 if birth else np.nan
    eff, reg = il_by.get(b, (np.array([], dtype="datetime64[ns]"), np.array([])))
    k = np.searchsorted(eff, days, side="left")
    r["n_prev_il"] = k
    nxt = np.where(k < len(eff), (eff[np.minimum(k, len(eff) - 1)] - days) / np.timedelta64(1, "D") if len(eff) else np.inf, np.inf)
    r["days_to_il"] = nxt
    r["next_region"] = np.where(k < len(eff), reg[np.minimum(k, len(eff) - 1)] if len(eff) else "", "")
    a = act_by.get(b, np.array([], dtype="datetime64[ns]"))
    ka = np.searchsorted(a, days, side="right")  # an activation on t is known on t
    r["days_since_act"] = np.where(ka > 0, np.minimum((days - a[np.maximum(ka - 1, 0)]) / np.timedelta64(1, "D"), 365) if len(a) else 365, 365)
    r["gd_season"] = np.arange(len(days))
    r["day_of_season"] = (days - np.datetime64(season_first[y])) / np.timedelta64(1, "D")
    recs.append(r)
rows = pd.concat(recs, ignore_index=True)
rows["il14"] = rows.days_to_il <= H
rows["il30"] = rows.days_to_il <= H30
rows["end"] = rows.season.map(season_last)
rows["ok14"] = rows.day + pd.Timedelta(days=H) <= rows.end
rows["ok30"] = rows.day + pd.Timedelta(days=H30) <= rows.end

Z18 = ["z_" + n for n in P18]
Z18S = ["z_" + n for n in P18S]
zm = rows[Z18S].to_numpy()
nz = np.isfinite(zm).sum(1)
rows["U1"] = np.where(nz >= MIN_Z, np.nanmax(np.abs(zm), 1), np.nan)
rows["U2"] = np.where(nz >= MIN_Z, np.nanmean(zm ** 2, 1), np.nan)
oht = rows[(rows.batter == OHTANI) & (rows.season == TEST)].copy()
rows = rows[~((rows.batter == OHTANI) & (rows.season == TEST))].reset_index(drop=True)
if FAKE:  # random placements (about 0.4 per hitter-season), unrelated to any feature
    rng0 = np.random.default_rng(99)
    dti = np.full(len(rows), np.inf)
    dayv = rows.day.to_numpy()
    for _, ix in rows.groupby(["batter", "season"]).indices.items():
        k = min(rng0.poisson(0.4), len(ix))
        if k:
            effs = np.sort(rng0.choice(dayv[ix], k, replace=False)) + np.timedelta64(1, "D")
            kk = np.searchsorted(effs, dayv[ix], side="left")
            ok = kk < len(effs)
            dti[ix[ok]] = (effs[kk[ok]] - dayv[ix][ok]) / np.timedelta64(1, "D")
    rows["days_to_il"] = dti
    rows["il14"], rows["il30"] = rows.days_to_il <= H, rows.days_to_il <= H30
    rows["next_region"] = np.where(rows.days_to_il < np.inf, rng0.choice(["upper", "lower", "other"], len(rows)), "")
train = rows[rows.season.isin(TRAIN) & rows.ok14].reset_index(drop=True)
test = rows[(rows.season == TEST) & rows.ok14].reset_index(drop=True)
assert set(train.season) <= set(TRAIN) and set(test.season) == {TEST}
_m = rows[rows.U2.notna()].assign(month=lambda d: d.day.dt.month).groupby(["season", "month"]).U2.median()
print("median raw U2 by season and month (label-free; the daily rank removes this drift):\n" + _m.round(2).unstack().to_string())
print(f"train {len(train):,} rows ({train.il14.sum():,} within {H} days of an IL placement), "
      f"test {len(test):,} rows ({test.il14.sum():,}); U2 defined on {test.U2.notna().mean():.1%} of test rows")

# %% [markdown]
# ## Anomaly scores without labels, and the models
#
# - **U1** largest |z|, **U2** mean z squared (the primary score), **U3** Isolation Forest fitted on 2024-2025 z vectors.
# - **B0** (rival): age, position group, injury-list history, days since the last activation, point in the season, playing time.
# - **M2** (anomaly model): B0 + the 18 own-normal deviations + U1 + U2. No raw levels, so it cannot learn who the hitter is.
# - **M1**: M2 + the raw values.

# %%
def zs(df):
    return np.nan_to_num(df[Z18S].to_numpy(), nan=0.0)


# U3 and U4 are fitted and scored only where U1/U2 are defined (at least MIN_Z short-window z).
fit_rows = train[train.U2.notna()]
iso = IsolationForest(n_estimators=300, random_state=0).fit(zs(fit_rows))
samp = np.random.default_rng(4).choice(len(fit_rows), min(len(fit_rows), 50000), replace=False)
mcd = MinCovDet(random_state=4).fit(zs(fit_rows)[samp])
for df in (train, test, oht):
    if len(df):
        ok_ = df.U2.notna().to_numpy()
        df["U3"], df["U4"] = np.nan, np.nan
        df.loc[ok_, "U3"] = -iso.score_samples(zs(df[ok_]))
        df.loc[ok_, "U4"] = np.sqrt(mcd.mahalanobis(zs(df[ok_])))
# Each score is also ranked among all hitters on the same day (every value is known that morning): the raw
# level drifts with the point in the season, which would confound a pooled AUC. The "d" versions are scored.
# Daily rank = (midrank - 0.5) / number of hitters with a value that day, so its mean is 0.5 on every day
# (rank / n would run higher on days with few hitters).
for df in (train, test):
    for u in ("U1", "U2", "U3", "U4"):
        g_ = df.groupby("day")[u]
        df[u + "d"] = (g_.rank(method="average") - 0.5) / g_.transform("count")
if len(oht):
    ref = test.loc[test.U2.notna(), ["day", "U2"]]
    by_day = {d: g_.U2.to_numpy() for d, g_ in ref.groupby("day")}
    oht["U2d"] = [((by_day[d] < v).sum() + 0.5 * (by_day[d] == v).sum()) / len(by_day[d])
                  if (np.isfinite(v) and d in by_day) else np.nan for d, v in zip(oht.day, oht.U2)]

CTX_NOPT = ["age", "pos", "n_prev_il", "days_since_act", "prev_games", "gd_season", "day_of_season"]
CTX = CTX_NOPT + PT + PT_RAW + ["z_" + n for n in PT]
BLOCK = Z18 + Z18S + ["U1", "U2"]
SETS = {"B0": CTX, "M2": CTX + BLOCK, "M1": CTX + BLOCK + P18 + P18S,
        "B0 no playing time": CTX_NOPT, "M2 no playing time": CTX_NOPT + BLOCK}


def fit(cols, X, y):
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=(20 if SMOKE else 200),
                                       l2_regularization=1.0, early_stopping=False, random_state=0,
                                       categorical_features=[cols.index("pos")])
    return m.fit(X[cols].to_numpy(float), y)


models = {k: fit(c, train, train.il14.to_numpy()) for k, c in SETS.items()}
for k, c in SETS.items():
    test["p_" + k] = models[k].predict_proba(test[c].to_numpy(float))[:, 1]
    if len(oht):
        oht["p_" + k] = models[k].predict_proba(oht[c].to_numpy(float))[:, 1]

# %% [markdown]
# ## 2026: how well does each score rank the days before an injured-list placement?
#
# Intervals: bootstrap over hitters (2,000 resamples, seed 2). The two primary tests are read at 97.5%.

# %%


def auc(y, s):
    m = np.isfinite(s)
    y, s = y[m], s[m]
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def boot(df, y, scores, diffs, seed=2, n=N_BOOT):
    idx_by = list(df.groupby("batter").indices.values())
    rng = np.random.default_rng(seed)
    res = {k: [] for k in list(scores) + [f"{a} - {b}" for a, b in diffs]}
    for _ in range(n):
        idx = np.concatenate([idx_by[i] for i in rng.integers(0, len(idx_by), len(idx_by))])
        a = {k: auc(y[idx], s[idx]) for k, s in scores.items()}
        for k in scores:
            res[k].append(a[k])
        for x, z in diffs:
            res[f"{x} - {z}"].append(a[x] - a[z])
    point = {k: auc(y, s) for k, s in scores.items()}
    for x, z in diffs:
        point[f"{x} - {z}"] = point[x] - point[z]
    out = []
    for k, v in res.items():
        v = np.array(v)
        v = v[np.isfinite(v)]
        out.append({"score": k, "AUC": point[k], "lo95": np.percentile(v, 2.5), "hi95": np.percentile(v, 97.5),
                    "lo97.5": np.percentile(v, 1.25), "hi97.5": np.percentile(v, 98.75)})
    return pd.DataFrame(out).set_index("score")


y14 = test.il14.to_numpy()
S = {"U1": test.U1d.to_numpy(), "U2": test.U2d.to_numpy(), "U3": test.U3d.to_numpy(), "U4": test.U4d.to_numpy(),
     **{k: test["p_" + k].to_numpy() for k in SETS}}
main = boot(test, y14, S, [("M2", "B0"), ("M1", "B0"), ("M2 no playing time", "B0 no playing time")])
print(main.round(4).to_string())
P1_ci = main.loc["U2", "lo97.5"] > 0.5
print(f"raw (unranked) U2 AUC {auc(y14, test.U2.to_numpy()):.4f} against ranked {main.loc['U2', 'AUC']:.4f}")

# P1 floor: rotate each hitter-season's daily-rank series in time (at least 30 game days, or half a short
# season), which keeps every hitter's level and spread and breaks only the timing; the real AUC must beat
# all of the rotations.
p1_floor = []
gix = list(test.groupby(["batter", "season"]).indices.values())
u2d = test.U2d.to_numpy()
fix_ = [ix[np.isfinite(u2d[ix])] for ix in gix]  # only defined values move, so rows and labels stay the same


def p1_rotations(vals):
    out = []
    for sd in range(1000, 1000 + N_FLOOR_P1):
        rng = np.random.default_rng(sd)
        rot = vals.copy()
        for f in fix_:
            n = len(f)
            if n >= 2:
                rot[f] = np.roll(vals[f], int(rng.integers(30, n - 30 + 1)) if n >= 61 else n // 2)
        out.append(auc(y14, rot))
    return np.array(out)


p1_floor = p1_rotations(u2d)
if FAKE:  # the floor must be beatable: a planted signal aligned with the (random) labels has to clear it
    planted = np.where(np.isfinite(u2d), np.clip(u2d + 0.3 * y14, 0, 1.3), np.nan)
    assert auc(y14, planted) > p1_rotations(planted).max(), "P1 floor cannot be beaten by a planted signal"
P1_floor = bool(np.all(main.loc["U2", "AUC"] > p1_floor))
P1 = P1_ci and P1_floor
print(f"P1 rotation floor: real {main.loc['U2', 'AUC']:.4f}, rotations max {p1_floor.max():.4f}, mean {p1_floor.mean():.4f}")
hm = test.groupby("batter").agg(rank=("U2d", "mean"), rows=("day", "size"), has_prev=("prev_games", lambda v: float(v.iloc[0] > 0)),
                                own_scale=("base_kind", lambda v: float(np.isin(v, [1, 3]).mean())))
print("label-free: spearman of a hitter's mean daily rank with rows / has previous season / share with own scale:",
      hm.corr(method="spearman").loc["rank", ["rows", "has_prev", "own_scale"]].round(3).to_dict())
print("baseline kinds on test rows (0 none, 1/2 in-season centre own/league scale, 3/4 previous centre own/league):",
      test.base_kind.value_counts(normalize=True).sort_index().round(3).to_dict())
print("sd of short-window z by season:", rows.groupby("season")[Z18S].std().median(axis=1).round(3).to_dict())
print("hitters per day with U2 (test): median", int(test.groupby("day").U2.count().median()),
      "min", int(test.groupby("day").U2.count().min()))
P2_ci = main.loc["M2 - B0", "lo97.5"] > 0

# %% [markdown]
# ### Time-shift floor for the anomaly model
#
# Twenty times, the true labels and the real B0 stay, and only the anomaly block (the 36 deviations and U1, U2) is rotated within each hitter-season by a random number of game days (at least 30, or half the season for short ones), in training and test alike; M2 is refitted. A real gain must beat all twenty; otherwise it comes from the block's structure, not from when the deviations happen.

# %%


def rotate_block(df, rng):
    out = df.copy()
    blk = out[BLOCK].to_numpy().copy()
    for _, ix in out.groupby(["batter", "season"]).indices.items():
        n = len(ix)
        if n < 2:
            continue
        k = int(rng.integers(30, n - 30 + 1)) if n >= 61 else n // 2
        blk[ix] = np.roll(blk[ix], k, axis=0)
    out[BLOCK] = blk
    return out


floor = []
auc_b0 = auc(y14, test.p_B0.to_numpy())
for sd in range(100, 100 + N_FLOOR):
    rng = np.random.default_rng(sd)
    tr_r, te_r = rotate_block(train, rng), rotate_block(test, rng)
    mm = fit(SETS["M2"], tr_r, train.il14.to_numpy())
    floor.append(auc(y14, mm.predict_proba(te_r[SETS["M2"]].to_numpy(float))[:, 1]) - auc_b0)
floor = np.array(floor)
real_gap = main.loc["M2 - B0", "AUC"]
P2_floor = bool(np.all(real_gap > floor))
print(f"real AUC(M2) - AUC(B0) = {real_gap:+.4f}; floor max {floor.max():+.4f}, mean {floor.mean():+.4f}")
print(f"P1 (U2 rank AUC > 0.5 at 97.5% and above all {N_FLOOR_P1} rotations): {'SUPPORTED' if P1 else 'NOT SUPPORTED'}")
print(f"P2 (M2 adds to B0 at 97.5% and beats all {N_FLOOR} shifted fits): {'SUPPORTED' if (P2_ci and P2_floor) else 'NOT SUPPORTED'}")

# %% [markdown]
# ### Reported alongside: 30 days, body region, after-activation days removed

# %%
test30 = rows[(rows.season == TEST) & rows.ok30].reset_index(drop=True)
test30 = test30.drop(columns=["U1", "U2"]).merge(
    test[["batter", "day", "U1d", "U2d", "U3d", "U4d"] + ["p_" + k for k in SETS]], on=["batter", "day"], how="inner")
S30 = {k: test30[k + "d"].to_numpy() for k in ("U1", "U2", "U3", "U4")} | {k: test30["p_" + k].to_numpy() for k in SETS}
r30 = boot(test30, test30.il30.to_numpy(), S30, [("M2", "B0")], n=N_BOOT)
print("IL within 30 days\n" + r30.round(4).to_string())

reg_rows = []
for region in ("upper", "lower"):
    keep = ~test.il14 | (test.next_region == region)
    t2 = test[keep.to_numpy()].reset_index(drop=True)
    rb = boot(t2, t2.il14.to_numpy(), {"U2": t2.U2d.to_numpy(), "M2": t2.p_M2.to_numpy(), "B0": t2.p_B0.to_numpy()},
              [("M2", "B0")], n=N_BOOT)
    rb["region"], rb["positives"] = region, int(t2.il14.sum())
    reg_rows.append(rb)
print(pd.concat(reg_rows).round(4).to_string())

sub = test[test.U2.notna()].reset_index(drop=True)
rs = boot(sub, sub.il14.to_numpy(), {"U2": sub.U2d.to_numpy(), "M2": sub.p_M2.to_numpy(), "B0": sub.p_B0.to_numpy()},
          [("M2", "B0")])
print("Rows where U2 is defined\n" + rs.round(4).to_string())

# Event level: each placement's pre-IL rows are one unit; a hitter's other rows are cut into 14-day blocks.
units = []
for b, g in test.groupby("batter"):
    pos_ = g[g.il14]
    for _, u in pos_.groupby(pos_.day + pd.to_timedelta(pos_.days_to_il, unit="D")):
        units.append((b, 1, u.U2d.max(), u.p_M2.max(), u.U2d.mean(), u.p_M2.mean(), len(u)))
    neg = g[~g.il14]
    for _, u in neg.groupby((neg.day - g.day.min()).dt.days // H):
        units.append((b, 0, u.U2d.max(), u.p_M2.max(), u.U2d.mean(), u.p_M2.mean(), len(u)))
units = pd.DataFrame(units, columns=["batter", "y", "U2 max", "M2 max", "U2 mean", "M2 mean", "rows"])
ue = boot(units, units.y.to_numpy(), {k: units[k].to_numpy() for k in ("U2 max", "M2 max", "U2 mean", "M2 mean")}, [])
print(f"Event level: {int(units.y.sum())} pre-IL units (median {units.rows[units.y == 1].median():.0f} rows), "
      f"{int((units.y == 0).sum())} other 14-day blocks (median {units.rows[units.y == 0].median():.0f} rows); "
      "max favours larger units, mean does not\n" + ue.round(4).to_string())

t3 = test[test.days_since_act > 30].reset_index(drop=True)
r3 = boot(t3, t3.il14.to_numpy(), {"U2": t3.U2d.to_numpy(), "M2": t3.p_M2.to_numpy(), "B0": t3.p_B0.to_numpy()},
          [("M2", "B0")], n=N_BOOT)
print("Rows more than 30 days after an activation\n" + r3.round(4).to_string())

# %% [markdown]
# ## An alarm a club could run every morning
#
# Rule fixed before the run: alarm when a hitter's U2 is in the top 1% of all hitters that day.

# %%
line = 0.99
n_day = test.groupby("day").U2.transform("count")
rank_desc = test.groupby("day").U2.rank(method="first", ascending=False)
test["alarm"] = rank_desc <= np.maximum(1, np.floor(0.01 * n_day))
alarm_days = int(test.alarm.sum())
ppv = test.loc[test.alarm, "il14"].mean() if alarm_days else np.nan
base_rate = test.il14.mean()
il_test = il[(il.effective.dt.year == TEST) & il.batter.isin(test.batter.unique()) & (il.batter != OHTANI)]
if FAKE:  # the code test must not compare anything with the real injured list
    il_test = (test[test.il14].assign(effective=lambda d: d.day + pd.to_timedelta(d.days_to_il, unit="D"))
               [["batter", "effective"]].drop_duplicates())
caught = []
for _, r in il_test.iterrows():
    w = test[(test.batter == r.batter) & (test.day >= r.effective - pd.Timedelta(days=H)) & (test.day <= r.effective)]
    caught.append(bool(w.alarm.any()) if len(w) else np.nan)
caught = pd.Series(caught, dtype=float)
per_bs = test.groupby("batter").alarm.sum()
first_day = test.groupby("batter").day.min()
pre_season = np.array([r.effective < first_day.get(r.batter, pd.Timestamp.max) for _, r in il_test.iterrows()], bool)
print(f"realised {TEST} alarm rate {test.alarm[test.U2.notna()].mean():.2%} of rows with U2 (top 1% a day, at least one); "
      f"{int((caught.isna().to_numpy() & ~pre_season).sum())} of {len(caught)} placements had no row in the {H} days before "
      f"although he had played earlier that season; {int(pre_season.sum())} came before his first game; "
      f"alarm days per hitter-season {per_bs.mean():.2f}")
print(f"alarm = top 1% of hitters that day: {alarm_days:,} alarm days in {TEST}, {per_bs.gt(0).mean():.1%} of hitters alarmed at least once "
      f"(median {per_bs[per_bs > 0].median() if (per_bs > 0).any() else 0:.0f} days among them); "
      f"{ppv:.1%} of alarm days had an IL placement within {H} days (base rate {base_rate:.1%}); "
      f"{caught.mean():.1%} of {int(caught.notna().sum())} placements had an alarm in the {H} days before")
alarms = test.loc[test.alarm, ["batter", "day", "U2", "il14"] + Z18S].copy()
alarms.insert(1, "name", alarms.batter.map(lambda b: people[b][2]))
zcols = alarms[Z18S].abs()
alarms["largest deviations"] = [", ".join(f"{c[2:]} {alarms.loc[i, c]:+.1f}" for c in zcols.loc[i].nlargest(3).index) for i in alarms.index]
alarms[["batter", "name", "day", "U2", "il14", "largest deviations"]].to_csv("alarms_2026.csv", index=False)
print(alarms[["name", "day", "U2", "il14", "largest deviations"]].head(15).to_string(index=False))

# %% [markdown]
# ## What the anomaly model leans on (2026, permutation importance)

# %%
pi = permutation_importance(models["M2"], test[SETS["M2"]].to_numpy(float), y14, scoring="roc_auc",
                            n_repeats=5, random_state=3)
imp = pd.Series(pi.importances_mean, index=SETS["M2"]).sort_values(ascending=False).head(10)
print(imp.round(4).to_string())

# %% [markdown]
# ## Figures

# %%
fig, ax = plt.subplots(figsize=(8, 7))
for k, c in (("B0", "#8a8a8a"), ("M2", "#c0392b"), ("U2", "#2471a3")):
    s = S[k]
    m = np.isfinite(s)
    fpr, tpr, _ = roc_curve(y14[m], s[m])
    ax.plot(fpr, tpr, color=c, lw=2.5, label=f"{k}  AUC {main.loc[k, 'AUC']:.3f}")
ax.plot([0, 1], [0, 1], color="#cccccc", ls="--")
ax.set_xlabel("False alarm rate")
ax.set_ylabel("Share of pre-IL days flagged")
ax.set_title(f"{TEST}: injured list within {H} days")
ax.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig("roc_2026.png", dpi=120)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp.index[::-1].str.replace("z_", "dev: "), imp.values[::-1], color="#c0392b")
ax.set_xlabel("AUC lost when shuffled")
ax.set_title("What the anomaly model uses")
plt.tight_layout()
plt.savefig("importance_2026.png", dpi=120)
plt.show()

if len(oht):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(oht.day, oht.U2d, color="#2471a3", lw=2)
    ax.axhline(line, color="#c0392b", ls="--", lw=1.5, label="alarm line (top 1% that day)")
    for e in il[(il.batter == OHTANI) & (il.effective.dt.year == TEST)].effective:
        ax.axvline(e, color="#333333", lw=2, label="injured list")
    ax.set_ylabel("Rank of U2 among hitters that day")
    ax.set_title(f"Ohtani {TEST}: distance from his own normal")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("ohtani_u2_2026.png", dpi=120)
    plt.show()

summary = pd.DataFrame({"test": ["P1 unsupervised U2", "P2 anomaly model over rival"],
                        "statistic": [main.loc["U2", "AUC"], real_gap],
                        "lo97.5": [main.loc["U2", "lo97.5"], main.loc["M2 - B0", "lo97.5"]],
                        "verdict": ["SUPPORTED" if P1 else "NOT SUPPORTED",
                                    "SUPPORTED" if (P2_ci and P2_floor) else "NOT SUPPORTED"]})
if FAKE:
    print("FAKE LABELS: code test only, no verdict written")
else:
    summary.to_csv("verdict.csv", index=False)
main.to_csv("auc_2026.csv")
print(summary.to_string(index=False))
