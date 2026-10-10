# %% [markdown]
# # Does a Triple-A Hitter Drift From His Own Normal Before the Injured List?
#
# [Do Hitters Drift From Normal Before the IL?](https://www.kaggle.com/code/yasunorim/do-hitters-drift-from-normal-before-the-il) tested this on MLB 2026 and found nothing: the anomaly score ranked pre-IL days no better than chance (AUC 0.49) and an anomaly model added +0.007 AUC over a rival model. One MLB season can only see fairly large effects. This notebook repeats the test on **Triple-A 2024-2026**, whose injured-list moves had never been compared with any feature, and adds one new causal feature.
#
# - **11 parameters** per hitter and day (Triple-A has no bat tracking): batted balls, plate discipline and plate-appearance results, each against **the same hitter's own normal**.
# - **P1**: the frozen unsupervised score (U2, mean squared deviation) on all three seasons, no training.
# - **P2**: an anomaly model over a rival that knows age, position, injury history, playing time and call-ups (trained on 2024-2025, scored on 2026).
# - **P3**: does a hitter's **season-to-date** deviation add to the daily deviations?
#
# Everything was fixed before any feature was compared with a Triple-A injured-list move: [FREEZE_aaa.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_aaa.md). Pitches come from the [fetch notebook](https://www.kaggle.com/code/yasunorim/triple-a-statcast-2023-2026-fetch) (Baseball Savant minors search); injured-list moves, positions, birth dates and MLB game logs from MLB StatsAPI (internet on).

# %%
import glob
import hashlib
import os
import re
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
LEAGUE_SEASON, ROW_SEASONS, TRAIN, TEST = 2023, [2024, 2025, 2026], [2024, 2025], 2026
SEASONS = [LEAGUE_SEASON] + ROW_SEASONS
# SMOKE shrinks windows and resamples for a code test on a sample of hitters; SMOKE_FAKE_LABELS replaces the
# injured list with random placements so every path runs without reading the real outcome. Neither is set in
# the real run, and with fake labels no verdict file is written.
SMOKE = os.environ.get("SMOKE") == "1"
FAKE = os.environ.get("SMOKE_FAKE_LABELS") == "1"
# Window sizes (balls in play, pitches seen, chase-zone pitches, swings for whiff, PA), as FREEZE_anomaly.md.
LONG = (4, 12, 6, 8, 6) if SMOKE else (40, 150, 100, 150, 100)
SHORT = (2, 6, 3, 4, 3) if SMOKE else (15, 50, 30, 50, 30)
W_GAME = 2 if SMOKE else 10
BASE_LAG, MIN_BASE, MIN_SCALE, MIN_Z, MIN_D = (0, 2, 3, 3, 2) if SMOKE else (30, 20, 60, 5, 10)
N_BOOT, N_FLOOR, N_FLOOR_P1 = (30, 4, 20) if SMOKE else (2000, 20, 200)
H, H30 = 14, 30
LO, HI = 100 * 0.05 / 3 / 2, 100 - 100 * 0.05 / 3 / 2  # Bonferroni over three primaries: 0.833% and 99.167%

COLS = ["batter", "game_date", "game_pk", "at_bat_number", "pitch_number", "type", "description", "events",
        "zone", "stand", "hc_x", "hc_y", "launch_speed", "launch_angle", "estimated_woba_using_speedangle",
        "woba_value", "woba_denom", "game_type"]
FETCH_MD5 = {2023: "a81f27d0a67211923e337214bbdaeb5c", 2024: "73d3e7081e04b34fc1bdbc6cd18e872e",
             2025: "093dc465d9b7eae8ef29dce8f66685b4", 2026: "fc32bbaf9bc9476d14416c5463760a91"}
frames = []
for y in SEASONS:
    p = glob.glob(os.path.join(INPUT_ROOT, "**", f"aaa_statcast_{y}.parquet"), recursive=True)
    if not p:
        raise FileNotFoundError(f"Attach the output of yasunorim/triple-a-statcast-2023-2026-fetch (aaa_statcast_{y}.parquet).")
    md5 = hashlib.md5(open(p[0], "rb").read()).hexdigest()
    print(f"aaa_statcast_{y}.parquet md5 {md5}")
    assert FAKE or md5 == FETCH_MD5[y], "not the fetch output recorded in FREEZE_aaa.md"
    d = pd.read_parquet(p[0], columns=COLS)
    d["season"] = y
    frames.append(d)
pitches = pd.concat(frames, ignore_index=True)
del frames
pitches["game_date"] = pd.to_datetime(pitches["game_date"])
pitches = pitches.sort_values(["batter", "game_date", "game_pk", "at_bat_number", "pitch_number"], kind="stable")
assert not pitches.duplicated(["game_pk", "at_bat_number", "pitch_number"]).any(), "duplicate pitches"
cov = pd.read_csv(glob.glob(os.path.join(INPUT_ROOT, "**", "aaa_coverage.csv"), recursive=True)[0])
print(cov.to_string(index=False))
assert set(cov.season) == set(SEASONS) and (cov["games not in the schedule"] == 0).all() and (cov.share >= 0.97).all()
season_last = pitches.groupby("season").game_date.max()
season_first = pitches.groupby("season").game_date.min()
print(f"{len(pitches):,} Triple-A pitches, {pitches.batter.nunique():,} batters, seasons {SEASONS}")

# Study gates (label-free): fill rates per season, and the season end (a truncated file would move the censoring line).
bip_ = pitches.type.eq("X")
fill = pd.DataFrame({
    **{c: pitches.groupby("season")[c].apply(lambda v: v.notna().mean()) for c in ("zone", "description", "stand", "woba_denom")},
    **{c + " (in play)": pitches[bip_].groupby("season")[c].apply(lambda v: v.notna().mean())
       for c in ("launch_speed", "launch_angle", "hc_x", "hc_y")}})
print(fill.round(4).to_string())
assert (pitches.game_type == "R").all()
if not SMOKE:
    assert (fill[["zone", "description", "stand"]] >= 0.99).all().all(), "zone/description/stand fill"
    assert fill.woba_denom.between(0.24, 0.27).all(), "woba_denom share"
    assert (fill.filter(like="(in play)") >= 0.97).all().all(), "batted-ball fill"
    for y, d in season_last.items():
        assert pd.Timestamp(y, 9, 15) <= d <= pd.Timestamp(y, 10, 6), (y, d)

# %% [markdown]
# ## MLB StatsAPI: teams, transactions, people, MLB game logs

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


HIST_YEARS = list(range(2022, TEST + 1))
LEVEL = {1: "mlb", 11: "aaa", 12: "aa"}
team_level = {}
for y in HIST_YEARS:
    for sp in LEVEL:
        for t in get("/teams", sportId=sp, season=y)["teams"]:
            team_level[(y, t["id"])] = LEVEL[sp]

PLACED = re.compile(r"\bplaced\b.*\bon the\b.*injured list", re.I)
RETURNED = re.compile(r"\b(activated|reinstated|returned)\b.*\bfrom the\b.*injured list", re.I)
REHAB = re.compile(r"on a rehab assignment", re.I)
END_TYPES = {"Released", "Outrighted", "Designated for Assignment", "Optioned"}  # an injured player cannot be optioned
seen_ids, tx_rows = set(), []
for y in HIST_YEARS:
    for m in range(1, 13):
        s = pd.Timestamp(y, m, 1)
        e = s + pd.offsets.MonthEnd(0)
        for sp in LEVEL:
            for t in get("/transactions", startDate=f"{s:%Y-%m-%d}", endDate=f"{e:%Y-%m-%d}", sportId=sp)["transactions"]:
                if t.get("id") in seen_ids or not t.get("person"):
                    continue
                seen_ids.add(t.get("id"))
                d = t.get("description", "")
                kind = ("placed" if PLACED.search(d) else "returned" if RETURNED.search(d) else
                        "rehab" if REHAB.search(d) else "end" if t.get("typeDesc") in END_TYPES else None)
                if kind:
                    club = (t.get("toTeam") or t.get("fromTeam") or {}).get("id")
                    eff = pd.Timestamp(str(t.get("effectiveDate") or t.get("date"))[:10])
                    tx_rows.append({"batter": t["person"]["id"], "kind": kind, "description": d, "effective": eff,
                                    "level": team_level.get((eff.year, club)),
                                    "from_level": team_level.get((eff.year, (t.get("fromTeam") or {}).get("id")))})
            time.sleep(0.3)
tx = pd.DataFrame(tx_rows)
print(tx.groupby(["kind", "level"], dropna=False).size().to_string())


def merged(df):
    """Placements of the same player less than 10 days apart are one stint (as in the MLB notebook)."""
    df = df.drop_duplicates(["batter", "effective"]).sort_values(["batter", "effective"])
    return df[~(df.batter.eq(df.batter.shift()) & (df.effective - df.effective.shift()).dt.days.lt(10))].reset_index(drop=True)


placed = tx[tx.kind == "placed"]
il_lab = merged(placed[placed.level.isin(["mlb", "aaa"])])  # labels: MLB and Triple-A clubs only
il_hist = merged(placed[placed.level.isin(["mlb", "aaa", "aa"])])  # history for the rival model
returns = tx[(tx.kind == "returned") & tx.level.isin(["mlb", "aaa", "aa"])].drop_duplicates(["batter", "effective"])
UPPER = ["shoulder", "elbow", "wrist", "hand", "finger", "thumb", "biceps", "triceps", "forearm", "back",
         "oblique", "rib", "intercostal", "neck"]
LOWER = ["hamstring", "quad", "calf", "knee", "ankle", "foot", "hip", "groin", "toe", "heel", "achilles"]
last_ = il_lab.description.str.strip().str.rstrip(".").str.split(". ", regex=False).str[-1].str.lower()
injury = last_.where(~last_.str.contains("injured list"), "")


def has(words):
    pat = r"\b(?:" + "|".join(w.rstrip("s") for w in words) + r")"
    return injury.str.contains(pat, regex=True)


up_, lo_ = has(UPPER), has(LOWER)
il_lab["region"] = np.select([up_ & ~lo_, lo_ & ~up_], ["upper", "lower"], "other")
il_lab["text"] = injury.ne("").to_numpy()
il_lab["list"] = il_lab.description.str.extract(r"(\d+-day|full-season)", expand=False).fillna("other")
# Fixed parse check (chosen while sizing, before any feature): a Triple-A 7-day placement must be found as Triple-A.
chk = il_lab[il_lab.description.str.contains("Jason Delay", regex=False) & (il_lab.effective == "2026-08-22")]
assert chk.level.tolist() == ["aaa"], chk
print("label placements by level and list:\n" + il_lab.groupby(["level", "list"]).size().to_string())
print(f"label placements {len(il_lab)} ({il_lab.level.value_counts().to_dict()}); injury text on {il_lab.text.mean():.1%}; "
      f"regions {il_lab.region.value_counts().to_dict()}; history placements {len(il_hist)}; returns {len(returns)}")

ids = pitches.batter.unique()
people = {}
for i in range(0, len(ids), 150):
    for p in get("/people", personIds=",".join(map(str, ids[i:i + 150])))["people"]:
        people[p["id"]] = (p.get("primaryPosition", {}).get("abbreviation"), p.get("birthDate"), p.get("fullName"))
assert len(people) == len(ids), "every batter needs a person record"
pitchers = {b for b, v in people.items() if v[0] == "P"}


def season_games(js, sport):
    out = {}
    for p in js["people"]:
        sp = [x for s_ in p.get("stats", []) for x in s_.get("splits", []) if x.get("sport", {}).get("id", sport) == sport]
        tot = [x for x in sp if "team" not in x]
        out[p["id"]] = float(sum(x["stat"].get("gamesPlayed", 0) for x in (tot or sp)))
    return out


prev_games, mlb_days = {}, {b: [] for b in ids}
for i in range(0, len(ids), 150):
    chunk = ",".join(map(str, ids[i:i + 150]))
    for y in ROW_SEASONS:
        g1 = season_games(get("/people", personIds=chunk, hydrate=f"stats(group=[hitting],type=[season],season={y - 1},sportId=1)"), 1)
        g11 = season_games(get("/people", personIds=chunk, hydrate=f"stats(group=[hitting],type=[season],season={y - 1},sportId=11)"), 11)
        for b in ids[i:i + 150]:
            prev_games[(b, y)] = g1.get(b, 0.0) + g11.get(b, 0.0)
    for y in SEASONS:
        js = get("/people", personIds=chunk, hydrate=f"stats(group=[hitting],type=[gameLog],season={y},sportId=1)")
        for p in js["people"]:
            mlb_days[p["id"]] += [x["date"] for s_ in p.get("stats", []) for x in s_.get("splits", [])
                                  if x.get("sport", {}).get("id", 1) == 1 and x.get("date")]
    time.sleep(0.3)
mlb_days = {b: np.unique(np.array(v, dtype="datetime64[D]")).astype("datetime64[ns]") for b, v in mlb_days.items()}
assert len(prev_games) == len(ids) * len(ROW_SEASONS)
POS_GROUP = {"C": 0, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1, "LF": 2, "CF": 2, "RF": 2, "OF": 2, "DH": 3, "TWP": 3}
print(f"{len(pitchers)} batters with primary position P are left out; "
      f"{sum(len(v) > 0 for v in mlb_days.values()):,} batters have MLB game days 2023-{TEST}")

# %% [markdown]
# ## Rehab windows: pitches of a player on an MLB club's injured list are removed

# %%
# Windows start at an MLB club's placement or an MLB club's rehab assignment (a minor-league club's own rehab
# assignment to a lower level is not one), and end at a return at MLB or Triple-A, a release, outright, option
# or designation, or the season end.
mlb_rehab = (tx.kind == "rehab") & (tx.from_level == "mlb")
print(f"rehab assignments: {int(mlb_rehab.sum()):,} by MLB clubs used, {int(((tx.kind == 'rehab') & ~mlb_rehab).sum()):,} by other clubs not used")
starts = tx[((tx.kind == "placed") & (tx.level == "mlb")) | mlb_rehab]
starts = starts[starts.batter.isin(pitches.batter.unique())]
ends_ = tx[((tx.kind == "returned") & tx.level.isin(["mlb", "aaa"])) | (tx.kind == "end")]
ends_by = {b: np.sort(g.effective.to_numpy()) for b, g in ends_.groupby("batter")}
rehab_seasons = set(zip(tx.loc[mlb_rehab, "batter"], tx.loc[mlb_rehab, "effective"].dt.year))
windows = []
for b, g in starts.groupby("batter"):
    en = ends_by.get(b, np.array([], dtype="datetime64[ns]"))
    for s0 in np.sort(g.effective.to_numpy()):
        y = pd.Timestamp(s0).year
        stop = np.datetime64(pd.Timestamp(y, 12, 31) + pd.Timedelta(days=1))
        if y in season_last.index:
            stop = min(stop, np.datetime64(season_last[y] + pd.Timedelta(days=1)))
        later = en[en > s0]
        if len(later):
            stop = min(stop, later[0])
        windows.append((b, s0, stop))
windows = pd.DataFrame(windows, columns=["batter", "start", "stop"])
in_rehab = np.zeros(len(pitches), bool)
pb, pdte = pitches.batter.to_numpy(), pitches.game_date.to_numpy()
for b, g in windows.groupby("batter"):
    m = pb == b
    for s0, s1 in zip(g.start, g.stop):
        in_rehab[m] |= (pdte[m] >= np.datetime64(s0)) & (pdte[m] < np.datetime64(s1))
rem = pd.DataFrame({"batter": pb, "season": pitches.season.to_numpy(), "rehab": in_rehab}).groupby(["batter", "season"]).rehab.agg(["mean", "sum"])
lost = rem[(rem["mean"] > 0.5) & ~pd.Series([k in rehab_seasons for k in rem.index], index=rem.index)]
print(f"rehab windows {len(windows):,}; Triple-A pitches removed {in_rehab.sum():,} ({in_rehab.mean():.2%}) "
      f"in {int((rem['sum'] > 0).sum()):,} hitter-seasons; hitter-seasons losing >50% without a rehab assignment: {len(lost)}")
assert len(lost) == 0, lost.head(20)
pitches = pitches[~in_rehab]

# %% [markdown]
# ## Eleven parameters, day by day (only earlier Triple-A pitches)

# %%
SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt",
         "missed_bunt", "bunt_foul_tip"}
MISS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
BASE11 = ["EV90", "Launch angle", "Sweet spot", "Pull", "xwOBAcon", "Swing rate", "Chase", "Whiff", "K", "BB", "wOBA"]
RESULTS = ["EV90", "xwOBAcon", "K", "BB", "wOBA"]
P11 = BASE11
P11S = [n + " (short)" for n in BASE11]
PT = ["Days between games", "PA per game"]
PT_RAW = ["Rest gap", "Early exit"]
ZPARAMS = P11 + P11S + PT
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
    bip = g[(g.type == "X").to_numpy() & g.launch_speed.notna().to_numpy()]
    bd = bip.game_date.to_numpy()
    la = bip.launch_angle.to_numpy(float)
    sweet = np.where(np.isnan(la), np.nan, ((la >= 8) & (la <= 32)).astype(float))
    ang = np.degrees(np.arctan2(bip.hc_x.to_numpy(float) - 125.42, 198.27 - bip.hc_y.to_numpy(float)))
    pull = np.where(np.isnan(ang), np.nan, np.where(bip.stand.to_numpy() == "R", ang < -15, ang > 15).astype(float))
    ch = g[g.zone.isin([11, 12, 13, 14]).to_numpy()]
    pa = g[(g.woba_denom.gt(0) & g.woba_value.notna()).to_numpy()]
    pd_ = pa.game_date.to_numpy()
    for sfx, (w_bip, w_seen, w_chase, w_whiff, w_pa) in (("", LONG), (" (short)", SHORT)):
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
    n = len(days)
    gap, pag, rest, early = (np.full(n, np.nan) for _ in range(4))
    pa_day = pd.Series(g.events.notna().to_numpy(int), index=gd).groupby(level=0).sum().reindex(days).to_numpy(float)
    for i in range(1, n):
        rest[i] = (days[i] - days[i - 1]) / np.timedelta64(1, "D")
        if i - 1 >= 3:
            early[i] = pa_day[i - 1] - np.median(pa_day[:i - 1])
        if i >= W_GAME:
            gap[i] = (days[i] - days[i - W_GAME]) / np.timedelta64(1, "D") / W_GAME
            pag[i] = pa_day[i - W_GAME:i].mean()
    out["Days between games"], out["PA per game"], out["Rest gap"], out["Early exit"] = gap, pag, rest, early
    return days, out


print(f"PA pitches without woba_value (left out of K/BB/wOBA): {int((pitches.woba_denom.gt(0) & pitches.woba_value.isna()).sum())}")
groups = {k: g for k, g in pitches[~pitches.batter.isin(pitchers)].groupby(["batter", "season"], sort=False)}
series = {k: params_for(g) for k, g in groups.items()}
print(f"{len(series):,} batter-seasons")

# Planted check: no day may use its own or later pitches.
k0 = max(series, key=lambda k: len(series[k][0]))
days0, p0 = series[k0]
t0 = days0[len(days0) // 2]
g0 = groups[k0].copy()
late = (g0.game_date >= t0).to_numpy()
for c in ["launch_speed", "launch_angle", "hc_x", "hc_y", "estimated_woba_using_speedangle", "woba_value"]:
    g0.loc[late, c] = 999.0
g0.loc[late, "zone"] = 12.0
g0.loc[late, "description"] = "swinging_strike"
g0.loc[late, "events"] = "strikeout"
_, p0b = params_for(g0)
i0 = list(days0).index(t0)
for name in PARAMS:
    assert np.array_equal(p0[name][: i0 + 1], p0b[name][: i0 + 1], equal_nan=True), name
assert any(not np.array_equal(p0[n][i0 + 1:], p0b[n][i0 + 1:], equal_nan=True) for n in P11S)

# %% [markdown]
# ## Each parameter against the same hitter's own normal

# %%


def mad(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def own_z(days, x, prev, league_scale):
    """As FREEZE_anomaly.md. Returns z and the baseline kind per day:
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


league_scale = {}
for n in ZPARAMS:
    ms = [mad(v[np.isfinite(v)]) for (b_, y_), (_, pp) in series.items() if y_ == LEAGUE_SEASON
          for v in [pp[n]] if np.isfinite(v).sum() >= MIN_SCALE]
    league_scale[n] = float(np.median(ms)) if ms else None
print("league-typical scales (Triple-A 2023):", {k: round(v, 4) for k, v in league_scale.items() if v is not None and "(short)" in k})


def zs_for(key):
    days, p = series[key]
    prev = series.get((key[0], key[1] - 1))
    return {n: own_z(days, p[n], None if prev is None else (prev[0], prev[1][n]), league_scale[n]) for n in ZPARAMS}


# Planted checks on synthetic days, as in the MLB notebook.
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

# %% [markdown]
# ## One row per hitter and Triple-A game day (no label yet)

# %%
Z11 = ["z_" + n for n in P11]
Z11S = ["z_" + n for n in P11S]
DCOLS = ["D_" + n for n in P11S] + ["D_U2"]


def u_scores(zm):
    nz = np.isfinite(zm).sum(1)
    u1 = np.where(nz >= MIN_Z, np.nanmax(np.abs(np.where(np.isfinite(zm), zm, 0.0)), 1), np.nan)
    u2 = np.where(nz >= MIN_Z, np.nanmean(zm ** 2, 1), np.nan)
    return u1, u2


def season_to_date(df):
    """D: mean of each short z and of U2 over his earlier game days of the season (strictly before t)."""
    out = np.full((len(df), len(DCOLS)), np.nan)
    vals = df[Z11S + ["U2"]].to_numpy(float)
    for _, ix in df.groupby(["batter", "season"]).indices.items():
        v = vals[ix]
        f = np.isfinite(v)
        cs = np.cumsum(np.where(f, v, 0.0), 0)
        cn = np.cumsum(f, 0)
        prev_s = np.vstack([np.zeros((1, v.shape[1])), cs[:-1]])
        prev_n = np.vstack([np.zeros((1, v.shape[1])), cn[:-1]])
        out[ix] = np.where(prev_n >= MIN_D, prev_s / np.maximum(prev_n, 1), np.nan)
    return out


recs = []
for key, (days, p) in series.items():
    b, y = key
    if y not in ROW_SEASONS:
        continue
    zk = zs_for(key)
    r = pd.DataFrame({"batter": b, "season": y, "day": days})
    for n in PARAMS:
        r[n] = p[n]
    for n in ZPARAMS:
        r["z_" + n] = zk[n][0]
    r["prev_games"] = prev_games[(b, y)]
    r["base_kind"] = zk["Swing rate (short)"][1]
    pos, birth, _ = people[b]
    r["pos"] = POS_GROUP.get(pos, np.nan)
    r["age"] = (days - np.datetime64(birth)) / np.timedelta64(1, "D") / 365.25 if birth else np.nan
    md = mlb_days.get(b, np.array([], dtype="datetime64[ns]"))
    km = np.searchsorted(md, days, side="left")  # MLB game days strictly before t
    r["days_since_mlb"] = np.where(km > 0, np.minimum((days - md[np.maximum(km - 1, 0)]) / np.timedelta64(1, "D"), 365) if len(md) else 365, 365)
    r["mlb_days_season"] = km - np.searchsorted(md, np.datetime64(f"{y}-01-01"), side="left")
    r["gd_season"] = np.arange(len(days))
    r["day_of_season"] = (days - np.datetime64(season_first[y])) / np.timedelta64(1, "D")
    recs.append(r)
rows = pd.concat(recs, ignore_index=True).sort_values(["batter", "season", "day"], kind="stable").reset_index(drop=True)
rows["U1"], rows["U2"] = u_scores(rows[Z11S].to_numpy())
rows[DCOLS] = season_to_date(rows)
rows["end"] = rows.season.map(season_last)
rows["ok14"] = rows.day + pd.Timedelta(days=H) <= rows.end
rows["ok30"] = rows.day + pd.Timedelta(days=H30) <= rows.end
# Planted check: D on day t must not move when t's own z (and later z) change.
_gi = rows.groupby(["batter", "season"]).indices
_ix = _gi[max(_gi, key=lambda k: len(_gi[k]))]
assert len(_ix) > 3 * MIN_D
_t = rows.loc[_ix].copy()
_t.loc[_t.index[len(_ix) // 2:], Z11S] = 9.0
_t["U1"], _t["U2"] = u_scores(_t[Z11S].to_numpy())
_d = season_to_date(_t)
assert np.array_equal(_d[: len(_ix) // 2 + 1], rows.loc[_ix, DCOLS].to_numpy()[: len(_ix) // 2 + 1], equal_nan=True)
assert not np.array_equal(_d[len(_ix) // 2 + 1:], rows.loc[_ix, DCOLS].to_numpy()[len(_ix) // 2 + 1:], equal_nan=True)

# Unsupervised scores and daily ranks are computed on all rows before any label is attached.
fit_rows = rows[rows.season.isin(TRAIN) & rows.ok14 & rows.U2.notna()]


def zs(df):
    return np.nan_to_num(df[Z11S].to_numpy(), nan=0.0)


iso = IsolationForest(n_estimators=300, random_state=0).fit(zs(fit_rows))
samp = np.random.default_rng(4).choice(len(fit_rows), min(len(fit_rows), 50000), replace=False)
mcd = MinCovDet(random_state=4).fit(zs(fit_rows)[samp])
ok_ = rows.U2.notna().to_numpy()
rows["U3"], rows["U4"] = np.nan, np.nan
rows.loc[ok_, "U3"] = -iso.score_samples(zs(rows[ok_]))
rows.loc[ok_, "U4"] = np.sqrt(mcd.mahalanobis(zs(rows[ok_])))


def daily_rank(df, u):
    g_ = df.groupby("day")[u]
    return (g_.rank(method="average") - 0.5) / g_.transform("count")


for u in ("U1", "U2", "U3", "U4"):
    rows[u + "d"] = daily_rank(rows, u)

# Label-free checks, printed before any placement is joined.
print("median raw U2 by season and month:\n" + rows[rows.U2.notna()].assign(month=lambda d: d.day.dt.month)
      .groupby(["season", "month"]).U2.median().round(2).unstack().to_string())
print("U2 defined by season:", rows.groupby("season").U2.apply(lambda v: round(v.notna().mean(), 3)).to_dict(),
      "| D_U2 defined:", rows.groupby("season").D_U2.apply(lambda v: round(v.notna().mean(), 3)).to_dict())
print("baseline kinds by season (0 none, 1/2 in-season centre own/league scale, 3/4 previous centre own/league):\n"
      + rows.groupby("season").base_kind.value_counts(normalize=True).round(3).unstack().to_string())
print("sd of short-window z by season:", rows.groupby("season")[Z11S].std().median(axis=1).round(3).to_dict())
print("hitters per day with U2: median", int(rows.groupby("day").U2.count().median()), "min",
      int(rows.groupby("day").U2.count().min()))
hm = rows.groupby("batter").agg(rank=("U2d", "mean"), n=("day", "size"), has_prev=("prev_games", lambda v: float(v.iloc[0] > 0)),
                                own_scale=("base_kind", lambda v: float(np.isin(v, [1, 3]).mean())))
print("spearman of a hitter's mean daily U2 rank with rows / previous season / own-scale share:",
      hm.corr(method="spearman").loc["rank", ["n", "has_prev", "own_scale"]].round(3).to_dict())

# %% [markdown]
# ## Labels: an injured-list placement within 14 days (MLB or Triple-A club, not after an MLB game)

# %%
if FAKE:  # random placements (about 0.4 per hitter-season), unrelated to any feature
    rng0 = np.random.default_rng(99)
    fk = []
    for (b, y), g in rows.groupby(["batter", "season"]):
        k = min(rng0.poisson(0.4), len(g))
        for d in np.sort(rng0.choice(g.day.to_numpy(), k, replace=False)) if k else []:
            fk.append({"batter": b, "effective": pd.Timestamp(d) + pd.Timedelta(days=1),
                       "region": rng0.choice(["upper", "lower", "other"]), "text": bool(rng0.random() < 0.5)})
    il_lab = pd.DataFrame(fk, columns=["batter", "effective", "region", "text"])
il_by = {b: (g.effective.to_numpy(), g.region.to_numpy(), g.text.to_numpy()) for b, g in il_lab.sort_values("effective").groupby("batter")}
hist_by = {b: np.sort(g.effective.to_numpy()) for b, g in il_hist.groupby("batter")}
ret_by = {b: np.sort(g.effective.to_numpy()) for b, g in returns.groupby("batter")}
parts = []
for (b, y), ix in rows.groupby(["batter", "season"]).indices.items():
    days = rows.day.to_numpy()[ix]
    eff, reg, txt = il_by.get(b, (np.array([], dtype="datetime64[ns]"), np.array([]), np.array([], bool)))
    k = np.searchsorted(eff, days, side="left")
    has_next = k < len(eff)
    nxt = np.where(has_next, eff[np.minimum(k, max(len(eff) - 1, 0))] if len(eff) else days, days)
    dti = np.where(has_next, (nxt - days) / np.timedelta64(1, "D"), np.inf)
    md = mlb_days.get(b, np.array([], dtype="datetime64[ns]"))
    mlb_between = (np.searchsorted(md, nxt, side="right") - np.searchsorted(md, days, side="right")) > 0
    h = hist_by.get(b, np.array([], dtype="datetime64[ns]"))
    a = ret_by.get(b, np.array([], dtype="datetime64[ns]"))
    ka = np.searchsorted(a, days, side="right")  # a return on t is known on t
    parts.append(pd.DataFrame({
        "idx": ix, "days_to_il": dti, "mlb_before_il": has_next & mlb_between,
        "next_region": np.where(has_next, reg[np.minimum(k, max(len(eff) - 1, 0))] if len(eff) else "", ""),
        "next_text": np.where(has_next, txt[np.minimum(k, max(len(eff) - 1, 0))] if len(eff) else False, False),
        "n_prev_il": np.searchsorted(h, days, side="left"),
        "days_since_act": np.where(ka > 0, np.minimum((days - a[np.maximum(ka - 1, 0)]) / np.timedelta64(1, "D"), 365) if len(a) else 365, 365)}))
lab = pd.concat(parts).set_index("idx").sort_index()
rows = rows.join(lab)
rows["il14"] = rows.days_to_il <= H
rows["il30"] = rows.days_to_il <= H30
rows["drop14"] = rows.il14 & rows.mlb_before_il  # the injury may have happened in MLB play
rows["drop30"] = rows.il30 & rows.mlb_before_il
use14 = rows.ok14 & ~rows.drop14
train_m = (rows.season.isin(TRAIN) & use14).to_numpy()
test_m = ((rows.season == TEST) & use14).to_numpy()
p1_m = (rows.season.isin(ROW_SEASONS) & use14 & rows.U2.notna()).to_numpy()
train, test = rows[train_m], rows[test_m]
print(f"rows dropped by the MLB-game rule: {int((rows.drop14 & rows.ok14).sum()):,} (14 days), {int((rows.drop30 & rows.ok30).sum()):,} (30 days)")
print(f"rows {len(rows):,} of {rows.batter.nunique():,} hitters; train {len(train):,} rows ({int(train.il14.sum()):,} within {H} days), test {len(test):,} ({int(test.il14.sum()):,}); "
      f"P1 rows {int(p1_m.sum()):,} ({int(rows.il14[p1_m].sum()):,} positive)")
pre = rows[use14 & rows.il14].assign(pl=lambda d: d.day + pd.to_timedelta(d.days_to_il, unit="D"))
eff_n = pre.groupby(["batter", "pl"]).U2.apply(lambda v: v.notna().any())
eff_share = pre.groupby(["batter", "pl"]).U2.apply(lambda v: v.notna().mean())
print(f"P1 effective sample (placements x share of their pre-IL rows with U2): {eff_share.sum():.1f}")
print(f"placements with a pre-IL Triple-A row: {len(eff_n):,}; with U2 defined on at least one of them: {int(eff_n.sum()):,} "
      f"(by season {pre.assign(u=pre.U2.notna()).groupby(['season', 'batter', 'pl']).u.any().groupby('season').sum().to_dict()})")

aaa_days = {b: np.sort(g.day.to_numpy()) for b, g in rows.groupby("batter")}
ret_seen = []
for _, r in il_lab[il_lab.batter.isin(aaa_days) & il_lab.effective.dt.year.isin(ROW_SEASONS)].iterrows():
    d_ = aaa_days[r.batter]
    later_ = d_[d_ > np.datetime64(r.effective)]
    if len(later_):
        a_ = ret_by.get(r.batter, np.array([], dtype="datetime64[ns]"))
        ret_seen.append(bool(((a_ >= np.datetime64(r.effective)) & (a_ <= later_[0])).any()))
print(f"placements followed by a later Triple-A game: {len(ret_seen):,}; with a parsed return before it: "
      f"{np.mean(ret_seen) if ret_seen else float('nan'):.1%}")

# %% [markdown]
# ## Models (trained on Triple-A 2024-2025)

# %%
CTX_NOPT = ["age", "pos", "n_prev_il", "days_since_act", "prev_games", "gd_season", "day_of_season", "days_since_mlb", "mlb_days_season"]
CTX = CTX_NOPT + PT + PT_RAW + ["z_" + n for n in PT]
BLOCK = Z11 + Z11S + ["U1", "U2"]
SETS = {"B0": CTX, "M2": CTX + BLOCK, "M3": CTX + BLOCK + DCOLS}


def fit(cols, X, y):
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=(20 if SMOKE else 200),
                                       l2_regularization=1.0, early_stopping=False, random_state=0,
                                       categorical_features=[cols.index("pos")])
    return m.fit(X[cols].to_numpy(float), y)


def predict(m, cols, X):
    return m.predict_proba(X[cols].to_numpy(float))[:, 1]


y_tr = train.il14.to_numpy()
models = {k: fit(c, train, y_tr) for k, c in SETS.items()}
for k, c in SETS.items():
    rows["p_" + k] = np.nan
    rows.loc[test_m, "p_" + k] = predict(models[k], c, test)
test = rows[test_m]
y14 = test.il14.to_numpy()

# %% [markdown]
# ## Results
#
# Intervals: bootstrap over hitters (2,000 resamples, seed 2); the three primaries are read at 0.833% / 99.167%.

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
                    "lo_bonf": np.percentile(v, LO), "hi_bonf": np.percentile(v, HI)})
    return pd.DataFrame(out).set_index("score")


# --- P1: U2 daily rank, 2024-2026 pooled ---
p1df = rows[p1_m]
y_p1 = p1df.il14.to_numpy()
u2d_p1 = p1df.U2d.to_numpy()
r_p1 = boot(p1df, y_p1, {"U2": u2d_p1}, [])
gix_p1 = list(p1df.groupby(["batter", "season"]).indices.values())


def p1_rotated(vals, sd):
    rng = np.random.default_rng(sd)
    rot = vals.copy()
    for f in gix_p1:
        n = len(f)
        if n >= 2:
            rot[f] = np.roll(vals[f], int(rng.integers(30, n - 30 + 1)) if n >= 61 else n // 2)
    return rot


rots = [p1_rotated(u2d_p1, sd) for sd in range(1000, 1000 + N_FLOOR_P1)]
p1_floor = np.array([auc(y_p1, r) for r in rots])
if FAKE:
    planted = np.clip(u2d_p1 + 0.3 * y_p1, 0, 1.3)
    assert auc(y_p1, planted) > max(auc(y_p1, p1_rotated(planted, sd)) for sd in range(1000, 1000 + N_FLOOR_P1)), "P1 floor cannot be beaten"
P1 = bool(r_p1.loc["U2", "lo_bonf"] > 0.5 and np.all(r_p1.loc["U2", "AUC"] > p1_floor))
print(f"P1: U2 daily-rank AUC {r_p1.loc['U2', 'AUC']:.4f} [{r_p1.loc['U2', 'lo_bonf']:.4f}, {r_p1.loc['U2', 'hi_bonf']:.4f}]; "
      f"rotations mean {p1_floor.mean():.4f}, max {p1_floor.max():.4f} -> {'SUPPORTED' if P1 else 'NOT SUPPORTED'}")
seas = p1df.season.to_numpy()
by_season = []
for y in ROW_SEASONS:
    m = seas == y
    fl = np.array([auc(y_p1[m], r[m]) for r in rots])
    by_season.append({"season": y, "AUC": auc(y_p1[m], u2d_p1[m]), "floor mean": fl.mean(), "floor max": fl.max(),
                      "positives": int(y_p1[m].sum())})
print("P1 by season\n" + pd.DataFrame(by_season).round(4).to_string(index=False))
bk = p1df.base_kind.to_numpy()
print("P1 by baseline kind:", {int(k): round(auc(y_p1[bk == k], u2d_p1[bk == k]), 4) for k in np.unique(bk)})

# --- P2 and P3 on 2026 ---
main = boot(test, y14, {k: test["p_" + k].to_numpy() for k in SETS} | {u: test[u + "d"].to_numpy() for u in ("U1", "U2", "U3", "U4")},
            [("M2", "B0"), ("M3", "M2"), ("M3", "B0")])
print(main.round(4).to_string())


def k_shift(n, rng):
    return int(rng.integers(30, n - 30 + 1)) if n >= 61 else n // 2


def rotate_z(df, rng, cols=Z11 + Z11S):
    """One shift per hitter-season; each column's defined values roll by it (or by half when there is no room);
    missing values stay on their dates. U1/U2 and D are recomputed from the rotated z."""
    out = df.copy()
    z = out[cols].to_numpy().copy()
    small, moved = 0, 0
    for _, ix in out.groupby(["batter", "season"]).indices.items():
        k = k_shift(len(ix), rng) if len(ix) >= 2 else 0
        for c in range(z.shape[1]):
            f = ix[np.isfinite(z[ix, c])]
            nc = len(f)
            if nc < 2:
                continue
            kc = k if k <= nc - 30 else nc // 2
            z[f, c] = np.roll(z[f, c], kc)
            moved += nc
            small += nc * (min(kc, nc - kc) < 15)
    out[cols] = z
    out["U1"], out["U2"] = u_scores(out[Z11S].to_numpy())
    out[DCOLS] = season_to_date(out)
    return out, small / max(moved, 1)


def swap_d(df, rng):
    """Within each season, hitter-seasons get another hitter-season's D (derangement); the recipient keeps his mask."""
    out = df.copy()
    dv = out[DCOLS].to_numpy().copy()
    src = df[DCOLS].to_numpy()
    empty = 0
    for y, gy in out.groupby("season"):
        keys = [(b_, gy.index.to_numpy()[ix]) for b_, ix in gy.groupby("batter").indices.items()]
        keys = [(b_, ix) for b_, ix in keys if np.isfinite(src[ix, -1]).any()]  # hitter-seasons with any D
        n = len(keys)
        perm = rng.permutation(n)
        while n > 1 and np.any(perm == np.arange(n)):
            perm = rng.permutation(n)
        for i_r, (b_r, ix_r) in enumerate(keys):
            for c in range(len(DCOLS)):
                rf = ix_r[np.isfinite(src[ix_r, c])]
                if len(rf) == 0:
                    continue
                j_, steps = perm[i_r], 0
                dvals = src[keys[j_][1], c][np.isfinite(src[keys[j_][1], c])]
                while len(dvals) == 0 and steps < n:  # this donor has none in the column: next along the cycle
                    j_, steps = perm[j_], steps + 1
                    if j_ != i_r:
                        dvals = src[keys[j_][1], c][np.isfinite(src[keys[j_][1], c])]
                if len(dvals) == 0:
                    empty += 1
                    continue
                dv[rf, c] = np.resize(dvals, len(rf))
    out[DCOLS] = dv
    return out, empty


def refit_gap(rows_x, set_a, set_b_auc):
    m = fit(SETS[set_a], rows_x[train_m], y_tr)
    return auc(y14, predict(m, SETS[set_a], rows_x[test_m])) - set_b_auc


auc_b0, auc_m2 = main.loc["B0", "AUC"], main.loc["M2", "AUC"]
p2_floor, p3_floor, m3rot, smalls, empties = [], [], [], [], []
for i in range(N_FLOOR):
    rz, sm = rotate_z(rows, np.random.default_rng(100 + i))
    smalls.append(sm)
    p2_floor.append(refit_gap(rz, "M2", auc_b0))
    m2r = fit(SETS["M2"], rz[train_m], y_tr)
    m3r = fit(SETS["M3"], rz[train_m], y_tr)
    m3rot.append(auc(y14, predict(m3r, SETS["M3"], rz[test_m])) - auc(y14, predict(m2r, SETS["M2"], rz[test_m])))
    sw, em = swap_d(rows, np.random.default_rng(200 + i))
    empties.append(em)
    p3_floor.append(refit_gap(sw, "M3", auc_m2))
p2_floor, p3_floor, m3rot = map(np.array, (p2_floor, p3_floor, m3rot))
assert sum(empties) == 0, "a swap recipient kept his own D"
print(f"P2 floor: share of defined z moved by fewer than 15 positions {np.mean(smalls):.3f}; "
      f"P3 swap: columns left empty by a donor {np.mean(empties):.1f} per floor")
gap2, gap3 = main.loc["M2 - B0", "AUC"], main.loc["M3 - M2", "AUC"]
P2 = bool(main.loc["M2 - B0", "lo_bonf"] > 0 and np.all(gap2 > p2_floor))
P3 = bool(main.loc["M3 - M2", "lo_bonf"] > 0 and np.all(gap3 > p3_floor))
print(f"P2: AUC(M2) - AUC(B0) {gap2:+.4f} [{main.loc['M2 - B0', 'lo_bonf']:+.4f}, {main.loc['M2 - B0', 'hi_bonf']:+.4f}]; "
      f"floor mean {p2_floor.mean():+.4f}, max {p2_floor.max():+.4f} -> {'SUPPORTED' if P2 else 'NOT SUPPORTED'}")
print(f"P3: AUC(M3) - AUC(M2) {gap3:+.4f} [{main.loc['M3 - M2', 'lo_bonf']:+.4f}, {main.loc['M3 - M2', 'hi_bonf']:+.4f}]; "
      f"swap floor mean {p3_floor.mean():+.4f}, max {p3_floor.max():+.4f} -> {'SUPPORTED' if P3 else 'NOT SUPPORTED'}")
print(f"reported: M3 - M2 under rotation of z with D recomputed: mean {m3rot.mean():+.4f}, max {m3rot.max():+.4f}")

if FAKE:  # the P2 and P3 floors must be beatable by timing / a D trait, and not by a hitter-season trait
    rate = rows.groupby(["batter", "season"]).il14.transform("mean").to_numpy()
    c0 = Z11S.index("z_wOBA (short)")

    def planted_gap(rows_p, set_a, set_b):
        ma, mb = fit(SETS[set_a], rows_p[train_m], y_tr), fit(SETS[set_b], rows_p[train_m], y_tr)
        return auc(y14, predict(ma, SETS[set_a], rows_p[test_m])) - auc(y14, predict(mb, SETS[set_b], rows_p[test_m]))

    for kind, val in (("timing", np.where(rows.il14, 4.0, 0.0)), ("trait", 8.0 * rate)):
        rp = rows.copy()
        col = Z11S[c0]
        rp[col] = np.where(np.isfinite(rp[col]), val, np.nan)
        rp["U1"], rp["U2"] = u_scores(rp[Z11S].to_numpy())
        rp[DCOLS] = season_to_date(rp)
        real = planted_gap(rp, "M2", "B0")
        fl = [planted_gap(rotate_z(rp, np.random.default_rng(100 + i))[0], "M2", "B0") for i in range(N_FLOOR)]
        print(f"planted {kind}: real {real:+.4f}, floors mean {np.mean(fl):+.4f}, max {max(fl):+.4f}")
        if kind == "timing":
            assert real > max(fl), "P2 floor cannot be beaten by planted timing"
        else:  # the floor keeps a hitter-season trait: the floors must carry most of the trait's gain
            assert np.mean(fl) >= 0.8 * real, "P2 floor does not keep a planted trait"
    rp = rows.copy()
    rp["D_U2"] = np.where(np.isfinite(rp["D_U2"]), 8.0 * rate, np.nan)
    real = planted_gap(rp, "M3", "M2")
    fl = [planted_gap(swap_d(rp, np.random.default_rng(200 + i))[0], "M3", "M2") for i in range(N_FLOOR)]
    print(f"planted D trait: real {real:+.4f}, swap floors max {max(fl):+.4f}")
    assert real > max(fl), "P3 floor cannot be beaten by a planted D trait"

# %% [markdown]
# ### Interpretation rules (fixed in the plan): injury text, returns and demotions

# %%


def subset_report(mask, name):
    sub = rows[mask & test_m]
    sp1 = rows[mask & p1_m]
    r1 = auc(sp1.il14.to_numpy(), sp1.U2d.to_numpy())
    ys = sub.il14.to_numpy()
    g2 = auc(ys, sub.p_M2.to_numpy()) - auc(ys, sub.p_B0.to_numpy())
    g3 = auc(ys, sub.p_M3.to_numpy()) - auc(ys, sub.p_M2.to_numpy())
    return {"subset": name, "P1 AUC": r1, "P1 positives": int(sp1.il14.sum()), "M2 - B0": g2, "M3 - M2": g3,
            "2026 positives": int(sub.il14.sum())}


masks = {"injury-text placements only": (~rows.il14 | rows.next_text).to_numpy(),
         "more than 30 days after a return": (rows.days_since_act > 30).to_numpy(),
         "more than 30 days after his last MLB game": (rows.days_since_mlb > 30).to_numpy()}
interp = pd.DataFrame([subset_report(m, k) for k, m in masks.items()])
print(interp.round(4).to_string(index=False))
floor_mean = {"P1 AUC": p1_floor.mean(), "M2 - B0": p2_floor.mean(), "M3 - M2": p3_floor.mean()}
verdict_note = {}
for prim, col, ok in (("P1", "P1 AUC", P1), ("P2", "M2 - B0", P2), ("P3", "M3 - M2", P3)):
    if not ok:
        continue
    notes = []
    if interp.loc[0, col] <= floor_mean[col]:
        notes.append("not attributable to injury")
    if (interp.loc[1:, col] <= floor_mean[col]).any():
        notes.append("timing of a return or demotion")
    if notes:
        verdict_note[prim] = "; ".join(notes)
print("interpretation notes:", verdict_note or "none")

# %% [markdown]
# ### Reported alongside: 30 days, body region, event level, alarms

# %%
use30 = (rows.ok30 & ~rows.drop30 & (rows.season == TEST)).to_numpy()
t30 = rows[use30]
S30 = {u: t30[u + "d"].to_numpy() for u in ("U1", "U2", "U3", "U4")} | {k: t30["p_" + k].to_numpy() for k in SETS}
print("IL within 30 days\n" + boot(t30, t30.il30.to_numpy(), S30, [("M2", "B0"), ("M3", "M2")]).round(4).to_string())

for region in ("upper", "lower"):
    keep = (~test.il14 | (test.next_region == region)).to_numpy()
    t2 = test[keep]
    npos = int(t2.il14.sum())
    if npos >= 60:
        rb = boot(t2, t2.il14.to_numpy(), {"U2": t2.U2d.to_numpy(), "M2": t2.p_M2.to_numpy(), "B0": t2.p_B0.to_numpy()}, [("M2", "B0")])
        print(f"{region} body ({npos} positive rows)\n" + rb.round(4).to_string())
    else:
        print(f"{region} body: {npos} positive rows, fewer than 60, not analysed")

units = []
for b, g in test.groupby("batter"):
    pos_ = g[g.il14]
    for _, u in pos_.groupby(pos_.day + pd.to_timedelta(pos_.days_to_il, unit="D")):
        units.append((b, 1, u.U2d.max(), u.p_M3.max(), u.U2d.mean(), u.p_M3.mean(), len(u)))
    neg = g[~g.il14]
    for _, u in neg.groupby((neg.day - g.day.min()).dt.days // H):
        units.append((b, 0, u.U2d.max(), u.p_M3.max(), u.U2d.mean(), u.p_M3.mean(), len(u)))
units = pd.DataFrame(units, columns=["batter", "y", "U2 max", "M3 max", "U2 mean", "M3 mean", "rows"])
ue = boot(units, units.y.to_numpy(), {k: units[k].to_numpy() for k in ("U2 max", "M3 max", "U2 mean", "M3 mean")}, [])
print(f"Event level: {int(units.y.sum())} pre-IL units (median {units.rows[units.y == 1].median():.0f} rows), "
      f"{int((units.y == 0).sum())} other 14-day blocks\n" + ue.round(4).to_string())

n_day = test.groupby("day").U2.transform("count")
rank_desc = test.groupby("day").U2.rank(method="first", ascending=False)
alarm = (rank_desc <= np.maximum(1, np.floor(0.01 * n_day))).to_numpy()
il_test = (test[test.il14].assign(effective=lambda d: d.day + pd.to_timedelta(d.days_to_il, unit="D"))[["batter", "effective"]].drop_duplicates())
caught = [bool(alarm[(test.batter == r.batter).to_numpy() & (test.day >= r.effective - pd.Timedelta(days=H)).to_numpy()
                     & (test.day <= r.effective).to_numpy()].any()) for _, r in il_test.iterrows()]
il26 = il_lab[il_lab.effective.dt.year == TEST]
first_day = rows[rows.season == TEST].groupby("batter").day.min()
no_row = sum(1 for _, r in il26.iterrows() if r.batter in first_day.index and r.effective >= first_day[r.batter]
             and not ((test.batter == r.batter) & (test.day >= r.effective - pd.Timedelta(days=H)) & (test.day <= r.effective)).any())
before_first = int(sum(r.batter in first_day.index and r.effective < first_day[r.batter] for _, r in il26.iterrows()))
print(f"alarm = top 1% of hitters that day: {int(alarm.sum()):,} alarm days in {TEST}; {y14[alarm].mean():.1%} followed by IL14 "
      f"(base rate {y14.mean():.1%}); {np.mean(caught):.1%} of {len(caught)} placements with pre-IL rows had an alarm in the {H} days before")
n26 = int(sum(r.batter in first_day.index for _, r in il26.iterrows()))
print(f"{TEST} placements (MLB or Triple-A club) of hitters with Triple-A rows: {n26}; {no_row} ({no_row / max(n26, 1):.1%}) had no row in the {H} days before after his first "
      f"Triple-A game; {before_first} ({before_first / max(n26, 1):.1%}) came before his first {TEST} Triple-A game")

# %% [markdown]
# ## What the models lean on (2026, permutation importance)

# %%
imps = {}
for k in ("M2", "M3"):
    pi = permutation_importance(models[k], test[SETS[k]].to_numpy(float), y14, scoring="roc_auc", n_repeats=5, random_state=3)
    imps[k] = pd.Series(pi.importances_mean, index=SETS[k])


def group_of(c):
    if c in CTX:
        return "rival (B0)"
    base = c.replace("z_", "").replace("D_", "").replace(" (short)", "")
    if base in ("U1", "U2"):
        return "summary scores"
    return "results" if base in RESULTS else "process"


print(pd.DataFrame({k: v.groupby(v.index.map(group_of)).sum() for k, v in imps.items()}).round(4).to_string())
imp = imps["M3"].sort_values(ascending=False).head(10)
print(imp.round(4).to_string())

# %% [markdown]
# ## Figures

# %%
fig, ax = plt.subplots(figsize=(8, 7))
for k, c, s, yy in (("B0", "#8a8a8a", test.p_B0.to_numpy(), y14), ("M3", "#c0392b", test.p_M3.to_numpy(), y14),
                    ("U2 (2024-2026)", "#2471a3", u2d_p1, y_p1)):
    m = np.isfinite(s)
    fpr, tpr, _ = roc_curve(yy[m], s[m])
    ax.plot(fpr, tpr, color=c, lw=2.5, label=f"{k}  AUC {auc(yy, s):.3f}")
ax.plot([0, 1], [0, 1], color="#cccccc", ls="--")
ax.set_xlabel("False alarm rate")
ax.set_ylabel("Share of pre-IL days flagged")
ax.set_title(f"Triple-A: injured list within {H} days")
ax.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig("aaa_roc.png", dpi=120)
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))
for i, (name, real, fl) in enumerate((("P2  M2 - B0", gap2, p2_floor), ("P3  M3 - M2", gap3, p3_floor))):
    ax.scatter(np.full(len(fl), i), fl, color="#8a8a8a", s=40, label="floors" if i == 0 else None)
    ax.scatter([i], [real], color="#c0392b", s=110, zorder=3, label="real" if i == 0 else None)
ax.axhline(0, color="#cccccc", lw=1)
ax.set_xticks([0, 1], ["P2  M2 - B0", "P3  M3 - M2"])
ax.set_ylabel("AUC gain, 2026")
ax.set_title("Real gain against its floors")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("aaa_floors.png", dpi=120)
plt.show()

summary = pd.DataFrame({
    "test": ["P1 U2 daily rank, 2024-2026", "P2 M2 over B0, 2026", "P3 M3 over M2, 2026"],
    "statistic": [r_p1.loc["U2", "AUC"], gap2, gap3],
    "lo_bonf": [r_p1.loc["U2", "lo_bonf"], main.loc["M2 - B0", "lo_bonf"], main.loc["M3 - M2", "lo_bonf"]],
    "floor_max": [p1_floor.max(), p2_floor.max(), p3_floor.max()],
    "verdict": ["SUPPORTED" if v else "NOT SUPPORTED" for v in (P1, P2, P3)],
    "note": [verdict_note.get(k, "") for k in ("P1", "P2", "P3")]})
if FAKE:
    print("FAKE LABELS: code test only, no verdict written")
else:
    summary.to_csv("aaa_verdict.csv", index=False)
    main.to_csv("aaa_auc_2026.csv")
print(summary.to_string(index=False))
