# %% [markdown]
# # Does a Pitcher Drift From His Own Normal Before the Injured List?
#
# Two pre-registered studies found that a **hitter's** drift from his own normal does not come before the injured list ([MLB 2026](https://www.kaggle.com/code/yasunorim/do-hitters-drift-from-normal-before-the-il), and Triple-A 2024-2026). A hitter's numbers are outcomes of a swing against a pitch. A **pitcher's** delivery is measured on every pitch: velocity, spin, extension, release point, arm angle, movement. Most pitcher placements are for the arm.
#
# - **14 parameters** per pitcher and appearance, each against **the same pitcher's own normal**.
# - **P1** MLB 2024-2026, all placements; **P3** MLB, arm placements only; **P4** Triple-A 2024-2026 (unsupervised, no training).
# - **P2** MLB 2026: an anomaly model over a rival that knows age, injury history and workload.
#
# Everything was fixed before any pitcher feature was compared with an injured-list move: [FREEZE_pitch.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch.md).

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 12,
                     "xtick.labelsize": 12, "ytick.labelsize": 12})

INPUT_ROOT = os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input")
ROW_SEASONS, TRAIN, TEST = [2024, 2025, 2026], [2024, 2025], 2026
SMOKE = os.environ.get("SMOKE") == "1"
FAKE = os.environ.get("SMOKE_FAKE_LABELS") == "1"
# Windows: (fastball pitches, non-fastball pitches, all pitches, swings, batters faced, balls in play)
LONG = (12, 8, 24, 12, 8, 4) if SMOKE else (150, 100, 300, 150, 100, 40)
SHORT = (4, 3, 8, 4, 3, 2) if SMOKE else (50, 30, 100, 50, 30, 15)
BASE_LAG, MIN_BASE, MIN_SCALE, MIN_Z, MIN_D = (0, 2, 3, 4, 2) if SMOKE else (30, 10, 20, 7, 5)
N_BOOT, N_FLOOR, N_FLOOR_P1 = (30, 4, 20) if SMOKE else (2000, 20, 200)
H, H30 = 14, 30
LO, HI = 100 * 0.05 / 4 / 2, 100 - 100 * 0.05 / 4 / 2  # Bonferroni over four primaries: 0.625% and 99.375%
OHTANI = 660271
FB = ("FF", "SI", "FC")
SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt",
         "missed_bunt", "bunt_foul_tip"}
MISS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
ARM = re.compile(r"\b(elbow|shoulder|forearm|arm|ucl|ulnar|biceps?|triceps?|rotator cuff|flexor|lat|latissimus|teres|labrum|tommy john|thoracic outlet)\b")
NOT_ARM = re.compile(r"\b(hip|groin|oblique|back|knee|ankle|foot|hamstring|quad|calf)")


def is_arm(sentence):
    t = str(sentence).lower()
    return bool(ARM.search(t)) and not NOT_ARM.search(t)


for _t, _want in (("left hip flexor strain", False), ("right lateral meniscus", False), ("right arm fatigue", True),
                  ("right UCL sprain", True), ("Right elbow inflammation", True), ("Left shoulder impingement", True)):
    assert is_arm(_t) == _want, _t
EMPTY = np.array([], dtype="datetime64[ns]")
PCOLS = ["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number", "pitch_type", "release_speed",
         "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z", "arm_angle", "pfx_x", "pfx_z",
         "zone", "description", "events", "type", "launch_speed", "woba_denom", "inning", "game_type"]
assert not [c for c in PCOLS if "until" in c or "next" in c], "future columns"
AAA_MD5 = {2023: "508eb3a096050b50b6563e844440f06a", 2024: "af5f2b7fc1707155613e1a2087cbbaca",
           2025: "849279d320d0f578e3b241eec5b2cb9a", 2026: "05a32b74a8ecf54793cb236bd78b8b5f"}
MLB_MD5 = {2024: "1e59aa536f7741d0e6981eb9e80b88b3", 2025: "9aaf0ae673cc5b3f7ab29598da418abe",
           2026: "49cbf1be3bc454a7769ca447f2ad8e03"}
assert FAKE or not SMOKE, "SMOKE windows are for the random-label code test only"


def load(pattern, years, md5s=None):
    frames = []
    for y in years:
        p = glob.glob(os.path.join(INPUT_ROOT, "**", pattern.format(y=y)), recursive=True)
        assert len(p) == 1, (pattern.format(y=y), p)
        m = hashlib.md5(open(p[0], "rb").read()).hexdigest()
        print(f"{os.path.basename(p[0])} md5 {m}")
        assert FAKE or m == md5s.get(y), "not the file recorded in FREEZE_pitch.md"
        d = pd.read_parquet(p[0], columns=PCOLS)
        d["season"] = y
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["game_date"] = pd.to_datetime(out["game_date"])
    out = out[out.game_type == "R"]
    assert not out.duplicated(["game_pk", "at_bat_number", "pitch_number"]).any(), "duplicate pitches"
    return out.sort_values(["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number"], kind="stable").reset_index(drop=True)


mlb = load("statcast_{y}.parquet", ROW_SEASONS, MLB_MD5)
aaa = load("aaa_pitching_{y}.parquet", [2023] + ROW_SEASONS, AAA_MD5)
for name, d in (("MLB", mlb), ("Triple-A", aaa)):
    fill = d.groupby("season")[["release_speed", "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z",
                                 "arm_angle", "pfx_x", "pfx_z", "pitch_type", "zone"]].apply(lambda g: g.notna().mean())
    print(f"{name}: {len(d):,} pitches, {d.pitcher.nunique():,} pitchers\n" + fill.round(4).to_string())
    assert (fill.pitch_type >= 0.90).all(), f"{name}: pitch_type under 90% filled"
season_last = {"mlb": mlb.groupby("season").game_date.max(), "aaa": aaa.groupby("season").game_date.max()}
season_first = {"mlb": mlb.groupby("season").game_date.min(), "aaa": aaa.groupby("season").game_date.min()}
if not SMOKE:
    for lv in season_last:
        for y, d in season_last[lv].items():
            assert pd.Timestamp(y, 9, 15) <= d <= pd.Timestamp(y, 10, 6), (lv, y, d)

# %% [markdown]
# ## MLB StatsAPI: teams, transactions, people, game logs (as the Triple-A hitter study)

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
                    tx_rows.append({"pitcher": t["person"]["id"], "kind": kind, "description": d, "effective": eff,
                                    "level": team_level.get((eff.year, club)),
                                    "from_level": team_level.get((eff.year, (t.get("fromTeam") or {}).get("id")))})
            time.sleep(0.3)
tx = pd.DataFrame(tx_rows)


def merged(df):
    df = df.drop_duplicates(["pitcher", "effective"]).sort_values(["pitcher", "effective"])
    return df[~(df.pitcher.eq(df.pitcher.shift()) & (df.effective - df.effective.shift()).dt.days.lt(10))].reset_index(drop=True)


placed = tx[tx.kind == "placed"]
il_lab = merged(placed[placed.level.isin(["mlb", "aaa"])])
il_hist = merged(placed[placed.level.isin(["mlb", "aaa", "aa"])])
returns = tx[(tx.kind == "returned") & tx.level.isin(["mlb", "aaa", "aa"])].drop_duplicates(["pitcher", "effective"])
last_ = il_lab.description.str.strip().str.rstrip(".").str.split(". ", regex=False).str[-1]
injury = last_.where(~last_.str.lower().str.contains("injured list"), "")
il_lab["text"] = injury.ne("").to_numpy()
il_lab["arm"] = np.array([is_arm(x) for x in injury])
_lh = il_hist.description.str.strip().str.rstrip(".").str.split(". ", regex=False).str[-1]
il_hist["arm"] = np.array([is_arm(x) if "injured list" not in x.lower() else False for x in _lh])
il_lab["list"] = il_lab.description.str.extract(r"(\d+-day|full-season)", expand=False).fillna("other")
print("label placements by level and list:\n" + il_lab.groupby(["level", "list"]).size().to_string())
print(f"label placements {len(il_lab)}; injury text {il_lab.text.mean():.1%}; arm {il_lab.arm.mean():.1%} "
      f"(by level {il_lab.groupby('level').arm.mean().round(3).to_dict()})")

ids = np.union1d(mlb.pitcher.unique(), aaa.pitcher.unique())
people = {}
for i in range(0, len(ids), 150):
    for p in get("/people", personIds=",".join(map(str, ids[i:i + 150])))["people"]:
        people[p["id"]] = (p.get("primaryPosition", {}).get("abbreviation"), p.get("birthDate"), p.get("fullName"),
                           p.get("pitchHand", {}).get("code"))
assert len(people) == len(ids), "every pitcher needs a person record"
keep_ids = {b for b, v in people.items() if v[0] in ("P", "TWP")}


def season_pitches(js, sport):
    out = {}
    for p in js["people"]:
        sp = [x for s_ in p.get("stats", []) for x in s_.get("splits", []) if x.get("sport", {}).get("id", sport) == sport]
        tot = [x for x in sp if "team" not in x]
        out[p["id"]] = float(sum(x["stat"].get("numberOfPitches", 0) for x in (tot or sp)))
    return out


prev_pitches, app_days = {}, {lv: {b: [] for b in ids} for lv in ("mlb", "aaa")}
game_logs = {lv: {b: [] for b in ids} for lv in ("mlb", "aaa")}
for i in range(0, len(ids), 150):
    chunk = ",".join(map(str, ids[i:i + 150]))
    for y in ROW_SEASONS:
        g1 = season_pitches(get("/people", personIds=chunk, hydrate=f"stats(group=[pitching],type=[season],season={y - 1},sportId=1)"), 1)
        g11 = season_pitches(get("/people", personIds=chunk, hydrate=f"stats(group=[pitching],type=[season],season={y - 1},sportId=11)"), 11)
        for b in ids[i:i + 150]:
            prev_pitches[(b, y)] = g1.get(b, 0.0) + g11.get(b, 0.0)
    for y in [2023] + ROW_SEASONS:
        for sp, lv in ((1, "mlb"), (11, "aaa")):
            js = get("/people", personIds=chunk, hydrate=f"stats(group=[pitching],type=[gameLog],season={y},sportId={sp})")
            for p in js["people"]:
                for x in (x for s_ in p.get("stats", []) for x in s_.get("splits", []) if x.get("sport", {}).get("id", sp) == sp and x.get("date")):
                    app_days[lv][p["id"]].append(x["date"])
                    game_logs[lv][p["id"]].append((x["date"], float(x.get("stat", {}).get("numberOfPitches", 0) or 0)))
    time.sleep(0.3)
app_days = {lv: {b: np.unique(np.array(v, dtype="datetime64[D]")).astype("datetime64[ns]") for b, v in d.items()} for lv, d in app_days.items()}
game_pitches = {}
for lv, d in game_logs.items():
    game_pitches[lv] = {}
    for b, v in d.items():
        if v:
            gl = pd.DataFrame(v, columns=["date", "n"]).assign(date=lambda x: pd.to_datetime(x.date)).groupby("date").n.sum()
            game_pitches[lv][b] = (gl.index.to_numpy(), gl.to_numpy(float))
print(f"{len(ids) - len(keep_ids)} non-pitchers who pitched are left out; game logs: "
      f"{sum(len(v) > 0 for v in app_days['mlb'].values()):,} with MLB days, {sum(len(v) > 0 for v in app_days['aaa'].values()):,} with Triple-A days")

# %% [markdown]
# ## Rehab windows (Triple-A pitches of a pitcher on an MLB club's injured list are removed)

# %%
mlb_rehab = (tx.kind == "rehab") & (tx.from_level == "mlb")
starts = tx[(((tx.kind == "placed") & (tx.level == "mlb")) | mlb_rehab) & tx.pitcher.isin(ids)]
ends_ = tx[((tx.kind == "returned") & tx.level.isin(["mlb", "aaa"])) | (tx.kind == "end")]
ends_by = {b: np.sort(g.effective.to_numpy()) for b, g in ends_.groupby("pitcher")}
rehab_seasons = set(zip(tx.loc[mlb_rehab, "pitcher"], tx.loc[mlb_rehab, "effective"].dt.year))
windows = []
for b, g in starts.groupby("pitcher"):
    en = ends_by.get(b, np.array([], dtype="datetime64[ns]"))
    for s0 in np.sort(g.effective.to_numpy()):
        y = pd.Timestamp(s0).year
        stop = np.datetime64(pd.Timestamp(y, 12, 31) + pd.Timedelta(days=1))
        if y in season_last["aaa"].index:
            stop = min(stop, np.datetime64(season_last["aaa"][y] + pd.Timedelta(days=1)))
        later = en[en > s0]
        closed = bool(len(later)) and later[0] < stop
        if len(later):
            stop = min(stop, later[0])
        windows.append((b, s0, stop, closed))
windows = pd.DataFrame(windows, columns=["pitcher", "start", "stop", "closed"])
in_rehab, in_open = np.zeros(len(aaa), bool), np.zeros(len(aaa), bool)
pb, pdte = aaa.pitcher.to_numpy(), aaa.game_date.to_numpy()
for b, g in windows.groupby("pitcher"):
    m = pb == b
    for s0, s1, cl in zip(g.start, g.stop, g.closed):
        w_ = (pdte[m] >= np.datetime64(s0)) & (pdte[m] < np.datetime64(s1))
        in_rehab[m] |= w_
        if not cl:
            in_open[m] |= w_
rem = pd.DataFrame({"pitcher": pb, "season": aaa.season.to_numpy(), "rehab": in_rehab, "open": in_open}).groupby(["pitcher", "season"]).agg(
    mean=("rehab", "mean"), open_share=("open", "mean"))
no_rehab_tx = ~pd.Series([k in rehab_seasons for k in rem.index], index=rem.index)
lost = rem[(rem.open_share > 0.5) & no_rehab_tx]
print(f"rehab windows {len(windows):,} ({int((~windows.closed).sum()):,} closed only by the season end); Triple-A pitches removed "
      f"{in_rehab.sum():,} ({in_rehab.mean():.2%}); pitcher-seasons losing >50%: {int((rem['mean'] > 0.5).sum())}; "
      f"to windows closed only by the season end without a rehab assignment: {len(lost)}")
assert len(lost) == 0, lost.head(20)
aaa = aaa[~in_rehab].reset_index(drop=True)
# The label rule ignores Triple-A appearances inside rehab windows (a rehab after a placement is not "pitching at the
# other level before it"); workload items keep them.
rule_days = {"mlb": app_days["mlb"], "aaa": {}}
win_by = {b: list(zip(g.start.to_numpy(), g.stop.to_numpy())) for b, g in windows.groupby("pitcher")}
n_rehab_days = 0
for b, d_ in app_days["aaa"].items():
    keep_ = np.ones(len(d_), bool)
    for s0, s1 in win_by.get(b, []):
        keep_ &= ~((d_ >= np.datetime64(s0)) & (d_ < np.datetime64(s1)))
    n_rehab_days += int((~keep_).sum())
    rule_days["aaa"][b] = d_[keep_]
print(f"Triple-A appearance days inside rehab windows, ignored by the other-level rule: {n_rehab_days:,}")

# %% [markdown]
# ## Fourteen parameters per appearance (only earlier pitches, same season and level)
#
# Fastball parameters are separate series per fastball type (FF, SI, FC), each against its own normal; the z used on a day is the one of that day's primary fastball. Scales and daily ranks are taken within role (starter / reliever).

# %%
FBP = ["FB velocity", "FB top velocity", "FB spin", "Extension", "Release height", "Release side", "Arm angle",
       "FB vertical movement", "FB horizontal movement"]
FBCOL = {"FB velocity": "release_speed", "FB top velocity": "release_speed", "FB spin": "release_spin_rate",
         "Extension": "release_extension", "Release height": "release_pos_z", "Release side": "release_pos_x",
         "Arm angle": "arm_angle", "FB vertical movement": "pfx_z", "FB horizontal movement": "pfx_x"}
OTHER = ["Off-speed velocity", "Zone rate", "Whiff", "K - BB", "Hard-hit allowed"]
OTHERCOL = {"Off-speed velocity": "release_speed", "Zone rate": "zone", "Whiff": "description", "K - BB": "events",
            "Hard-hit allowed": "launch_speed"}
BASE14 = FBP + OTHER
DELIVERY = FBP + ["Off-speed velocity"]
SFX = ("", " (short)")
P14, P14S = BASE14, [n + " (short)" for n in BASE14]
ZPARAMS = P14 + P14S
# Fill gates (label-free): a parameter whose source column is under 90% filled in any season of a population is dropped.
DROPPED = {}
for name, d in (("mlb", mlb), ("aaa", aaa)):
    fr = d.groupby("season")[sorted(set(FBCOL.values()) | {"zone", "pitch_type", "description"})].apply(lambda g: g.notna().mean()).min()
    fr["launch_speed"] = d[d.type == "X"].groupby("season").launch_speed.apply(lambda v: v.notna().mean()).min()
    fr["events"] = 1.0  # a batter-faced row is defined by events itself
    src = {**FBCOL, **OTHERCOL}
    DROPPED[name] = sorted(n for n in BASE14 if fr[src[n]] < 0.90)
    print(f"{name}: lowest season fill {fr.round(4).to_dict()}; parameters dropped: {DROPPED[name] or 'none'}")
NOT_PA = ("caught_stealing", "pickoff", "stolen_base", "wild_pitch", "passed_ball", "other_advance", "truncated_pa",
          "game_advisory", "runner_double_play")
NOT_OFFSPEED = ("FA", "PO", "IN", "EP")


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


def wpct(dates, v, w, q, days):
    v = np.asarray(v, float)
    k = np.searchsorted(dates, days, side="left")
    out = np.full(len(days), np.nan)
    for i, kk in enumerate(k):
        if kk >= w:
            x = v[kk - w:kk]
            x = x[np.isfinite(x)]
            if len(x) >= w / 2:
                out[i] = np.percentile(x, q)
    return out


def params_for(g, prev_type):
    """g: one pitcher-season at one level, sorted by date, game, at-bat, pitch.
    Returns days, per-type fastball series {(type, name): values}, other series {name: values}, primary choice index
    per day (-1 none), start flag per day, and workload items before each day."""
    days = np.unique(g.game_date.to_numpy())
    gd = g.game_date.to_numpy()
    pt = g.pitch_type.fillna("").to_numpy(dtype=object)
    before = np.vstack([np.searchsorted(gd[pt == t], days, side="left") for t in FB])
    choice = np.where(before.max(0) > 0, before.argmax(0), FB.index(prev_type) if prev_type else -1)
    fb = {}
    for t in FB:
        s_ = g[pt == t]
        sd = s_.game_date.to_numpy()
        for sfx, w in zip(SFX, (LONG[0], SHORT[0])):
            for n, c in FBCOL.items():
                v = s_[c].to_numpy(float)
                fb[(t, n + sfx)] = wpct(sd, v, w, 90, days) if n == "FB top velocity" else wmean(sd, v, w, days)
    oth = {}
    nfb = g[(pt != "") & ~np.isin(pt, FB + NOT_OFFSPEED)]
    sw = g[g.description.isin(SWING).to_numpy()]
    ev = g.events.fillna("").astype(str)
    pa = g[((ev != "") & ~ev.str.startswith(NOT_PA)).to_numpy()]
    kbb = np.where(pa.events.isin(["strikeout", "strikeout_double_play"]), 1.0, np.where(pa.events.eq("walk"), -1.0, 0.0))
    bip = g[(g.type == "X").to_numpy() & g.launch_speed.notna().to_numpy()]
    zr = np.where(g.zone.isna(), np.nan, g.zone.between(1, 9).astype(float))
    for sfx, (w_fb, w_nfb, w_all, w_sw, w_bf, w_bip) in zip(SFX, (LONG, SHORT)):
        oth["Off-speed velocity" + sfx] = wmean(nfb.game_date.to_numpy(), nfb.release_speed.to_numpy(float), w_nfb, days)
        oth["Zone rate" + sfx] = wmean(gd, zr, w_all, days)
        oth["Whiff" + sfx] = wmean(sw.game_date.to_numpy(), sw.description.isin(MISS).to_numpy(float), w_sw, days)
        oth["K - BB" + sfx] = wmean(pa.game_date.to_numpy(), kbb, w_bf, days)
        oth["Hard-hit allowed" + sfx] = wmean(bip.game_date.to_numpy(), bip.launch_speed.ge(95).to_numpy(float), w_bip, days)
    per_day = pd.Series(1, index=gd).groupby(level=0).sum().reindex(days).to_numpy(float)
    start = g.groupby("game_date").inning.min().reindex(days).eq(1).to_numpy(float)
    inn_max = g.groupby(["game_date", "inning"]).size().groupby(level=0).max().reindex(days).to_numpy(float)
    n = len(days)
    wl = {k: np.full(n, np.nan) for k in ("rest", "prev_app_pitches", "p7", "p14", "p28", "start_share", "prev_start",
                                          "consec7", "prev_max_inning")}
    for i in range(n):
        if i:
            wl["rest"][i] = (days[i] - days[i - 1]) / np.timedelta64(1, "D")
            wl["prev_app_pitches"][i] = per_day[i - 1]
            wl["prev_start"][i] = start[i - 1]
            wl["prev_max_inning"][i] = inn_max[i - 1]
        wl["start_share"][i] = start[:i].mean() if i else np.nan
        for k_, L in (("p7", 7), ("p14", 14), ("p28", 28)):
            lo = np.searchsorted(days, days[i] - np.timedelta64(L, "D"), side="left")
            wl[k_][i] = per_day[lo:i].sum()
        lo7 = np.searchsorted(days, days[i] - np.timedelta64(7, "D"), side="left")
        d7 = days[lo7:i]
        wl["consec7"][i] = int(np.sum(np.diff(d7) == np.timedelta64(1, "D"))) if len(d7) > 1 else 0
    return days, fb, oth, choice, start, pd.DataFrame(wl)


def mad(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def own_z(days, x, prev, scale_t):
    """Centre as FREEZE_anomaly.md (in-season up to t - 30 days, at least MIN_BASE values, else the previous season's
    median); scale_t: the scale for each day (chosen by the caller). Returns z and the centre kind (1 in-season, 3 previous)."""
    fin = np.isfinite(x)
    cur_x, cur_d = x[fin], days[fin]
    px = prev[np.isfinite(prev)] if prev is not None else np.array([])
    prev_centre = np.median(px) if len(px) >= MIN_BASE else None
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
        if np.isfinite(scale_t[i]) and scale_t[i] > 0:
            z[i] = np.clip((x[i] - centre) / scale_t[i], -10, 10)
            kind[i] = k
    return z, kind


Z14, Z14S = ["z_" + n for n in P14], ["z_" + n for n in P14S]
ZDEL = ["z_" + n + " (short)" for n in DELIVERY]
DCOLS = ["D_" + n for n in P14S] + ["D_U2"]


def u_scores(zm, need):
    nz = np.isfinite(zm).sum(1)
    u1 = np.where(nz >= need, np.nanmax(np.abs(np.where(np.isfinite(zm), zm, 0.0)), 1), np.nan)
    u2 = np.where(nz >= need, np.nanmean(zm ** 2, 1), np.nan)
    return u1, u2


def season_to_date(df):
    out = np.full((len(df), len(DCOLS)), np.nan)
    vals = df[Z14S + ["U2"]].to_numpy(float)
    for _, ix in df.groupby(["pitcher", "season"]).indices.items():
        v = vals[ix]
        f = np.isfinite(v)
        cs, cn = np.cumsum(np.where(f, v, 0.0), 0), np.cumsum(f, 0)
        prev_s = np.vstack([np.zeros((1, v.shape[1])), cs[:-1]])
        prev_n = np.vstack([np.zeros((1, v.shape[1])), cn[:-1]])
        out[ix] = np.where(prev_n >= MIN_D, prev_s / np.maximum(prev_n, 1), np.nan)
    return out


def daily_rank(df, u, by_role=True):
    keys = ["day", "role"] if by_role else ["day"]
    g_ = df.groupby(keys)[u]
    return (g_.rank(method="average") - 0.5) / g_.transform("count")


def recompute_scores(df):
    df["U1"], df["U2"] = u_scores(df[Z14S].to_numpy(), MIN_Z)
    df["U2_del"] = u_scores(df[ZDEL].to_numpy(), 5)[1]
    df[DCOLS] = season_to_date(df)
    return df


def other_level_pitches(b, other):
    return game_pitches[other].get(b, (EMPTY, np.array([])))


def build_rows(pitches, level, league_season, other):
    """One row per pitcher and appearance day at this level, 2024-2026, no label."""
    pitches = pitches[pitches.pitcher.isin(keep_ids)]
    groups = {k: g for k, g in pitches.groupby(["pitcher", "season"], sort=False)}
    modes, fbvel = {}, {}
    for k, g in groups.items():
        c = g.pitch_type.value_counts()
        cnt = [c.get(t, 0) for t in FB]
        modes[k] = FB[int(np.argmax(cnt))] if max(cnt) > 0 else None
        fbvel[k] = g[g.pitch_type.isin(FB)].groupby("pitch_type").release_speed.mean().to_dict()
    series = {k: params_for(g, modes.get((k[0], k[1] - 1))) for k, g in groups.items()}
    srole = {k: float(np.nanmean(s[4]) > 0.5) for k, s in series.items()}  # season role (more than half starts)
    # planted look-ahead check: pitches from t0 on set to extreme values move nothing up to t0
    k0 = max((k for k in series if k[1] in ROW_SEASONS), key=lambda k: len(series[k][0]))
    s0 = series[k0]
    t0 = s0[0][len(s0[0]) // 2]
    g0 = groups[k0].copy()
    late = (g0.game_date >= t0).to_numpy()
    for c in ["release_speed", "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z", "arm_angle",
              "pfx_x", "pfx_z", "launch_speed"]:
        g0.loc[late, c] = 999.0
    g0.loc[late, "pitch_type"] = "FC"
    g0.loc[late, "events"] = "strikeout"
    g0.loc[late, ["zone", "description", "events", "type", "inning"]] = [12.0, "swinging_strike", "strikeout", "X", 1]
    s0b = params_for(g0, modes.get((k0[0], k0[1] - 1)))
    i0 = list(s0[0]).index(t0)
    for key in s0[1]:
        assert np.array_equal(s0[1][key][: i0 + 1], s0b[1][key][: i0 + 1], equal_nan=True), key
    for key in s0[2]:
        assert np.array_equal(s0[2][key][: i0 + 1], s0b[2][key][: i0 + 1], equal_nan=True), key
    assert np.array_equal(s0[3][: i0 + 1], s0b[3][: i0 + 1]) and s0[5].iloc[: i0 + 1].equals(s0b[5].iloc[: i0 + 1])
    assert any(not np.array_equal(s0[2][n][i0 + 1:], s0b[2][n][i0 + 1:], equal_nan=True) for n in s0[2])
    # league-typical scale per parameter and role (season role), in the league season
    lscale = {}
    for n in ZPARAMS:
        for role in (0.0, 1.0):
            ms = []
            for k, s in series.items():
                if k[1] != league_season or srole[k] != role:
                    continue
                v = s[1].get((modes[k], n)) if n.replace(" (short)", "") in FBP else s[2][n]
                if v is not None and np.isfinite(v).sum() >= MIN_SCALE:
                    ms.append(mad(v[np.isfinite(v)]))
            lscale[(n, role)] = float(np.median(ms)) if ms else np.nan
    recs, changes = [], []
    for key, (days, fb, oth, choice, start, wl) in series.items():
        b, y = key
        if y not in ROW_SEASONS or (b == OHTANI and y == TEST):
            continue
        prev = series.get((b, y - 1))
        prole = srole.get((b, y - 1))
        role = np.where(np.isfinite(wl.start_share.to_numpy()), (wl.start_share.to_numpy() > 0.5).astype(float),
                        prole if prole is not None else 0.0)
        changes.append(int(np.sum(np.diff(choice[choice >= 0]) != 0)))
        r = pd.DataFrame({"pitcher": b, "season": y, "day": days, "role": role, "no_primary": choice < 0})
        for n in ZPARAMS:
            base = n.replace(" (short)", "")
            lsc = np.where(role == 1.0, lscale[(n, 1.0)], lscale[(n, 0.0)])
            if base in FBP:
                zt = []
                for t in FB:
                    x = fb[(t, n)]
                    pv = prev[1].get((t, n)) if prev is not None else None
                    own = mad(pv[np.isfinite(pv)]) if pv is not None and np.isfinite(pv).sum() >= MIN_SCALE else np.nan
                    sc = np.where(np.isfinite(own) & (own > 0) & (role == prole), own, lsc)
                    zt.append(own_z(days, x, pv, sc))
                r[n] = np.where(choice >= 0, np.vstack([fb[(t, n)] for t in FB])[np.maximum(choice, 0), np.arange(len(days))], np.nan)
                zz = np.vstack([z_[0] for z_ in zt])
                kk = np.vstack([z_[1] for z_ in zt])
                r["z_" + n] = np.where(choice >= 0, zz[np.maximum(choice, 0), np.arange(len(days))], np.nan)
                kind = np.where(choice >= 0, kk[np.maximum(choice, 0), np.arange(len(days))], 0)
            else:
                x = oth[n]
                pv = prev[2][n] if prev is not None else None
                own = mad(pv[np.isfinite(pv)]) if pv is not None and np.isfinite(pv).sum() >= MIN_SCALE else np.nan
                sc = np.where(np.isfinite(own) & (own > 0) & (role == prole), own, lsc)
                r[n] = x
                r["z_" + n], kind = own_z(days, x, pv, sc)
            if n in DROPPED[level] or base in DROPPED[level]:
                r["z_" + n] = np.nan
            if n == "Zone rate (short)":
                r["base_kind"] = kind
        for c in wl.columns:
            r[c] = wl[c].to_numpy()
        r["start_share"] = r.start_share.fillna(0.5)
        r["prev_start_share"] = np.nanmean(prev[4]) if prev is not None else np.nan
        r["role_change"] = (role != prole).astype(float) if prole is not None else np.nan
        pos, birth, _, hand = people[b]
        r["throws"] = float(hand == "L")
        r["age"] = (days - np.datetime64(birth)) / np.timedelta64(1, "D") / 365.25 if birth else np.nan
        r["prev_season_pitches"] = prev_pitches[(b, y)]
        pvel = fbvel.get((b, y - 1), {})
        r["prev_fb_velocity"] = [pvel.get(FB[c], np.nan) if c >= 0 else np.nan for c in choice]
        od = app_days[other].get(b, EMPTY)
        ko = np.searchsorted(od, days, side="left")
        r["days_since_other"] = np.where(ko > 0, np.minimum((days - od[np.maximum(ko - 1, 0)]) / np.timedelta64(1, "D"), 365) if len(od) else 365, 365)
        r["other_apps_season"] = ko - np.searchsorted(od, np.datetime64(f"{y}-01-01"), side="left")
        # pitches at both levels before t (game logs)
        for k_, L in (("p7_all", 7), ("p14_all", 14), ("p28_all", 28)):
            tot = np.zeros(len(days))
            for lv in ("mlb", "aaa"):
                gdts, gp = game_pitches[lv].get(b, (EMPTY, np.array([])))
                if len(gdts):
                    cs = np.r_[0.0, np.cumsum(gp)]
                    hi = np.searchsorted(gdts, days, side="left")
                    lo = np.searchsorted(gdts, days - np.timedelta64(L, "D"), side="left")
                    tot += cs[hi] - cs[lo]
            r[k_] = tot
        r["apps_season"] = np.arange(len(days))
        r["day_of_season"] = (days - np.datetime64(season_first[level][y])) / np.timedelta64(1, "D")
        recs.append(r)
    rows = pd.concat(recs, ignore_index=True).sort_values(["pitcher", "season", "day"], kind="stable").reset_index(drop=True)
    rows = recompute_scores(rows)
    rows["end"] = rows.season.map(season_last[level])
    rows["ok14"] = rows.day + pd.Timedelta(days=H) <= rows.end
    rows["ok30"] = rows.day + pd.Timedelta(days=H30) <= rows.end
    for u in ("U1", "U2", "U2_del"):
        rows[u + "d"] = daily_rank(rows, u)
    rows["U2d_pooled"] = daily_rank(rows, "U2", by_role=False)
    print(f"{level}: {len(rows):,} rows, {rows.pitcher.nunique():,} pitchers; primary-fastball changes per pitcher-season: "
          f"mean {np.mean(changes):.2f}, max {max(changes)}; rows without a primary fastball: {int(rows.no_primary.sum()):,}")
    print("  league scales (short, reliever / starter):",
          {n: (round(lscale[(n, 0.0)], 4), round(lscale[(n, 1.0)], 4)) for n in P14S})
    print("  median raw U2 by season and month:\n" + rows[rows.U2.notna()].assign(month=lambda d: d.day.dt.month)
          .groupby(["season", "month"]).U2.median().round(2).unstack().to_string())
    print("  U2 defined by season:", rows.groupby("season").U2.apply(lambda v: round(v.notna().mean(), 3)).to_dict(),
          "| starter share of rows:", round(rows.role.mean(), 3))
    print("  centre kinds by role (0 none, 1 in-season, 3 previous season):\n"
          + rows.groupby("role").base_kind.value_counts(normalize=True).round(3).unstack().to_string())
    print("  sd of short z by role:", rows.groupby("role")[Z14S].std().median(axis=1).round(3).to_dict(),
          "| pitchers per day with U2: median", int(rows.groupby("day").U2.count().median()))
    hm = rows.groupby("pitcher").agg(rank=("U2d", "mean"), n=("day", "size"), has_prev=("prev_season_pitches", lambda v: float(v.iloc[0] > 0)),
                                     starter=("role", "mean"))
    print("  spearman of mean daily U2 rank with rows / previous season / starter share:",
          hm.corr(method="spearman").loc["rank", ["n", "has_prev", "starter"]].round(3).to_dict())
    return rows


rows_m = build_rows(mlb, "mlb", 2025, "aaa")
rows_a = build_rows(aaa, "aaa", 2023, "mlb")

# %% [markdown]
# ## Labels: a placement within 14 days (MLB or Triple-A club); rows with an appearance at the other level in the window are dropped

# %%
if FAKE:  # random placements (about 0.4 per pitcher-season), unrelated to any feature
    rng0 = np.random.default_rng(99)
    fk = []
    for rows_x in (rows_m, rows_a):
        for (b, y), g in rows_x.groupby(["pitcher", "season"]):
            k = min(rng0.poisson(0.4), len(g))
            for d in np.sort(rng0.choice(g.day.to_numpy(), k, replace=False)) if k else []:
                fk.append({"pitcher": b, "effective": pd.Timestamp(d) + pd.Timedelta(days=1),
                           "arm": bool(rng0.random() < 0.55), "text": True})
    il_lab = pd.DataFrame(fk, columns=["pitcher", "effective", "arm", "text"]).drop_duplicates(["pitcher", "effective"])
il_by = {b: (g.effective.to_numpy(), g.arm.to_numpy(), g.text.to_numpy()) for b, g in il_lab.sort_values("effective").groupby("pitcher")}
hist_by = {b: np.sort(g.effective.to_numpy()) for b, g in il_hist.groupby("pitcher")}
harm_by = {b: np.sort(g.effective.to_numpy()) for b, g in il_hist[il_hist.arm].groupby("pitcher")}
ret_by = {b: np.sort(g.effective.to_numpy()) for b, g in returns.groupby("pitcher")}


def attach_labels(rows, other):
    parts = []
    for (b, y), ix in rows.groupby(["pitcher", "season"]).indices.items():
        days = rows.day.to_numpy()[ix]
        eff, arm, txt = il_by.get(b, (EMPTY, np.array([], bool), np.array([], bool)))
        k = np.searchsorted(eff, days, side="left")
        has_next = k < len(eff)
        kk = np.minimum(k, max(len(eff) - 1, 0))
        nxt = np.where(has_next, eff[kk] if len(eff) else days, days)
        od = rule_days[other].get(b, EMPTY)
        n_after = np.searchsorted(od, days, side="right")
        other14 = (np.searchsorted(od, days + np.timedelta64(H, "D"), side="right") - n_after) > 0
        other30 = (np.searchsorted(od, days + np.timedelta64(H30, "D"), side="right") - n_after) > 0
        a = ret_by.get(b, EMPTY)
        ka = np.searchsorted(a, days, side="right")
        parts.append(pd.DataFrame({
            "idx": ix, "days_to_il": np.where(has_next, (nxt - days) / np.timedelta64(1, "D"), np.inf),
            "other14": other14, "other30": other30,
            "next_arm": np.where(has_next, arm[kk] if len(eff) else False, False),
            "next_text": np.where(has_next, txt[kk] if len(eff) else False, False),
            "n_prev_il": np.searchsorted(hist_by.get(b, EMPTY), days, side="left"),
            "n_prev_arm": np.searchsorted(harm_by.get(b, EMPTY), days, side="left"),
            "days_since_act": np.where(ka > 0, np.minimum((days - a[np.maximum(ka - 1, 0)]) / np.timedelta64(1, "D"), 365) if len(a) else 365, 365)}))
    lab = pd.concat(parts).set_index("idx").sort_index()
    rows = rows.join(lab)
    rows["il14"], rows["il30"] = rows.days_to_il <= H, rows.days_to_il <= H30
    rows["use14"] = rows.ok14 & ~rows.other14
    rows["use30"] = rows.ok30 & ~rows.other30
    print(f"rows dropped for an appearance at the other level: {int((rows.other14 & rows.ok14).sum()):,} (14 days), "
          f"{int((rows.other30 & rows.ok30).sum()):,} (30 days); "
          f"usable rows {int(rows.use14.sum()):,} ({int(rows.il14[rows.use14].sum()):,} positive)")
    pre = rows[rows.use14 & rows.il14].assign(pl=lambda d: d.day + pd.to_timedelta(d.days_to_il, unit="D"))
    eff_s = pre.groupby(["season", "pitcher", "pl"]).U2.apply(lambda v: v.notna().mean()).groupby("season").sum()
    print(f"placements with a pre-IL row: {pre.groupby(['pitcher', 'pl']).ngroups:,}; effective sample by season "
          f"(placements x share with U2): {eff_s.round(1).to_dict()}")
    return rows


rows_m = attach_labels(rows_m, "aaa")
rows_a = attach_labels(rows_a, "mlb")
for name, rows_x in (("MLB", rows_m), ("Triple-A", rows_a)):
    lab_y = il_lab[il_lab.pitcher.isin(rows_x.pitcher.unique()) & il_lab.effective.dt.year.isin(ROW_SEASONS)]
    first = rows_x.groupby(["pitcher", "season"]).day.min()
    before_first = sum((r.pitcher, r.effective.year) in first.index and r.effective < first[(r.pitcher, r.effective.year)] for _, r in lab_y.iterrows())
    no_pre = 0
    for _, r in lab_y.iterrows():
        k_ = (r.pitcher, r.effective.year)
        if k_ in first.index and r.effective >= first[k_]:
            d_ = rows_x.day[(rows_x.pitcher == r.pitcher)]
            no_pre += not ((d_ >= r.effective - pd.Timedelta(days=H)) & (d_ <= r.effective)).any()
    print(f"{name}: placements of these pitchers in 2024-2026 {len(lab_y):,}; before his first appearance of that season "
          f"{before_first:,}; later but with no appearance in the {H} days before {no_pre:,}")

# %% [markdown]
# ## Tests

# %%
CTX = ["age", "throws", "n_prev_il", "n_prev_arm", "days_since_act", "prev_season_pitches", "apps_season", "day_of_season",
       "rest", "prev_app_pitches", "p7", "p14", "p28", "p7_all", "p14_all", "p28_all", "start_share", "prev_start",
       "prev_start_share", "role_change", "prev_fb_velocity", "consec7", "prev_max_inning", "days_since_other",
       "other_apps_season"]
assert len(set(CTX)) == len(CTX)
BLOCK = Z14 + Z14S + ["U1", "U2"]
SETS = {"B0": CTX, "M2": CTX + BLOCK, "M3": CTX + BLOCK + DCOLS}


def fit(cols, X, y):
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=(20 if SMOKE else 200),
                                       l2_regularization=1.0, early_stopping=False, random_state=0)
    return m.fit(X[cols].to_numpy(float), y)


def predict(m, cols, X):
    return m.predict_proba(X[cols].to_numpy(float))[:, 1]


def auc(y, s):
    m = np.isfinite(s)
    y, s = y[m], s[m]
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def boot(df, y, scores, diffs, seed=2, n=N_BOOT):
    idx_by = list(df.groupby("pitcher").indices.values())
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


def draw_k(n, rng):
    """Pitcher rotation rule: k uniform on [m, n - m], m = max(2, ceil(n / 6)); None when n < 7 (not rotated)."""
    if n < 7:
        return None
    m = max(2, int(np.ceil(n / 6)))
    return int(rng.integers(m, n - m + 1))


def rotated(vals, groups, sd):
    rng = np.random.default_rng(sd)
    rot = vals.copy()
    for f in groups:
        k = draw_k(len(f), rng)
        if k is not None:
            rot[f] = np.roll(vals[f], k)
    return rot


def unsupervised_test(df, name, seed0, score="U2d", label="il14", gate=False):
    """Daily-rank AUC on the given rows, 2024-2026 pooled, with its rotation floor (defined values only)."""
    df = df[np.isfinite(df[score])].reset_index(drop=True)
    y_, s_ = df[label].to_numpy(), df[score].to_numpy()
    gix = list(df.groupby(["pitcher", "season"]).indices.values())
    rot_share = sum(len(f) for f in gix if len(f) >= 7) / max(len(df), 1)
    rots = [rotated(s_, gix, sd) for sd in range(seed0, seed0 + N_FLOOR_P1)]
    floor = np.array([auc(y_, x) for x in rots])
    r = boot(df, y_, {"U": s_}, [])
    ok = bool(r.loc["U", "lo_bonf"] > 0.5 and np.all(r.loc["U", "AUC"] > floor))
    print(f"{name} [{score}]: AUC {r.loc['U', 'AUC']:.4f} [{r.loc['U', 'lo_bonf']:.4f}, {r.loc['U', 'hi_bonf']:.4f}]; rotations mean "
          f"{floor.mean():.4f}, max {floor.max():.4f} ({len(np.unique(np.round(floor, 8)))} distinct); values in rotated series "
          f"{rot_share:.1%}; positives {int(y_.sum()):,} -> {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    if gate:
        assert rot_share >= 0.90 or SMOKE, "fewer than 90% of defined values can be rotated"
    if FAKE and gate:  # the floor must be beatable by a planted signal
        planted = np.clip(s_ + 0.3 * y_, 0, 1.3)
        assert auc(y_, planted) > max(auc(y_, rotated(planted, gix, sd)) for sd in range(seed0, seed0 + N_FLOOR_P1)), name
    return {"auc": r.loc["U", "AUC"], "lo": r.loc["U", "lo_bonf"], "hi": r.loc["U", "hi_bonf"], "floor": floor, "ok": ok,
            "df": df, "rots": rots, "y": y_, "s": s_}


def by_subsets(res, cols):
    """AUC and floor mean of the same rotations on subsets of the scored rows."""
    df, y_, s_, rots = res["df"], res["y"], res["s"], res["rots"]
    out = []
    for name, m in cols.items():
        m = np.asarray(m)
        out.append({"subset": name, "AUC": auc(y_[m], s_[m]), "floor mean": np.nanmean([auc(y_[m], x[m]) for x in rots]),
                    "positives": int(y_[m].sum())})
    return pd.DataFrame(out)


def subsets_of(df):
    return {**{f"season {y}": df.season.eq(y).to_numpy() for y in ROW_SEASONS},
            "starters": df.role.eq(1.0).to_numpy(), "relievers": df.role.eq(0.0).to_numpy(),
            **{f"centre kind {k}": df.base_kind.eq(k).to_numpy() for k in (1, 3)},
            "more than 30 days after a return": (df.days_since_act > 30).to_numpy(),
            "more than 30 days after the other level": (df.days_since_other > 30).to_numpy()}


p1_df = rows_m[rows_m.use14].reset_index(drop=True)
P1 = unsupervised_test(p1_df, "P1 MLB, all placements", 1000, gate=True)
arm_df = p1_df[~p1_df.il14 | p1_df.next_arm].reset_index(drop=True)
P3 = unsupervised_test(arm_df, "P3 MLB, arm placements", 2000, gate=True)
p4_df = rows_a[rows_a.use14].reset_index(drop=True)
P4 = unsupervised_test(p4_df, "P4 Triple-A, all placements", 3000, gate=True)
SUB = {}
for key, res in (("P1", P1), ("P3", P3), ("P4", P4)):
    SUB[key] = by_subsets(res, subsets_of(res["df"]))
    print(f"{key} subsets (same rotations)\n" + SUB[key].round(4).to_string(index=False))
DEL = {"P1": unsupervised_test(p1_df, "P1 delivery only", 1000, "U2_deld"),
       "P3": unsupervised_test(arm_df, "P3 delivery only", 2000, "U2_deld"),
       "P4": unsupervised_test(p4_df, "P4 delivery only", 3000, "U2_deld")}
u30_m = rows_m[rows_m.use30].reset_index(drop=True)
for key, d_, sd in (("P1", p1_df, 1000), ("P3", arm_df, 2000), ("P4", p4_df, 3000)):
    unsupervised_test(d_, f"{key} pooled-role rank (reported)", sd, "U2d_pooled")
for key, d_, sd in (("P1", u30_m, 1000), ("P3", u30_m[~u30_m.il30 | u30_m.next_arm].reset_index(drop=True), 2000),
                    ("P4", rows_a[rows_a.use30].reset_index(drop=True), 3000)):
    unsupervised_test(d_, f"{key} IL30 (reported)", sd, "U2d", "il30")
for key, d_, sd in (("P1", p1_df, 1000), ("P4", p4_df, 3000)):
    unsupervised_test(d_, f"{key} U1 (reported)", sd, "U1d")
if FAKE:  # a role artefact (placements twice as likely on starter rows) must not pass P1's interval
    rng_r = np.random.default_rng(5)
    art = p1_df.assign(il14=rng_r.random(len(p1_df)) < np.where(p1_df.role.eq(1.0), 0.08, 0.04))
    ra = unsupervised_test(art, "role artefact check", 1000)
    assert ra["lo"] <= 0.5, "a role artefact passes P1's interval"


def rotate_z(df, rng):
    """One k per pitcher-season (pitcher rule); each z column rolls over its defined values by round(k * n_c / n)."""
    out = df.copy()
    z = out[Z14 + Z14S].to_numpy().copy()
    moved = total = 0
    for _, ix in out.groupby(["pitcher", "season"]).indices.items():
        n_def = int(np.isfinite(z[ix]).sum())
        total += n_def
        k = draw_k(len(ix), rng)
        if k is None:
            continue
        moved += n_def
        for c in range(z.shape[1]):
            f = ix[np.isfinite(z[ix, c])]
            if len(f) >= 2:
                z[f, c] = np.roll(z[f, c], int(round(k * len(f) / len(ix))))
    out[Z14 + Z14S] = z
    out.attrs["rot_share"] = moved / max(total, 1)
    return recompute_scores(out)


def model_test(rows, name):
    tr = (rows.season.isin(TRAIN) & rows.use14).to_numpy()
    te = ((rows.season == TEST) & rows.use14).to_numpy()
    y_tr, y_te = rows.il14[tr].to_numpy(), rows.il14[te].to_numpy()
    models = {k: fit(c, rows[tr], y_tr) for k, c in SETS.items()}
    for k, c in SETS.items():
        rows["p_" + k] = np.nan
        rows.loc[te, "p_" + k] = predict(models[k], c, rows[te])
    test = rows[te]
    main = boot(test, y_te, {k: test["p_" + k].to_numpy() for k in SETS} | {"U2": test.U2d.to_numpy(), "U1": test.U1d.to_numpy()},
                [("M2", "B0"), ("M3", "M2"), ("M3", "B0")])
    auc_b0 = main.loc["B0", "AUC"]
    floor, floor_pred, shares = [], [], []
    for i in range(N_FLOOR):
        rz = rotate_z(rows, np.random.default_rng(100 + i))
        shares.append(rz.attrs["rot_share"])
        m = fit(SETS["M2"], rz[tr], y_tr)
        floor_pred.append(predict(m, SETS["M2"], rz[te]))
        floor.append(auc(y_te, floor_pred[-1]) - auc_b0)
    floor = np.array(floor)
    print(f"{name} floor: share of defined z in rotated pitcher-seasons {np.mean(shares):.1%}; "
          f"{len(np.unique(np.round(floor, 8)))} distinct floor values")
    assert np.mean(shares) >= 0.90 or SMOKE, "fewer than 90% of defined z can be rotated"
    test_ = rows[te]
    p_b0, p_m2 = test_.p_B0.to_numpy(), test_.p_M2.to_numpy()
    sub_rows = []
    for sn, m_ in (("starters", test_.role.eq(1.0).to_numpy()), ("relievers", test_.role.eq(0.0).to_numpy()),
                   ("more than 30 days after a return", (test_.days_since_act > 30).to_numpy()),
                   ("more than 30 days after the other level", (test_.days_since_other > 30).to_numpy())):
        ys = y_te[m_]
        sub_rows.append({"subset": sn, "M2 - B0": auc(ys, p_m2[m_]) - auc(ys, p_b0[m_]),
                         "floor mean": np.nanmean([auc(ys, fp[m_]) - auc(ys, p_b0[m_]) for fp in floor_pred]),
                         "positives": int(ys.sum())})
    sub_p2 = pd.DataFrame(sub_rows)
    print(f"{name} subsets (same floor models)\n" + sub_p2.round(4).to_string(index=False))
    u30 = (rows.season.eq(TEST) & rows.use30).to_numpy()
    t30 = rows[u30]
    if int(t30.il30.sum()) > 0:
        print(f"{name} IL30 (IL14-trained models)\n" + boot(t30, t30.il30.to_numpy(), {k: predict(models[k], SETS[k], t30) for k in SETS},
                                                         [("M2", "B0"), ("M3", "M2")]).round(4).to_string())
    gap = main.loc["M2 - B0", "AUC"]
    ok = bool(main.loc["M2 - B0", "lo_bonf"] > 0 and np.all(gap > floor))
    print(f"{name}\n" + main.round(4).to_string())
    print(f"{name}: M2 - B0 {gap:+.4f} [{main.loc['M2 - B0', 'lo_bonf']:+.4f}, {main.loc['M2 - B0', 'hi_bonf']:+.4f}]; "
          f"floor mean {floor.mean():+.4f}, max {floor.max():+.4f} -> {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    arm_m = (~test.il14 | test.next_arm).to_numpy()
    print(f"{name} arm placements only: M2 - B0 {auc(y_te[arm_m], test.p_M2.to_numpy()[arm_m]) - auc(y_te[arm_m], test.p_B0.to_numpy()[arm_m]):+.4f}")
    if FAKE:
        c0 = Z14S.index("z_FB velocity (short)")

        def gap_of(rp):
            ma, mb = fit(SETS["M2"], rp[tr], y_tr), fit(SETS["B0"], rp[tr], y_tr)
            return auc(y_te, predict(ma, SETS["M2"], rp[te])) - auc(y_te, predict(mb, SETS["B0"], rp[te]))

        rate = rows.groupby(["pitcher", "season"]).il14.transform("mean").to_numpy()
        for kind, val in (("timing", np.where(rows.il14, 4.0, 0.0)), ("trait", 8.0 * rate)):
            rp = rows.copy()
            col = Z14S[c0]
            rp[col] = np.where(np.isfinite(rp[col]), val, np.nan)
            rp = recompute_scores(rp)
            real = gap_of(rp)
            fl = [gap_of(rotate_z(rp, np.random.default_rng(100 + i))) for i in range(N_FLOOR)]
            print(f"planted {kind}: real {real:+.4f}, floors mean {np.mean(fl):+.4f}, max {max(fl):+.4f}")
            if kind == "timing":
                assert real > max(fl), "P2 floor cannot be beaten by planted timing"
            else:
                assert np.mean(fl) >= 0.8 * real, "P2 floor does not keep a planted trait"
    return {"main": main, "models": models, "floor": floor, "gap": gap, "ok": ok, "te": te, "y_te": y_te, "test": test, "sub": sub_p2}


P2 = model_test(rows_m, "P2 MLB 2026")
P2A = model_test(rows_a, "Triple-A 2026 (reported)")

# %% [markdown]
# ### Interpretation rules (fixed in the plan)

# %%
notes = {}
for key, res in (("P1", P1), ("P3", P3), ("P4", P4)):
    if not res["ok"]:
        continue
    n_ = []
    sub = SUB[key].set_index("subset")
    if any(sub.loc[s, "AUC"] <= sub.loc[s, "floor mean"] for s in ("more than 30 days after a return", "more than 30 days after the other level")):
        n_.append("timing of a return or level change")
    if DEL[key]["auc"] <= DEL[key]["floor"].mean():
        n_.append("results, not delivery" + (" (a roster move by results)" if key == "P4" else ""))
    if key == "P4":
        n_.append("before a Triple-A placement, not necessarily an injury")
    if n_:
        notes[key] = "; ".join(n_)
imps = {}
for nm, P in (("MLB", P2), ("Triple-A", P2A)):
    pi = permutation_importance(P["models"]["M2"], P["test"][SETS["M2"]].to_numpy(float), P["y_te"], scoring="roc_auc",
                                n_repeats=5, random_state=3)
    imps[nm] = pd.Series(pi.importances_mean, index=SETS["M2"])


def group_of(c):
    if c in CTX:
        return "rival (B0)"
    base = c.replace("z_", "").replace(" (short)", "")
    if base in ("U1", "U2"):
        return "summary scores"
    return "delivery" if base in DELIVERY else "results"


grp = pd.DataFrame({k: v.groupby(v.index.map(group_of)).sum() for k, v in imps.items()})
print(grp.round(4).to_string())
print(imps["MLB"].sort_values(ascending=False).head(10).round(4).to_string())
if P2["ok"]:
    sp = P2["sub"].set_index("subset")
    n_ = []
    if any(sp.loc[s, "M2 - B0"] <= sp.loc[s, "floor mean"] for s in ("more than 30 days after a return", "more than 30 days after the other level")):
        n_.append("timing of a return or level change")
    if grp.loc["delivery", "MLB"] <= 0:
        n_.append("results, not delivery")
    if n_:
        notes["P2"] = "; ".join(n_)
print("interpretation notes:", notes or "none")

# %% [markdown]
# ### Reported alongside: event level and an alarm rule (MLB 2026)

# %%
test = P2["test"]
units = []
for b, g in test.groupby("pitcher"):
    pos_ = g[g.il14]
    for _, u in pos_.groupby(pos_.day + pd.to_timedelta(pos_.days_to_il, unit="D")):
        units.append((b, 1, u.U2d.max(), u.p_M2.max(), u.U2d.mean(), u.p_M2.mean(), len(u)))
    neg = g[~g.il14]
    for _, u in neg.groupby((neg.day - g.day.min()).dt.days // H):
        units.append((b, 0, u.U2d.max(), u.p_M2.max(), u.U2d.mean(), u.p_M2.mean(), len(u)))
units = pd.DataFrame(units, columns=["pitcher", "y", "U2 max", "M2 max", "U2 mean", "M2 mean", "rows"])
print(f"Event level: {int(units.y.sum())} pre-IL units (median {units.rows[units.y == 1].median():.0f} rows), "
      f"{int((units.y == 0).sum())} other 14-day blocks\n"
      + boot(units, units.y.to_numpy(), {k: units[k].to_numpy() for k in ("U2 max", "M2 max", "U2 mean", "M2 mean")}, []).round(4).to_string())
n_day = test.groupby("day").U2.transform("count")
alarm = (test.groupby("day").U2.rank(method="first", ascending=False) <= np.maximum(1, np.floor(0.01 * n_day))).to_numpy()
y14 = P2["y_te"]
il_t = test[test.il14].assign(effective=lambda d: d.day + pd.to_timedelta(d.days_to_il, unit="D"))[["pitcher", "effective"]].drop_duplicates()
caught = [bool(alarm[(test.pitcher == r.pitcher).to_numpy() & (test.day >= r.effective - pd.Timedelta(days=H)).to_numpy()
                     & (test.day <= r.effective).to_numpy()].any()) for _, r in il_t.iterrows()]
print(f"alarm = top 1% of pitchers that day: {int(alarm.sum()):,} alarm days in {TEST}; {y14[alarm].mean():.1%} followed by IL14 "
      f"(base rate {y14.mean():.1%}); {np.mean(caught):.1%} of {len(caught)} placements with pre-IL rows had an alarm in the {H} days before")

# %% [markdown]
# ## Figures

# %%
fig, ax = plt.subplots(figsize=(8, 7))
for k, c, res in (("P1 MLB", "#2471a3", P1), ("P3 MLB arm", "#c0392b", P3), ("P4 Triple-A", "#8a8a8a", P4)):
    fpr, tpr, _ = roc_curve(res["y"], res["s"])
    ax.plot(fpr, tpr, color=c, lw=2.5, label=f"{k}  AUC {res['auc']:.3f}")
ax.plot([0, 1], [0, 1], color="#cccccc", ls="--")
ax.set_xlabel("False alarm rate")
ax.set_ylabel("Share of pre-IL days flagged")
ax.set_title(f"Pitchers: injured list within {H} days")
ax.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig("pitcher_roc.png", dpi=120)
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))
for i, (nm, res) in enumerate((("P1", P1), ("P3", P3), ("P4", P4))):
    ax.scatter(np.full(len(res["floor"]), i), res["floor"], color="#bbbbbb", s=12, label="rotations" if i == 0 else None)
    ax.scatter([i], [res["auc"]], color="#c0392b", s=110, zorder=3, label="real" if i == 0 else None)
ax.axhline(0.5, color="#cccccc", lw=1)
ax.set_xticks([0, 1, 2], ["P1 MLB", "P3 MLB arm", "P4 Triple-A"])
ax.set_ylabel("AUC")
ax.set_title("Real AUC against its rotations")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("pitcher_floors.png", dpi=120)
plt.show()

summary = pd.DataFrame({
    "test": ["P1 MLB U2 daily rank, 2024-2026", "P2 MLB M2 over B0, 2026", "P3 MLB arm, U2 daily rank", "P4 Triple-A U2 daily rank"],
    "statistic": [P1["auc"], P2["gap"], P3["auc"], P4["auc"]],
    "lo_bonf": [P1["lo"], P2["main"].loc["M2 - B0", "lo_bonf"], P3["lo"], P4["lo"]],
    "floor_max": [P1["floor"].max(), P2["floor"].max(), P3["floor"].max(), P4["floor"].max()],
    "verdict": ["SUPPORTED" if v else "NOT SUPPORTED" for v in (P1["ok"], P2["ok"], P3["ok"], P4["ok"])],
    "note": [notes.get(k, "") for k in ("P1", "P2", "P3", "P4")]})
if FAKE:
    print("FAKE LABELS: code test only, no verdict written")
else:
    summary.to_csv("pitcher_verdict.csv", index=False)
    P2["main"].to_csv("pitcher_auc_mlb_2026.csv")
print(summary.to_string(index=False))
