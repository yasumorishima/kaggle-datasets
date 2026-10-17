# %%
import os
os.environ.setdefault("MU_SWAPS", "5")
os.environ.setdefault("MU_BOOT2", "500")

# %% [markdown]
# # Pitch shape x swing shape: development diagnostic D2 (2024 -> 2025, pitch level)
#
# Development only: reads the 2024 and 2025 pitch files and **never opens the 2026 file**
# (a file-access gate enforces it).
# %% [markdown]
# # Pitch shape x swing shape: development run D1 (2024 -> 2025)
#
# Development only: reads the 2024 and 2025 pitch files and **never opens the 2026 file**
# (a file-access gate enforces it). Plan: PLAN_matchup.md v1.
#
# Stage 1, rival R: LightGBM regression on per-PA xRV with interaction constraints that let each
# player's shape interact with his own quality and the context, and the two qualities (including
# own-side shape-based scores) interact with each other, but never one player's shape with the other player.
# Stage 2, test arm T: init_score = R, features = double-centred pairing features X_dc + context only.

# %%
import builtins
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T0 = time.time()
INPUT_ROOT = os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input")
SEASONS = tuple(int(s) for s in os.environ.get("MU_SEASONS", "2024,2025").split(","))
TRAIN_SEASONS = (SEASONS[0],)
EVAL_SEASON = SEASONS[-1]
SMALL = os.environ.get("MU_SMALL", "") == "1"          # smoke test on fake data
N_DECOY = int(os.environ.get("MU_DECOYS", "19"))
N_BOOT = int(os.environ.get("MU_BOOT", "2000"))
NB, NP = 100, 300                                       # prior tracked swings / prior pitches (from D0 counts)
K_SHAPE, K_LOOK = 48, 24
RNG = np.random.default_rng(20261017)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- file-access gate (G1)
ALLOWED = {f"statcast_{y}.parquet" for y in SEASONS}
assert 2026 not in SEASONS


def _guard(path):
    b = os.path.basename(str(path))
    if b not in ALLOWED:
        raise RuntimeError(f"file-access gate: {b} may not be read")


_PF, _RP, _RC, _OPEN = pq.ParquetFile, pd.read_parquet, pd.read_csv, builtins.open


class _GuardedPF(_PF):
    def __init__(self, source, *a, **k):
        _guard(source)
        super().__init__(source, *a, **k)


def _rp(path, *a, **k):
    _guard(path)
    return _RP(path, *a, **k)


def _rc(path, *a, **k):
    _guard(path)
    return _RC(path, *a, **k)


def _open(file, *a, **k):
    if isinstance(file, (str, os.PathLike)) and os.path.abspath(str(file)).startswith(os.path.abspath(INPUT_ROOT)):
        _guard(file)
    return _OPEN(file, *a, **k)


pq.ParquetFile, pd.read_parquet, pd.read_csv, builtins.open = _GuardedPF, _rp, _rc, _open
for planted in ("batter_season.csv", "pitcher_season.csv", "players.csv", "statcast_2026.parquet"):
    p = os.path.join(INPUT_ROOT, "x", planted)
    for reader in (pd.read_csv, pd.read_parquet, pq.ParquetFile, open):
        try:
            reader(p)
        except RuntimeError:
            continue
        except Exception as e:  # noqa: BLE001
            raise AssertionError(f"G1: reading {planted} via {reader} did not hit the gate ({e!r})")
        raise AssertionError(f"G1: reading {planted} via {reader} was allowed")
log("G1 file-access gate refuses the season tables, players.csv and the 2026 file")

# ---------------------------------------------------------------- load
COLS = ["game_pk", "game_date", "at_bat_number", "pitch_number", "batter", "pitcher", "stand", "p_throws",
        "pitch_type", "is_swing", "is_whiff", "in_zone", "zone", "description", "events", "woba_value", "woba_denom",
        "delta_run_exp", "estimated_woba_using_speedangle", "inning", "inning_topbot", "outs_when_up",
        "on_1b", "on_2b", "on_3b", "bat_score", "fld_score", "home_team",
        "bat_speed", "swing_length", "attack_angle", "attack_direction", "swing_path_tilt",
        "intercept_ball_minus_batter_pos_x_inches", "intercept_ball_minus_batter_pos_y_inches",
        "release_speed", "release_pos_x", "release_pos_z", "release_extension", "release_spin_rate", "spin_axis",
        "pfx_x", "pfx_z", "vx0", "vy0", "vz0", "ax", "ay", "az", "arm_angle", "plate_x", "plate_z", "balls", "strikes"]


def find(name):
    p = sorted(glob.glob(os.path.join(INPUT_ROOT, "**", name), recursive=True))
    assert p, f"missing {name}"
    return p[0]


frames = []
for y in SEASONS:
    f = pq.ParquetFile(find(f"statcast_{y}.parquet"))
    miss = [c for c in COLS if c not in f.schema_arrow.names]
    assert not miss, f"{y}: missing {miss}"
    t = f.read(columns=COLS).to_pandas()
    t["season"] = y
    frames.append(t)
d = pd.concat(frames, ignore_index=True)
del frames
assert set(d["season"].unique()) <= set(SEASONS) and 2026 not in set(d["season"])   # G6
d["game_date"] = pd.to_datetime(d["game_date"])
d["date_i"] = (d["game_date"] - pd.Timestamp("2024-01-01")).dt.days.astype(np.int32)
d = d.sort_values(["date_i", "game_pk", "at_bat_number", "pitch_number"], kind="mergesort").reset_index(drop=True)
log("pitches", len(d))

# ---------------------------------------------------------------- per-pitch derived columns
sgn = np.where(d["stand"].to_numpy() == "R", 1.0, -1.0)          # mirror to the batter's frame
d["sgn"] = sgn
d["velo"] = d["release_speed"].astype(float)
d["ivb"] = d["pfx_z"] * 12.0
d["hbr"] = d["pfx_x"] * 12.0                                     # raw (catcher view)
d["hb"] = d["hbr"] * sgn
d["relz"] = d["release_pos_z"].astype(float)
d["relxr"] = d["release_pos_x"].astype(float)
d["relx"] = d["relxr"] * sgn
d["ext"] = d["release_extension"].astype(float)
d["spin"] = d["release_spin_rate"].astype(float)
d["axs"] = np.sin(np.radians(d["spin_axis"].astype(float)))
d["axc"] = np.cos(np.radians(d["spin_axis"].astype(float)))
d["arm"] = d["arm_angle"].astype(float)
yf = 17.0 / 12.0
vy0, ay, vz0, az, vx0, ax = (d[c].astype(float).to_numpy() for c in ("vy0", "ay", "vz0", "az", "vx0", "ax"))
vyf = -np.sqrt(np.maximum(vy0 ** 2 - 2 * ay * (50.0 - yf), 0))
tp = (vyf - vy0) / ay                                            # time from y=50 to the plate
vzf = vz0 + az * tp
d["vaa"] = -np.degrees(np.arctan(vzf / vyf))
t23 = (-vy0 - np.sqrt(np.maximum(vy0 ** 2 - 2 * ay * (50.0 - 23.8), 0))) / ay   # time from y=50 to y=23.8 ft
dt_ = tp - t23
d["x23r"] = d["plate_x"] - (vx0 + ax * t23) * dt_ - 0.5 * ax * dt_ ** 2        # raw x at 23.8 ft
d["z23"] = d["plate_z"] - (vz0 + az * t23) * dt_ - 0.5 * az * dt_ ** 2
d["pxr"] = d["plate_x"].astype(float)
d["pz"] = d["plate_z"].astype(float)

desc = d["description"].astype(str)
d["swing"] = d["is_swing"].astype("boolean").fillna(False).astype(bool)
d["whiff"] = d["is_whiff"].astype("boolean").fillna(False).astype(bool)
d["inz"] = d["in_zone"].astype("boolean").fillna(False).astype(bool)
d["csw"] = d["whiff"] | (desc == "called_strike")
d["bip"] = desc == "hit_into_play"
d["xwb"] = np.where(d["bip"], d["estimated_woba_using_speedangle"].astype(float), np.nan)
d["rv"] = d["delta_run_exp"].astype(float).fillna(0.0)
d["tracked"] = d["swing"] & d["bat_speed"].notna()

SLOT = {"FF": 0, "SI": 1, "FC": 2, "SL": 3, "ST": 4, "CU": 5, "KC": 5, "CS": 5, "SV": 5, "CH": 6, "FS": 7, "FO": 7, "SC": 7}
FAM = {"FF": 0, "SI": 0, "FC": 0, "FA": 0, "SL": 1, "ST": 1, "SV": 1, "CU": 1, "KC": 1, "CS": 1,
       "CH": 2, "FS": 2, "FO": 2, "SC": 2, "KN": 2, "EP": 2}
NSLOT = 8
d["slot"] = d["pitch_type"].map(SLOT).fillna(-1).astype(int)
d["fam"] = d["pitch_type"].map(FAM).fillna(-1).astype(int)
z = d["zone"].astype("Int64").fillna(-1).astype(int).to_numpy()
mir = {1: 3, 3: 1, 4: 6, 6: 4, 7: 9, 9: 7, 11: 12, 12: 11, 13: 14, 14: 13}
zm = np.where(sgn < 0, pd.Series(z).map(lambda v: mir.get(v, v)).to_numpy(), z)
REG = {k: i for i, k in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14])}
d["reg"] = pd.Series(zm).map(REG).fillna(-1).astype(int).to_numpy()
NREG = 13
SHAPE = ["velo", "ivb", "hb", "relz", "relx", "ext", "vaa"]
LOOK = ["relz", "relx", "ext", "arm"]
shape_ok = d[SHAPE].notna().all(axis=1).to_numpy()
look_ok = d[LOOK].notna().all(axis=1).to_numpy()

# ---------------------------------------------------------------- PA table
key = ["game_pk", "at_bat_number"]
first = d.groupby(key, sort=False).head(1)
last = d.groupby(key, sort=False).tail(1)
pa = first[["season", "date_i", "game_pk", "at_bat_number", "batter", "pitcher", "stand", "p_throws", "inning",
            "inning_topbot", "outs_when_up", "on_1b", "on_2b", "on_3b", "bat_score", "fld_score", "home_team"]].copy()
pa["first_row"] = first.index.to_numpy()
lt = last.set_index(key)[["events", "woba_value", "woba_denom", "estimated_woba_using_speedangle"]]
pa = pa.join(lt, on=key)
pa = pa.reset_index(drop=True)
ev = pa["events"].astype(str)
CLS = {"strikeout": 0, "strikeout_double_play": 0, "walk": 1, "hit_by_pitch": 1, "single": 3, "double": 4,
       "triple": 4, "home_run": 5}
pa["cls"] = ev.map(CLS).fillna(2).astype(int)                      # 2 = out on a ball in play (errors, FC included)
pa["counted"] = (pa["woba_denom"].fillna(0) == 1) & pa["events"].notna() & (ev != "intent_walk")
pa["xrv"] = np.where(pa["cls"].isin([0, 1]), pa["woba_value"],
                     pa["estimated_woba_using_speedangle"].where(pa["estimated_woba_using_speedangle"].notna(),
                                                                 pa["woba_value"]))
pa["xrv_from_actual"] = (~pa["cls"].isin([0, 1])) & pa["estimated_woba_using_speedangle"].isna()
# starter = first pitcher of the fielding side in the game; lineup starter = batter whose first PA is among his side's first 9
side = pa["inning_topbot"].astype(str)
grp = pa.groupby(["game_pk", side], sort=False)
pa["starter_p"] = (pa["pitcher"] == grp["pitcher"].transform("first")).astype(int)
pa["side_rank"] = grp.cumcount()
pa["lineup_starter"] = (pa.groupby(["game_pk", side, "batter"])["side_rank"].transform("min") <= 8).astype(int)
gp = pa.groupby(["game_pk", "pitcher"], sort=False)
pa["tto"] = gp.cumcount() // 9 + 1
pa["home_bat"] = (side == "Bot").astype(int)
pa["platoon"] = (pa["stand"] != pa["p_throws"]).astype(int)
pa["standR"] = (pa["stand"] == "R").astype(int)
pa["throwsR"] = (pa["p_throws"] == "R").astype(int)
pa["runners"] = pa["on_1b"].notna().astype(int) + 2 * pa["on_2b"].notna().astype(int) + 4 * pa["on_3b"].notna().astype(int)
pa["score_diff"] = (pa["bat_score"] - pa["fld_score"]).astype(float)
pa["park"] = pa["home_team"].astype("category").cat.codes
pa["month"] = (pd.Timestamp("2024-01-01") + pd.to_timedelta(pa["date_i"], "D")).dt.month
log("PAs", len(pa), "counted", int(pa["counted"].sum()))

# ---------------------------------------------------------------- in-game form (earlier PAs of the same game only)
fb = d["slot"].isin([0, 1]).to_numpy() & d["velo"].notna().to_numpy()
gpp = d.groupby(["game_pk", "pitcher"], sort=False)
d["_fbv"] = np.where(fb, d["velo"], 0.0)
d["_fbn"] = fb.astype(float)
d["_one"] = 1.0
for c in ("_fbv", "_fbn", "_one"):
    d[c + "_ex"] = gpp[c].cumsum() - d[c]                         # pitches strictly before this pitch in the game
gbt = d.groupby(["game_pk", "batter"], sort=False)
tb = d["tracked"].to_numpy()
d["_bs"] = np.where(tb, d["bat_speed"], 0.0)
d["_bn"] = tb.astype(float)
for c in ("_bs", "_bn"):
    d[c + "_ex"] = gbt[c].cumsum() - d[c]
fr = pa["first_row"].to_numpy()
pa["game_pitches_before"] = d["_one_ex"].to_numpy()[fr]
pa["_fbv_today"] = d["_fbv_ex"].to_numpy()[fr] / np.where(d["_fbn_ex"].to_numpy()[fr] > 0, d["_fbn_ex"].to_numpy()[fr], np.nan)
pa["_bs_today"] = d["_bs_ex"].to_numpy()[fr] / np.where(d["_bn_ex"].to_numpy()[fr] > 0, d["_bn_ex"].to_numpy()[fr], np.nan)


# ---------------------------------------------------------------- prior (strictly earlier dates) aggregates
def keyed(player, date):
    return player.astype(np.int64) * 100000 + date.astype(np.int64)


class Prior:
    """Daily sums per (player, date); `look` returns sums over strictly earlier dates."""

    def __init__(self, player, date, ncol):
        k = keyed(np.asarray(player), np.asarray(date))
        self.u, self.inv = np.unique(k, return_inverse=True)
        self.pl = self.u // 100000
        self.M = np.zeros((len(self.u), ncol))
        self.done = False

    def add(self, col, vals, mask=None):
        v = np.asarray(vals, float)
        m = np.isfinite(v) if mask is None else (np.asarray(mask, bool) & np.isfinite(v))
        if np.ndim(col) == 0:
            np.add.at(self.M[:, col], self.inv[m], v[m])
        else:
            col = np.asarray(col)
            np.add.at(self.M, (self.inv[m], col[m]), v[m])

    def finish(self):
        M = self.M
        cs = np.cumsum(M, axis=0)
        start = np.r_[True, self.pl[1:] != self.pl[:-1]]
        sidx = np.maximum.accumulate(np.where(start, np.arange(len(self.pl)), 0))
        before = np.where((sidx > 0)[:, None], cs[np.maximum(sidx - 1, 0)], 0.0)
        self.P = cs - before - M                                  # strictly before the row's date
        self.done = True

    def look(self, player, date):
        assert self.done
        k = keyed(np.asarray(player), np.asarray(date))
        i = np.searchsorted(self.u, k)
        assert np.all(self.u[np.minimum(i, len(self.u) - 1)] == k), "a (player, date) has no daily row"
        return self.P[i]


cnt = pa["counted"].to_numpy()
cls = pa["cls"].to_numpy()
hR = (pa["p_throws"] == "R").to_numpy()
sR = (pa["stand"] == "R").to_numpy()

# batter: PA classes (all, vs R, vs L)
BPA = Prior(pa["batter"], pa["date_i"], 18)
for k in range(6):
    BPA.add(k, (cls == k).astype(float), cnt)
    BPA.add(6 + k, (cls == k).astype(float), cnt & hR)
    BPA.add(12 + k, (cls == k).astype(float), cnt & ~hR)
BPA.finish()
# pitcher: PA classes (all, vs R batters, vs L batters)
PPA = Prior(pa["pitcher"], pa["date_i"], 18)
for k in range(6):
    PPA.add(k, (cls == k).astype(float), cnt)
    PPA.add(6 + k, (cls == k).astype(float), cnt & sR)
    PPA.add(12 + k, (cls == k).astype(float), cnt & ~sR)
PPA.finish()
log("PA priors done")

# batter pitch-level
TRK = ["bat_speed", "swing_length", "attack_angle", "attack_direction", "swing_path_tilt",
       "intercept_ball_minus_batter_pos_x_inches", "intercept_ball_minus_batter_pos_y_inches"]
for c in ("attack_direction", "intercept_ball_minus_batter_pos_x_inches"):
    d[c + "_m"] = d[c] * sgn
TRKM = ["bat_speed", "swing_length", "attack_angle", "attack_direction_m", "swing_path_tilt",
        "intercept_ball_minus_batter_pos_x_inches_m", "intercept_ball_minus_batter_pos_y_inches"]
B_BASIC = ["n", "swing", "whiff", "csw", "chase", "ozone", "zswing", "zcontact", "zn", "bip", "xw", "rv"]
nb0 = len(B_BASIC)
B = Prior(d["batter"], d["date_i"], nb0 + 9 + 1 + 14 + 2 * NREG)
sw, wh, iz = d["swing"].to_numpy(), d["whiff"].to_numpy(), d["inz"].to_numpy()
one = np.ones(len(d))
for i, (vals, m) in enumerate([(one, None), (sw, None), (wh, None), (d["csw"].to_numpy(), None), (sw & ~iz, None),
                               (~iz, None), (sw & iz, None), (sw & iz & ~wh, None), (iz, None),
                               (d["bip"].to_numpy(), None), (d["xwb"].to_numpy(), d["bip"].to_numpy()),
                               (d["rv"].to_numpy(), None)]):
    B.add(i, np.asarray(vals, float), m)
fam = d["fam"].to_numpy()
for f_ in range(3):
    B.add(nb0 + 3 * f_ + 0, (fam == f_).astype(float))
    B.add(nb0 + 3 * f_ + 1, ((fam == f_) & sw).astype(float))
    B.add(nb0 + 3 * f_ + 2, ((fam == f_) & wh).astype(float))
o_trk = nb0 + 9
B.add(o_trk, tb.astype(float))
for j, c in enumerate(TRKM):
    v = d[c].to_numpy(float)
    B.add(o_trk + 1 + 2 * j, v, tb)
    B.add(o_trk + 2 + 2 * j, v ** 2, tb)
o_reg = o_trk + 15
reg = d["reg"].to_numpy()
okr = reg >= 0
B.add(np.where(okr, o_reg + reg, 0), d["rv"].to_numpy(), okr)
B.add(np.where(okr, o_reg + NREG + reg, 0), one, okr)
B.finish()
log("batter pitch priors done")

# shape clusters (label-free, fitted on the first training season, frozen)
from sklearn.cluster import MiniBatchKMeans  # noqa: E402

fit_rows = shape_ok & (d["season"].to_numpy() == TRAIN_SEASONS[0])
Xs = d.loc[fit_rows, SHAPE].to_numpy(float)
SH_MU, SH_SD = Xs.mean(0), Xs.std(0)
km = MiniBatchKMeans(n_clusters=K_SHAPE, random_state=7, n_init=5, batch_size=8192)
idx = RNG.choice(len(Xs), size=min(len(Xs), 300000), replace=False)
km.fit((Xs[idx] - SH_MU) / SH_SD)
C_SH = km.cluster_centers_
zs_all = (np.nan_to_num(d[SHAPE].to_numpy(float)) - SH_MU) / SH_SD
lab_sh = np.where(shape_ok, km.predict(zs_all), -1)
nn = np.sqrt(((((Xs[idx[:20000]] - SH_MU) / SH_SD)[:, None, :] - C_SH[None]) ** 2).sum(-1)).min(1)
H_SH = float(np.median(nn))
Xl = d.loc[look_ok & (d["season"].to_numpy() == TRAIN_SEASONS[0]), LOOK].to_numpy(float)
LK_MU, LK_SD = Xl.mean(0), Xl.std(0)
kl = MiniBatchKMeans(n_clusters=K_LOOK, random_state=7, n_init=5, batch_size=8192)
il = RNG.choice(len(Xl), size=min(len(Xl), 300000), replace=False)
kl.fit((Xl[il] - LK_MU) / LK_SD)
C_LK = kl.cluster_centers_
zl_all = (np.nan_to_num(d[LOOK].to_numpy(float)) - LK_MU) / LK_SD
lab_lk = np.where(look_ok, kl.predict(zl_all), -1)
nl = np.sqrt(((((Xl[il[:20000]] - LK_MU) / LK_SD)[:, None, :] - C_LK[None]) ** 2).sum(-1)).min(1)
H_LK = float(np.median(nl))
log("clusters", K_SHAPE, "shape bandwidth", round(H_SH, 3), "|", K_LOOK, "look bandwidth", round(H_LK, 3))

# batter per shape cluster: n, swing, whiff, csw, rv, bip, xw ; per look cluster: n
BC = Prior(d["batter"], d["date_i"], 7 * K_SHAPE + K_LOOK)
okc = lab_sh >= 0
for j, (vals, m) in enumerate([(one, None), (sw, None), (wh, None), (d["csw"].to_numpy(), None),
                               (d["rv"].to_numpy(), None), (d["bip"].to_numpy(), None),
                               (d["xwb"].to_numpy(), d["bip"].to_numpy())]):
    mm = okc if m is None else (okc & m)
    BC.add(np.where(okc, j * K_SHAPE + lab_sh, 0), np.asarray(vals, float), mm)
okl = lab_lk >= 0
BC.add(np.where(okl, 7 * K_SHAPE + lab_lk, 0), one, okl)
BC.finish()
log("batter cluster priors done")

# pitcher per slot: n + sums of shape and tunnel quantities (raw frame), usage and location per batter side
PSUM = ["velo", "ivb", "hbr", "spin", "axs", "axc", "vaa", "relz", "relxr", "ext", "arm", "x23r", "z23", "pxr", "pz"]
NPS = 1 + len(PSUM)
P = Prior(d["pitcher"], d["date_i"], NSLOT * NPS + 2 * NSLOT + 2 * NREG + 5)
slot = d["slot"].to_numpy()
oks = slot >= 0
for j, c in enumerate([None] + PSUM):
    vals = one if c is None else d[c].to_numpy(float)
    P.add(np.where(oks, slot * NPS + j, 0), vals, oks)
o_use = NSLOT * NPS
P.add(np.where(oks, o_use + slot + NSLOT * (~(d["stand"].to_numpy() == "R")).astype(int), 0), one, oks)
o_loc = o_use + 2 * NSLOT
P.add(np.where(okr, o_loc + reg + NREG * (~(d["stand"].to_numpy() == "R")).astype(int), 0), one, okr)
o_rel = o_loc + 2 * NREG
for j, (vals, m) in enumerate([(one, None), (d["relz"].to_numpy(float), None), (d["relz"].to_numpy(float) ** 2, None),
                               (d["relxr"].to_numpy(float), None), (d["relxr"].to_numpy(float) ** 2, None)]):
    P.add(o_rel + j, vals, d["relz"].notna().to_numpy() & d["relxr"].notna().to_numpy())
P.finish()
log("pitcher priors done")

# %% [markdown]
# ## Features per plate appearance

# %%
is_train = pa["season"].isin(TRAIN_SEASONS).to_numpy()
tr_cnt = is_train & cnt
LG = np.bincount(cls[tr_cnt], minlength=6) / tr_cnt.sum()          # frozen from training seasons
KAPPA_PA = 200.0
bpa = BPA.look(pa["batter"], pa["date_i"])
ppa = PPA.look(pa["pitcher"], pa["date_i"])


def shrink6(c6):
    n = c6.sum(1, keepdims=True)
    return (c6 + KAPPA_PA * LG) / (n + KAPPA_PA), n[:, 0]


F = {}
r_all, n_all = shrink6(bpa[:, 0:6])
r_vs, n_vs = shrink6(np.where(hR[:, None], bpa[:, 6:12], bpa[:, 12:18]))
for k in range(6):
    F[f"bq_all{k}"], F[f"bq_vs{k}"] = r_all[:, k], r_vs[:, k]
F["bq_n"], F["bq_nvs"] = n_all, n_vs
r_all, n_all = shrink6(ppa[:, 0:6])
r_vs, n_vs = shrink6(np.where(sR[:, None], ppa[:, 6:12], ppa[:, 12:18]))
for k in range(6):
    F[f"pq_all{k}"], F[f"pq_vs{k}"] = r_all[:, k], r_vs[:, k]
F["pq_n"], F["pq_nvs"] = n_all, n_vs
BQ = [c for c in F if c.startswith("bq_")]
PQ = [c for c in F if c.startswith("pq_")]

# league constants for the batter pitch rates (training seasons, frozen)
btr = B.M[np.isin(B.pl, pa.loc[is_train, "batter"].unique())]
trd = d["season"].isin(TRAIN_SEASONS).to_numpy()
LGB = {
    "swing": sw[trd].mean(), "whiff": wh[trd & sw].mean(), "csw": d["csw"].to_numpy()[trd].mean(),
    "chase": sw[trd & ~iz].mean(), "zswing": sw[trd & iz].mean(), "zcontact": (~wh)[trd & sw & iz].mean(),
    "xw": np.nanmean(d["xwb"].to_numpy()[trd]), "rv": d["rv"].to_numpy()[trd].mean(),
}


def rate(num, den, prior, k):
    return (num + k * prior) / (den + k)


bb = B.look(pa["batter"], pa["date_i"])
g = {nm: bb[:, i] for i, nm in enumerate(B_BASIC)}
F["bs_n"] = g["n"]
F["bs_swing"] = rate(g["swing"], g["n"], LGB["swing"], 50)
F["bs_whiff"] = rate(g["whiff"], g["swing"], LGB["whiff"], 50)
F["bs_csw"] = rate(g["csw"], g["n"], LGB["csw"], 50)
F["bs_chase"] = rate(g["chase"], g["ozone"], LGB["chase"], 50)
F["bs_zswing"] = rate(g["zswing"], g["zn"], LGB["zswing"], 50)
F["bs_zcontact"] = rate(g["zcontact"], g["zswing"], LGB["zcontact"], 50)
F["bs_xwcon"] = rate(g["xw"], g["bip"], LGB["xw"], 30)
F["bs_rv"] = rate(g["rv"], g["n"], LGB["rv"], 100)
for f_ in range(3):
    n_f, s_f, w_f = bb[:, nb0 + 3 * f_], bb[:, nb0 + 3 * f_ + 1], bb[:, nb0 + 3 * f_ + 2]
    F[f"bs_fam{f_}_swing"] = rate(s_f, n_f, LGB["swing"], 50)
    F[f"bs_fam{f_}_whiff"] = rate(w_f, s_f, LGB["whiff"], 50)
ntrk = bb[:, o_trk]
F["bs_ntrk"] = ntrk
with np.errstate(invalid="ignore", divide="ignore"):
    for j, c in enumerate(TRKM):
        s1, s2 = bb[:, o_trk + 1 + 2 * j], bb[:, o_trk + 2 + 2 * j]
        m = np.where(ntrk >= 20, s1 / ntrk, np.nan)
        F[f"bs_{c}_mean"] = m
        F[f"bs_{c}_sd"] = np.where(ntrk >= 20, np.sqrt(np.maximum(s2 / ntrk - m ** 2, 0)), np.nan)
    F["bs_bat_speed_today"] = pa["_bs_today"].to_numpy() - F["bs_bat_speed_mean"]
rv_b = F["bs_rv"]
RDEV = np.stack([rate(bb[:, o_reg + r], bb[:, o_reg + NREG + r], rv_b, 50) - rv_b for r in range(NREG)], 1)
for r in range(NREG):
    F[f"bs_reg{r}"] = RDEV[:, r]
BS = [c for c in F if c.startswith("bs_")]
log("BQ PQ BS done")


# pitcher-side vectors for any (pitcher, date, batter side)
def pitcher_side(pid, date, sg):
    pp = P.look(pid, date)
    n_s = pp[:, [s * NPS for s in range(NSLOT)]]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = {c: pp[:, [s * NPS + 1 + j for s in range(NSLOT)]] / n_s for j, c in enumerate(PSUM)}
    for c in ("hbr", "relxr", "x23r", "pxr"):
        mean[c] = mean[c] * sg[:, None]                             # mirror to this batter side
    use_raw = np.where((sg > 0)[:, None], pp[:, o_use:o_use + NSLOT], pp[:, o_use + NSLOT:o_use + 2 * NSLOT])
    valid = n_s >= 10
    use = np.where(valid, use_raw, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        use = use / use.sum(1, keepdims=True)
    loc_raw = np.where((sg > 0)[:, None], pp[:, o_loc:o_loc + NREG], pp[:, o_loc + NREG:o_loc + 2 * NREG])
    loc = (loc_raw + 5.0 * LOC_LG) / (loc_raw.sum(1, keepdims=True) + 5.0)
    return pp, n_s, mean, use, valid, loc


LOC_LG = np.bincount(reg[trd & okr], minlength=NREG) / (trd & okr).sum()
LOOK_LG = np.bincount(lab_lk[trd & okl], minlength=K_LOOK) / (trd & okl).sum()


def soft(zv, centers, h):
    d2 = ((zv[:, None, :] - centers[None]) ** 2).sum(-1)
    w = np.exp(-(d2 - d2.min(1, keepdims=True)) / (2 * h * h))
    return w / w.sum(1, keepdims=True)


def pitcher_vectors(pid, date, sg):
    """a (shape-cluster profile of the arsenal), look assignment, location share, usage-weighted VAA."""
    pp, n_s, mean, use, valid, loc = pitcher_side(pid, date, sg)
    n = len(pid)
    a = np.zeros((n, K_SHAPE))
    for s in range(NSLOT):
        sv = np.stack([mean["velo"][:, s], mean["ivb"][:, s], mean["hbr"][:, s], mean["relz"][:, s],
                       mean["relxr"][:, s], mean["ext"][:, s], mean["vaa"][:, s]], 1)
        ok = valid[:, s] & np.isfinite(sv).all(1)
        if ok.any():
            a[ok] += use[ok, s][:, None] * soft((sv[ok] - SH_MU) / SH_SD, C_SH, H_SH)
    has = valid.any(1) & np.isfinite(use).all(1)
    a[~has] = np.nan
    tot = n_s.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        lk = np.stack([(np.nansum(np.where(valid, mean[c] * n_s, 0), 1) / np.where(valid, n_s, 0).sum(1))
                       for c in ("relz", "relxr", "ext", "arm")], 1)
    okl_ = np.isfinite(lk).all(1)
    blook = np.full((n, K_LOOK), np.nan)
    blook[okl_] = soft((lk[okl_] - LK_MU) / LK_SD, C_LK, H_LK)
    vaa_u = np.nansum(np.where(valid, use * mean["vaa"], 0), 1)
    vaa_u[~has] = np.nan
    return a, blook, loc, vaa_u, (pp, n_s, mean, use, valid, tot)


sg_pa = np.where(sR, 1.0, -1.0)
A_real, BL_real, LOC_real, VAA_real, extra = pitcher_vectors(pa["pitcher"].to_numpy(), pa["date_i"].to_numpy(), sg_pa)
pp, n_s, mean, use, valid, tot = extra
F["ps_n"] = tot
for s in range(NSLOT):
    F[f"ps_use{s}"] = use[:, s]
    for c in ("velo", "ivb", "hbr", "spin", "vaa", "relz", "relxr", "ext", "arm"):
        F[f"ps_{c}{s}"] = np.where(valid[:, s], mean[c][:, s], np.nan)
    F[f"ps_axis{s}"] = np.where(valid[:, s], np.degrees(np.arctan2(mean["axs"][:, s], mean["axc"][:, s])), np.nan)
fbslots = np.where(valid[:, :3], use[:, :3], -1.0)
pfb = fbslots.argmax(1)
hasfb = fbslots.max(1) > 0
ii = np.arange(len(pa))


def at(c):
    return np.where(hasfb, mean[c][ii, pfb], np.nan)


nonfb = valid.copy()
nonfb[ii, pfb] = False
w_o = np.where(nonfb, use, 0.0)
with np.errstate(invalid="ignore", divide="ignore"):
    w_o = w_o / w_o.sum(1, keepdims=True)

    def wmean(x):
        return np.nansum(w_o * x, 1) / np.where(np.isfinite(w_o).all(1) & (w_o.sum(1) > 0), 1.0, np.nan)

    F["ps_fb_velo"] = at("velo")
    F["ps_velo_gap"] = at("velo") - wmean(mean["velo"])
    F["ps_move_sep"] = wmean(np.sqrt((mean["ivb"] - at("ivb")[:, None]) ** 2 + (mean["hbr"] - at("hbr")[:, None]) ** 2))
    tun = np.sqrt((mean["x23r"] - at("x23r")[:, None]) ** 2 + (mean["z23"] - at("z23")[:, None]) ** 2)
    plt_ = np.sqrt((mean["pxr"] - at("pxr")[:, None]) ** 2 + (mean["pz"] - at("pz")[:, None]) ** 2)
    F["ps_tunnel23"] = wmean(tun)
    F["ps_plate_sep"] = wmean(plt_)
    F["ps_tunnel_ratio"] = wmean(plt_ / np.maximum(tun, 0.05))
    rn = pp[:, o_rel]
    F["ps_relz_sd"] = np.sqrt(np.maximum(pp[:, o_rel + 2] / rn - (pp[:, o_rel + 1] / rn) ** 2, 0))
    F["ps_relx_sd"] = np.sqrt(np.maximum(pp[:, o_rel + 4] / rn - (pp[:, o_rel + 3] / rn) ** 2, 0))
    fbv_prior = (pp[:, 0 * NPS + 1] + pp[:, 1 * NPS + 1]) / (pp[:, 0] + pp[:, NPS])
F["ps_form_fbv"] = pa["_fbv_today"].to_numpy() - fbv_prior
F["ps_game_pitches"] = pa["game_pitches_before"].to_numpy()
for r in range(NREG):
    F[f"ps_loc{r}"] = LOC_real[:, r]
PS = [c for c in F if c.startswith("ps_")]
CX = ["standR", "throwsR", "platoon", "home_bat", "park", "inning", "outs_when_up", "runners", "score_diff", "tto",
      "starter_p", "month"]
for c in CX:
    F[c] = pa[c].to_numpy(float)
FE = pd.DataFrame(F)
log("features", FE.shape)

# ---------------------------------------------------------------- batter vectors for the pairing features
bc = BC.look(pa["batter"], pa["date_i"])
nC, swC, whC, cswC, rvC, bipC, xwC = (bc[:, j * K_SHAPE:(j + 1) * K_SHAPE] for j in range(7))
D_w = rate(whC, swC, F["bs_whiff"][:, None], 30) - F["bs_whiff"][:, None]
D_c = rate(cswC, nC, F["bs_csw"][:, None], 50) - F["bs_csw"][:, None]
D_r = rate(rvC, nC, F["bs_rv"][:, None], 100) - F["bs_rv"][:, None]
D_x = rate(xwC, bipC, F["bs_xwcon"][:, None], 20) - F["bs_xwcon"][:, None]
nL = bc[:, 7 * K_SHAPE:7 * K_SHAPE + K_LOOK]
S_look = (nL + 5.0 * LOOK_LG) / (nL.sum(1, keepdims=True) + 5.0)
AA = F["bs_attack_angle_mean"]

elig = (cnt & (ntrk >= NB) & (tot >= NP) & np.isfinite(A_real).all(1) & np.isfinite(BL_real).all(1)
        & np.isfinite(VAA_real) & np.isfinite(AA))
log("eligible PAs", int(elig.sum()), "by season", pa.loc[elig, "season"].value_counts().sort_index().to_dict(),
    "| xRV from woba_value for missing xwOBA:", int((elig & pa["xrv_from_actual"].to_numpy()).sum()))
grp_key = (pa["date_i"].to_numpy().astype(np.int64) * 4 + 2 * hR + sR)


def centred_dot(Pv, Bv, mask):
    """sum_c (Pv - mean_g Pv)(Bv - mean_g Bv) with group means over eligible PAs of the same date, hand, side."""
    out = np.full(len(Pv), np.nan)
    mask = mask & np.isfinite(Pv).all(1) & np.isfinite(Bv).all(1)
    gk = grp_key[mask]
    u, inv = np.unique(gk, return_inverse=True)
    cnt_g = np.bincount(inv).astype(float)

    def gmean(M):
        S = np.zeros((len(u), M.shape[1]))
        np.add.at(S, inv, M)
        return (S / cnt_g[:, None])[inv]

    Pm, Bm = Pv[mask], Bv[mask]
    out[mask] = ((Pm - gmean(Pm)) * (Bm - gmean(Bm))).sum(1)
    return out


def pairing_features(A, BL, LOC, VAA):
    X = {}
    for nm, D in (("x1_whiff", D_w), ("x1_csw", D_c), ("x1_rv", D_r), ("x1_xw", D_x)):
        X[nm] = centred_dot(A, D, elig)
    X["x2_aa_vaa"] = 2.0 * centred_dot(VAA[:, None], AA[:, None], elig)
    X["x3_look"] = centred_dot(BL, S_look, elig)
    X["x4_loc"] = centred_dot(LOC, RDEV, elig)
    return pd.DataFrame(X)


XR = pairing_features(A_real, BL_real, LOC_real, VAA_real)
XN = list(XR.columns)
log("pairing features", XN)

# %% [markdown]
# ## D2: does the batter's swing shape interact with the pitch's shape and location, pitch by pitch?
#
# Development only (fit 2024, evaluate 2025). Three LightGBM models on the same rows and features, differing only in
# which features may interact (interaction constraints):
# * **A** additive: {pitch shape + location + count} and {count + batter profile} never meet;
# * **Q** quality interaction: the pitch may interact with the batter's *quality* rates (swing, chase, zone swing,
#   zone contact, whiff), but not with his *swing shape* (bat speed, swing length, attack angle, attack direction,
#   swing path tilt, intercept point x/y);
# * **S** full: everything may interact.
# The shape-specific interaction is loss(Q) - loss(S) on 2025 pitches. Floor: the swing-shape block is swapped
# between batters who batted the same day, from the same side and the same tercile of prior whiff rate (quality kept,
# shape detached from the batter), and Q and S are refitted on that world.
# The batter profile uses dates strictly before the pitch's date only.

# %%
import lightgbm as lgb  # noqa: E402

SL5 = SHAPE + ["pxm", "pz"]
d["pxm"] = d["pxr"] * sgn
CNT = ["balls", "strikes"]
for c_ in CNT:
    d[c_] = d[c_].astype(float)
ok5 = (d[SL5].notna().all(axis=1) & d[CNT].notna().all(axis=1)).to_numpy()
SPN = ["s_bat_speed", "s_swing_length", "s_attack_angle", "s_attack_dir", "s_tilt", "s_int_x", "s_int_y"]
QPN = ["q_swing", "q_chase", "q_zswing", "q_zcontact", "q_whiff"]
N_SWAP = int(os.environ.get("MU_SWAPS", "5"))
N_BOOT2 = int(os.environ.get("MU_BOOT2", "500"))


def profile_at(batter, date):
    """Swing shape (7) then quality rates (5), from dates strictly before `date`; also the tracked-swing count."""
    bb_ = B.look(batter, date)
    g_ = {nm: bb_[:, i] for i, nm in enumerate(B_BASIC)}
    nt = bb_[:, o_trk]
    cols = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for j in range(len(TRKM)):
            cols.append(np.where(nt >= 20, bb_[:, o_trk + 1 + 2 * j] / nt, np.nan))
    cols += [rate(g_["swing"], g_["n"], LGB["swing"], 50), rate(g_["chase"], g_["ozone"], LGB["chase"], 50),
             rate(g_["zswing"], g_["zn"], LGB["zswing"], 50), rate(g_["zcontact"], g_["zswing"], LGB["zcontact"], 50),
             rate(g_["whiff"], g_["swing"], LGB["whiff"], 50)]
    return np.column_stack(cols), nt


PROF_ALL, NT_ALL = profile_at(d["batter"].to_numpy(), d["date_i"].to_numpy())
use = ok5 & (NT_ALL >= NB) & np.isfinite(PROF_ALL).all(1) & d["season"].isin(SEASONS).to_numpy()
seas = d["season"].to_numpy()
tr_rows = np.where(use & (seas == TRAIN_SEASONS[0]))[0]
ev_rows = np.where(use & (seas == EVAL_SEASON))[0]
log("D2 rows: train", len(tr_rows), "eval", len(ev_rows))

# G9: the profile is built from strictly earlier dates (brute-force recomputation on sampled pitches)
bat_all, date_all = d["batter"].to_numpy(), d["date_i"].to_numpy()
tbv = d["tracked"].to_numpy()
for i_ in RNG.choice(np.concatenate([tr_rows, ev_rows]), size=60, replace=False):
    m_ = (bat_all == bat_all[i_]) & (date_all < date_all[i_]) & tbv
    assert m_.sum() == NT_ALL[i_], "G9: tracked-swing count"
    for j, c in enumerate(TRKM):
        assert abs(np.nansum(d[c].to_numpy(float)[m_]) / m_.sum() - PROF_ALL[i_, j]) < 1e-6, f"G9: {c}"
    n_ = ((bat_all == bat_all[i_]) & (date_all < date_all[i_])).sum()
    sw_ = (((bat_all == bat_all[i_]) & (date_all < date_all[i_])) & sw).sum()
    assert abs((sw_ + 50 * LGB["swing"]) / (n_ + 50) - PROF_ALL[i_, 7]) < 1e-9, "G9: swing rate"
log("G9 profile recomputation passed on 60 pitches")
print("tracked swings with a missing value (the profile divides by all tracked swings):",
      {c: round(float(np.isnan(d[c].to_numpy(float)[tbv]).mean()), 5) for c in TRKM})

FEAT = SL5 + CNT + QPN + SPN
iP = list(range(len(SL5)))
iC = [len(SL5) + k for k in range(len(CNT))]
iQ = [len(SL5) + len(CNT) + k for k in range(len(QPN))]
iS = [len(SL5) + len(CNT) + len(QPN) + k for k in range(len(SPN))]
CONS2 = {"A": [iP + iC, iC + iQ + iS], "Q": [iP + iC + iQ, iC + iQ + iS], "S": None}


def design(prof):
    return np.column_stack([d[SL5].to_numpy(float), d[CNT].to_numpy(float), prof[:, 7:12], prof[:, 0:7]])


OUT = {
    "whiff|swing": (sw, wh.astype(float), "binary"),
    "xwOBA|contact": (d["bip"].to_numpy() & np.isfinite(d["xwb"].to_numpy()), d["xwb"].to_numpy(), "regression"),
    "run value": (np.ones(len(d), bool), d["rv"].to_numpy(), "regression"),
}
P2 = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, seed=5)
MAXR = 60 if SMALL else 3000
dtr = date_all[tr_rows]
vcut = np.quantile(dtr, 0.8)


def fit_one(X, y_, rows_fit, kind, cons):
    """Fit on the first 80% of the training season's dates, early-stop on the last 20%."""
    a_ = rows_fit[date_all[rows_fit] < vcut]
    b_ = rows_fit[date_all[rows_fit] >= vcut]
    prm = {**P2, "objective": kind}
    if cons is not None:
        prm["interaction_constraints"] = cons
    m = lgb.train(prm, lgb.Dataset(X[a_], y_[a_]), MAXR, valid_sets=[lgb.Dataset(X[b_], y_[b_])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def loss(p, y_, kind):
    if kind == "binary":
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y_ * np.log(p) + (1 - y_) * np.log(1 - p))
    return (y_ - p) ** 2


def paths2(node, acc, outl):
    if "split_feature" in node:
        a2 = acc | {node["split_feature"]}
        paths2(node["left_child"], a2, outl)
        paths2(node["right_child"], a2, outl)
    else:
        outl.append(acc)


def crosses(model, groups):
    bad = 0
    for t in model.dump_model()["tree_info"]:
        pl_ = []
        paths2(t["tree_structure"], set(), pl_)
        bad += sum(1 for p_ in pl_ if not any(p_ <= set(g_) for g_ in groups))
    return bad


def run_world(prof, tag, do_A):
    X = design(prof)
    res = {}
    for nm, (msk, y_, kind) in OUT.items():
        rf = tr_rows[msk[tr_rows]]
        re_ = ev_rows[msk[ev_rows]]
        mods = {k: fit_one(X, y_, rf, kind, CONS2[k]) for k in (("A", "Q", "S") if do_A else ("Q", "S"))}
        for k in mods:
            if CONS2[k] is not None:
                assert crosses(mods[k], CONS2[k]) == 0, f"G11: model {k} ({nm}, {tag}) crosses its constraint sets"
        pr = {k: mods[k].predict(X[re_]) for k in mods}
        res[nm] = dict(rows=re_, y=y_[re_], kind=kind, pred=pr,
                       loss={k: loss(pr[k], y_[re_], kind) for k in pr}, iters={k: mods[k].best_iteration for k in mods})
    return res


# G11 probe: the path check catches an unconstrained model
_Xp = design(PROF_ALL)
_rf = tr_rows[sw[tr_rows]][:20000]
_probe = lgb.train({**P2, "objective": "binary"}, lgb.Dataset(_Xp[_rf], wh[_rf].astype(float)), 30)
assert crosses(_probe, CONS2["Q"]) > 0, "G11: the path check did not catch an unconstrained model"
del _Xp
log("G11 probe: an unconstrained model is caught")

REAL = run_world(PROF_ALL, "real", True)
log("real world fitted")

# shape swap between batters of the same date, side and prior-whiff tercile
stand_all = d["stand"].to_numpy()
urow = np.where(use)[0]
ukey = pd.DataFrame({"b": bat_all[urow], "dt": date_all[urow], "st": stand_all[urow]})
units = ukey.drop_duplicates().reset_index(drop=True)
first_row = urow[ukey.drop_duplicates().index.to_numpy()]
units["qw"] = PROF_ALL[first_row, 11]
cuts = np.quantile(units["qw"], [1 / 3, 2 / 3])
units["ter"] = np.searchsorted(cuts, units["qw"].to_numpy())
SP_unit = PROF_ALL[first_row, 0:7]
row_unit = pd.MultiIndex.from_frame(units[["b", "dt", "st"]]).get_indexer(pd.MultiIndex.from_frame(ukey))
assert (row_unit >= 0).all()
# within a unit (batter, date, side) the profile is constant by construction
assert np.allclose(PROF_ALL[urow], PROF_ALL[first_row][row_unit]), "unit profile"
grp = units.groupby(["dt", "st", "ter"]).ngroup().to_numpy()


def swap_map(rng):
    tgt = np.arange(len(units))
    order = np.lexsort((rng.random(len(units)), grp))
    g_sorted = grp[order]
    starts = np.r_[0, np.where(np.diff(g_sorted) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(order)]
    for s_, e_ in zip(starts, ends):
        if e_ - s_ >= 2:
            tgt[order[s_:e_]] = np.roll(order[s_:e_], 1)          # cyclic shift: no unit keeps its own shape
    return tgt


SW_RNG = np.random.default_rng(31)
swap_res = []
for r in range(N_SWAP):
    tgt = swap_map(SW_RNG)
    moved = tgt != np.arange(len(units))
    # G10: a moved unit takes the shape of a different batter from the same date, side and tercile
    assert np.all(units["b"].to_numpy()[tgt[moved]] != units["b"].to_numpy()[moved]), "G10: same batter"
    for c_ in ("dt", "st", "ter"):
        assert np.all(units[c_].to_numpy()[tgt] == units[c_].to_numpy()), f"G10: stratum ({c_})"
    assert moved.mean() >= 0.9, f"G10: only {moved.mean():.3f} of the batter-days were swapped"
    assert np.abs(SP_unit[tgt][moved] - SP_unit[moved]).sum(1).min() > 0, "G10: a swapped shape equals the real one"
    prof_s = PROF_ALL.copy()
    prof_s[urow, 0:7] = SP_unit[tgt][row_unit]
    swap_res.append(run_world(prof_s, f"swap{r}", False))
    log(f"swap world {r + 1}/{N_SWAP}: units moved {moved.mean():.3f}")

# statistics
gm_all = d["game_pk"].to_numpy()
pit_all = d["pitcher"].to_numpy()
BR = np.random.default_rng(57)
print("\nD2 results on 2025 pitches (models fitted on 2024; positive = the second model predicts better)")
for nm in OUT:
    R0 = REAL[nm]
    rows = R0["rows"]
    bi = pd.factorize(bat_all[rows])[0]
    pi = pd.factorize(pit_all[rows])[0]
    gi = pd.factorize(gm_all[rows])[0]
    dQS = R0["loss"]["Q"] - R0["loss"]["S"]
    dAQ = R0["loss"]["A"] - R0["loss"]["Q"]
    bs_qs, bs_aq = [], []
    for _ in range(N_BOOT2):
        w = (BR.exponential(size=bi.max() + 1)[bi] * BR.exponential(size=pi.max() + 1)[pi]
             * BR.exponential(size=gi.max() + 1)[gi])
        bs_qs.append((w * dQS).sum() / w.sum())
        bs_aq.append((w * dAQ).sum() / w.sum())
    sw_qs = np.array([S_[nm]["loss"]["Q"].mean() - S_[nm]["loss"]["S"].mean() for S_ in swap_res])
    p_sw = (1 + (sw_qs >= dQS.mean()).sum()) / (1 + len(sw_qs))
    base = R0["loss"]["Q"].mean()
    print(f"[{nm}] n {len(rows)} | mean loss A {R0['loss']['A'].mean():.6f} Q {base:.6f} S {R0['loss']['S'].mean():.6f}"
          f" | rounds {R0['iters']}")
    print(f"  quality x pitch  (A-Q): {dAQ.mean():+.3e}  95% [{np.quantile(bs_aq, .025):+.3e}, {np.quantile(bs_aq, .975):+.3e}]"
          f"  ({dAQ.mean() / R0['loss']['A'].mean() * 100:+.3f}% of A)")
    print(f"  SHAPE x pitch    (Q-S): {dQS.mean():+.3e}  95% [{np.quantile(bs_qs, .025):+.3e}, {np.quantile(bs_qs, .975):+.3e}]"
          f"  ({dQS.mean() / base * 100:+.3f}% of Q) | swap floor {np.round(sw_qs, 8).tolist()} p {p_sw:.3f}")
    # direction: does S's adjustment over Q point toward the outcome?
    adj = R0["pred"]["S"] - R0["pred"]["Q"]
    res_q = R0["y"] - R0["pred"]["Q"]
    slope = np.polyfit(adj, res_q, 1)[0] if adj.std() > 0 else np.nan
    print(f"  sd of S-Q adjustment {adj.std():.5f}; slope of (y - Q) on (S - Q) {slope:.3f}; "
          f"corr {np.corrcoef(adj, res_q)[0, 1]:+.4f}")
    for S_ in swap_res:
        a2 = S_[nm]["pred"]["S"] - S_[nm]["pred"]["Q"]
        print(f"    swap: sd of S-Q {a2.std():.5f}, corr with y-Q {np.corrcoef(a2, S_[nm]['y'] - S_[nm]['pred']['Q'])[0, 1]:+.4f}")
log("D2 done")
