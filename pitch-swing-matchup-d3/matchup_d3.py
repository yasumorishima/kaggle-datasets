# %%
import os
os.environ.setdefault("MU_DONORS", "19")
os.environ.setdefault("MU_DONORS_RAW", "9")
os.environ.setdefault("MU_SEEDS", "5")
os.environ.setdefault("MU_LEVEL_W", "40")
os.environ.setdefault("MU_LEVEL_X", "30")
os.environ.setdefault("MU_PLANT_K", "9")
os.environ.setdefault("MU_BOOT2", "500")

# %% [markdown]
# # Pitch shape x swing shape: development diagnostic D3 (2024 -> 2025, pitch level)
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
# ## D3: is the pitch x swing-shape gain a mechanism, an identifier, or a better quality measure?
#
# Development only (fit 2024, evaluate 2025). Changes from D2, after the plan audit:
# * **Q+ rival**: the pitch may interact with a much richer batter quality block (the 5 D2 rates plus CSW, xwOBA on
#   contact, run value per pitch, swing and whiff rate by pitch family, run value by zone region, and a shape-implied
#   whiff / xwOBA-on-contact score), so S can beat Q+ only through swing geometry beyond its quality projection.
# * **Swing shape residualised**: each swing's metric minus a league model of that metric on the pitch's location,
#   velocity, movement and count (training season only), averaged per batter over earlier dates with per-metric
#   non-missing counts. The raw-metric version (also per-metric counts) is reported beside it.
# * Pitch block adds spin rate, spin axis and arm angle; context adds pitcher hand and platoon.
# * **Donor floor**: every batter receives one other batter's whole shape trajectory (same prior-whiff tercile and
#   first-eligible-date bucket), fixed over all dates and both seasons: an identifier survives, the link between a
#   batter's own swing and his own response does not.
# * Seeds, a level check with labels drawn from Q+ (no shape x pitch term), two planted effects (a shape-free
#   per-batter zone effect that must fail; an attack-angle x VAA effect that must pass), and batters unseen in 2024.

# %%
import lightgbm as lgb  # noqa: E402

SL5 = SHAPE + ["pxm", "pz"]
d["pxm"] = d["pxr"] * sgn
d["arm_f"] = d["arm_angle"].astype(float)
PIT = SL5 + ["spin", "axs", "axc", "arm_f"]
d["p_R"] = (d["p_throws"] == "R").astype(float)
d["platoon"] = (d["p_throws"] != d["stand"]).astype(float)
for c_ in ("balls", "strikes"):
    d[c_] = d[c_].astype(float)
CNT = ["balls", "strikes", "p_R", "platoon"]
ok5 = (d[SL5].notna().all(axis=1) & d[["balls", "strikes"]].notna().all(axis=1)).to_numpy()
EV_ = lambda k, v: int(os.environ.get(k, v))  # noqa: E731
N_DON = EV_("MU_DONORS", "19")
N_DON_RAW = EV_("MU_DONORS_RAW", "9")
N_SEED = EV_("MU_SEEDS", "5")
N_LEV_W = EV_("MU_LEVEL_W", "40")
N_LEV_X = EV_("MU_LEVEL_X", "30")
N_PL = EV_("MU_PLANT_K", "9")
N_BOOT2 = EV_("MU_BOOT2", "500")
BETA_PL = float(os.environ.get("MU_BETA", "0.15"))
GAMMA_PL = float(os.environ.get("MU_GAMMA", "0.3"))
seas = d["season"].to_numpy()
bat_all, date_all = d["batter"].to_numpy(), d["date_i"].to_numpy()
tbv = d["tracked"].to_numpy()

# ---------------------------------------------------------------- residualised swing metrics
XLc = ["pxm", "pz", "velo", "ivb", "hb", "vaa", "balls", "strikes"]
XL = d[XLc].to_numpy(float)
LP = dict(objective="regression", learning_rate=0.1, num_leaves=31, min_data_in_leaf=1000, verbose=-1,
          num_threads=4, seed=3)
RESM = np.full((len(d), len(TRKM)), np.nan)
RAWM = np.full((len(d), len(TRKM)), np.nan)
for j, c in enumerate(TRKM):
    v = d[c].to_numpy(float)
    okj = tbv & np.isfinite(v)
    fitm = okj & trd & np.isfinite(XL).all(1)
    mj = lgb.train(LP, lgb.Dataset(XL[fitm], v[fitm]), 40 if SMALL else 300)
    RAWM[okj, j] = v[okj]
    RESM[okj, j] = v[okj] - mj.predict(XL[okj])
    print(f"league model of {c}: sd raw {np.nanstd(v[fitm]):.3f}, sd residual {np.nanstd(RESM[fitm, j]):.3f}")
print("tracked swings with a missing metric by season (now divided by per-metric counts):",
      {int(s): np.round(np.isnan(RAWM[tbv & (seas == s)]).mean(0), 5).tolist() for s in SEASONS})


def metric_prior(M_):
    P_ = Prior(d["batter"], d["date_i"], 2 * len(TRKM))
    for j in range(len(TRKM)):
        okj = np.isfinite(M_[:, j])
        P_.add(j, M_[:, j], okj)
        P_.add(len(TRKM) + j, okj.astype(float))
    P_.finish()
    return P_


BP_RES, BP_RAW = metric_prior(RESM), metric_prior(RAWM)
log("swing-metric priors done (residual and raw, per-metric counts)")


def look_asof(P_, player, date):
    """Sums over the player's dates strictly before `date`, whether or not he played on `date`."""
    player, date = np.asarray(player), np.asarray(date)
    k = keyed(player, date)
    n_ = len(P_.u)
    i = np.searchsorted(P_.u, k, side="left")
    ic = np.minimum(i, n_ - 1)
    same = (i < n_) & (P_.pl[ic] == player)
    jc = np.maximum(i - 1, 0)
    prev = (i >= 1) & (P_.pl[jc] == player)
    return np.where(same[:, None], P_.P[ic], np.where(prev[:, None], P_.P[jc] + P_.M[jc], 0.0))


def shape_of(S_, mincnt=20):
    nt_ = S_[:, len(TRKM):]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(nt_ >= mincnt, S_[:, :len(TRKM)] / nt_, np.nan)


SH = {"res": shape_of(BP_RES.look(bat_all, date_all)), "raw": shape_of(BP_RAW.look(bat_all, date_all))}

# G12: look_asof equals look on played dates, and a brute-force sum on dates the batter did not play
_ii = RNG.choice(len(d), size=200, replace=False)
assert np.allclose(look_asof(BP_RES, bat_all[_ii], date_all[_ii]), BP_RES.look(bat_all[_ii], date_all[_ii])), "G12"
_n_after = _n_gap = 0
for _b in RNG.choice(np.unique(bat_all), 40, replace=False):
    _db = np.unique(date_all[bat_all == _b])
    _gap = np.setdiff1d(np.arange(_db[0], _db[-1]), _db)
    # always test a date after his last game, a date he did not play inside his span, and a random date
    _ts = [_db[-1] + 1 + int(RNG.integers(0, 10)), int(RNG.integers(date_all.min(), date_all.max() + 5))]
    if len(_gap):
        _ts.append(int(RNG.choice(_gap)))
        _n_gap += 1
    _n_after += 1
    for _t in _ts:
        _m = (bat_all == _b) & (date_all < _t) & np.isfinite(RESM[:, 2])
        _s = look_asof(BP_RES, np.array([_b]), np.array([_t]))[0]
        assert _s[len(TRKM) + 2] == _m.sum() and abs(_s[2] - RESM[_m, 2].sum()) < 1e-6, "G12: brute force"
log(f"G12 as-of lookup passed (after-last-game dates {_n_after}, non-played dates inside the span {_n_gap})")

# ---------------------------------------------------------------- batter quality block (Q+)
bb_all = B.look(bat_all, date_all)
NT_ALL = bb_all[:, o_trk]


def quality_block(bb_):
    g_ = {nm: bb_[:, i] for i, nm in enumerate(B_BASIC)}
    rv0 = rate(g_["rv"], g_["n"], LGB["rv"], 50)
    cols = [rate(g_["swing"], g_["n"], LGB["swing"], 50), rate(g_["chase"], g_["ozone"], LGB["chase"], 50),
            rate(g_["zswing"], g_["zn"], LGB["zswing"], 50), rate(g_["zcontact"], g_["zswing"], LGB["zcontact"], 50),
            rate(g_["whiff"], g_["swing"], LGB["whiff"], 50),
            rate(g_["csw"], g_["n"], LGB["csw"], 50), rate(g_["xw"], g_["bip"], LGB["xw"], 50), rv0]
    for f_ in range(3):
        n_f, s_f, w_f = bb_[:, nb0 + 3 * f_], bb_[:, nb0 + 3 * f_ + 1], bb_[:, nb0 + 3 * f_ + 2]
        cols += [rate(s_f, n_f, LGB["swing"], 50), rate(w_f, s_f, LGB["whiff"], 50)]
    for r_ in range(NREG):
        cols.append(rate(bb_[:, o_reg + r_], bb_[:, o_reg + NREG + r_], rv0, 50) - rv0)
    return np.column_stack(cols)


QB_ALL = quality_block(bb_all)
NQ = QB_ALL.shape[1]
use = (ok5 & (NT_ALL >= NB) & np.isfinite(SH["res"]).all(1) & np.isfinite(SH["raw"]).all(1)
       & np.isfinite(QB_ALL).all(1) & d["season"].isin(SEASONS).to_numpy())
urow = np.where(use)[0]
L = len(urow)
tr_l = np.where(seas[urow] == TRAIN_SEASONS[0])[0]          # local indices into urow
ev_l = np.where(seas[urow] == EVAL_SEASON)[0]
ub, ud = bat_all[urow], date_all[urow]
log("D3 rows: train", len(tr_l), "eval", len(ev_l), "| quality columns", NQ)

# G9b: per-metric profile = brute-force mean of earlier-date residuals with per-metric counts
for i_ in RNG.choice(urow, size=60, replace=False):
    m_ = (bat_all == bat_all[i_]) & (date_all < date_all[i_])
    for j in range(len(TRKM)):
        mj_ = m_ & np.isfinite(RESM[:, j])
        assert abs(RESM[mj_, j].mean() - SH["res"][i_, j]) < 1e-6, f"G9b: {TRKM[j]}"
        mr_ = m_ & np.isfinite(RAWM[:, j])
        assert abs(RAWM[mr_, j].mean() - SH["raw"][i_, j]) < 1e-6, f"G9b raw: {TRKM[j]}"
log("G9b profile recomputation passed on 60 pitches")

# shape-implied quality score: batter-day units, 7 shape means -> prior whiff rate and prior xwOBA on contact,
# fitted on training-season units only and applied to every row (and to donor shapes in donor worlds)
ukey = pd.DataFrame({"b": ub, "dt": ud})
first_l = ukey.drop_duplicates().index.to_numpy()
unit_tr = first_l[seas[urow][first_l] == TRAIN_SEASONS[0]]
SP_ = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, verbose=-1,
           num_threads=4, seed=7)
SCORE_M = {}
for v_ in ("res", "raw"):
    Xs = SH[v_][urow]
    SCORE_M[v_] = [lgb.train(SP_, lgb.Dataset(Xs[unit_tr], QB_ALL[urow][unit_tr, k]), 30 if SMALL else 200)
                   for k in (4, 6)]                                  # whiff rate, xwOBA on contact


def score_of(v_, shp):
    return np.column_stack([m.predict(shp) for m in SCORE_M[v_]])


# ---------------------------------------------------------------- design and models
PITX = d[PIT].to_numpy(float)[urow]
CNTX = d[CNT].to_numpy(float)[urow]
QBX = QB_ALL[urow]
nP, nC, nQ, nS = len(PIT), len(CNT), NQ + 2, len(TRKM)
iP = list(range(nP))
iC = [nP + k for k in range(nC)]
iQ = [nP + nC + k for k in range(nQ)]
iS = [nP + nC + nQ + k for k in range(nS)]
CONS = {"A": [iP + iC, iC + iQ + iS], "Q": [iP + iC + iQ, iC + iQ + iS], "S": None}


def design(v_, shp):
    return np.column_stack([PITX, CNTX, QBX, score_of(v_, shp), shp]).astype(np.float32)


sw_u, wh_u = sw[urow], wh[urow].astype(float)
bip_u = d["bip"].to_numpy()[urow]
xw_u = d["xwb"].to_numpy()[urow]
rv_u = d["rv"].to_numpy()[urow]
OUT = {"whiff|swing": (sw_u, wh_u, "binary"), "xwOBA|contact": (bip_u & np.isfinite(xw_u), xw_u, "regression"),
       "run value": (np.ones(L, bool), rv_u, "regression")}
P2 = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
MAXR = 60 if SMALL else 3000
vcut = np.quantile(ud[tr_l], 0.8)


def fit_one(X, y_, rows_fit, kind, cons, seed=5):
    a_ = rows_fit[ud[rows_fit] < vcut]
    b_ = rows_fit[ud[rows_fit] >= vcut]
    prm = {**P2, "objective": kind, "seed": seed}
    if cons is not None:
        prm["interaction_constraints"] = cons
    return lgb.train(prm, lgb.Dataset(X[a_], y_[a_]), MAXR, valid_sets=[lgb.Dataset(X[b_], y_[b_])],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


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


def run_world(X, outs, arms=("Q", "S"), seed=5, ylab=None):
    """Fit the arms on 2024 rows, return per-row 2025 losses. `ylab` overrides an outcome's labels (synthetic)."""
    res = {}
    for nm in outs:
        msk, y_, kind = OUT[nm]
        if ylab is not None and nm in ylab:
            y_ = ylab[nm]
        rf = tr_l[msk[tr_l]]
        re_ = ev_l[msk[ev_l]]
        mods = {k: fit_one(X, y_, rf, kind, CONS[k], seed) for k in arms}
        for k in mods:
            if CONS[k] is not None:
                assert crosses(mods[k], CONS[k]) == 0, f"G11: model {k} ({nm}) crosses its constraint sets"
        pr = {k: mods[k].predict(X[re_]) for k in mods}
        res[nm] = dict(rows=re_, y=y_[re_], pred=pr, loss={k: loss(pr[k], y_[re_], kind) for k in pr},
                       iters={k: mods[k].best_iteration for k in mods})
    return res


# G11 probe
_X = design("res", SH["res"][urow])
_rf = tr_l[sw_u[tr_l]][:20000]
_probe = lgb.train({**P2, "objective": "binary"}, lgb.Dataset(_X[_rf], wh_u[_rf]), 30)
assert crosses(_probe, CONS["Q"]) > 0, "G11: the path check did not catch an unconstrained model"
del _probe
log("G11 probe: an unconstrained model is caught")

gi_ = pd.factorize(d["game_pk"].to_numpy()[urow])[0]
bi_ = pd.factorize(ub)[0]
pi_ = pd.factorize(d["pitcher"].to_numpy()[urow])[0]
BR = np.random.default_rng(57)


def boot(vals, rows, nb=None):
    nb = N_BOOT2 if nb is None else nb
    b_, p_, g_ = pd.factorize(bi_[rows])[0], pd.factorize(pi_[rows])[0], pd.factorize(gi_[rows])[0]
    out = np.empty(nb)
    for t in range(nb):
        w = (BR.exponential(size=b_.max() + 1)[b_] * BR.exponential(size=p_.max() + 1)[p_]
             * BR.exponential(size=g_.max() + 1)[g_])
        out[t] = (w * vals).sum() / w.sum()
    return out


def qs(world, nm, sub=None):
    R0 = world[nm]
    dv = R0["loss"]["Q"] - R0["loss"]["S"]
    if sub is not None:
        dv = dv[sub(R0["rows"])]
    return dv


# ---------------------------------------------------------------- real worlds (both shape versions, seeds)
X_REAL = {v_: design(v_, SH[v_][urow]) for v_ in ("res", "raw")}
REAL = {v_: run_world(X_REAL[v_], list(OUT), arms=("A", "Q", "S") if v_ == "res" else ("Q", "S"))
        for v_ in ("res", "raw")}
log("real worlds fitted (seed 5)")
SEED_QS = {v_: {nm: [] for nm in ("whiff|swing", "xwOBA|contact")} for v_ in ("res", "raw")}
for s_ in range(1, N_SEED):
    for v_ in ("res", "raw"):
        w_ = run_world(X_REAL[v_], ["whiff|swing", "xwOBA|contact"], seed=5 + s_)
        for nm in SEED_QS[v_]:
            SEED_QS[v_][nm].append(qs(w_, nm).mean())
log("seed replicates done")

# unseen batters: no row of any kind in the training season
seen_b = np.unique(bat_all[seas == TRAIN_SEASONS[0]])
unseen_l = ~np.isin(ub, seen_b)
SUB_UNSEEN = lambda rows: unseen_l[rows]  # noqa: E731

# ---------------------------------------------------------------- donor floor
bt = pd.DataFrame({"b": ub, "dt": ud, "qw": QBX[:, 4], "s": seas[urow]})
first_b = bt.sort_values(["b", "dt"]).groupby("b", sort=True).first()
tr_first = first_b["s"].to_numpy() == TRAIN_SEASONS[0]
qcut = np.quantile(first_b["qw"].to_numpy()[tr_first], [1 / 3, 2 / 3])
first_b["ter"] = np.searchsorted(qcut, first_b["qw"].to_numpy())
first_b["fe"] = 0
for s_ in SEASONS:
    m_ = first_b["s"].to_numpy() == s_
    if m_.sum() >= 5:
        first_b.loc[m_, "fe"] = np.searchsorted(np.quantile(first_b["dt"].to_numpy()[m_], [.2, .4, .6, .8]),
                                                first_b["dt"].to_numpy()[m_]) + 10 * (s_ - SEASONS[0])
BLIST = first_b.index.to_numpy()
bgrp = first_b.groupby(["ter", "fe"]).ngroup().to_numpy()
FILL = {v_: SH[v_][urow][tr_l].mean(0) for v_ in ("res", "raw")}


def donor_map(rng):
    tgt = np.arange(len(BLIST))
    order = np.lexsort((rng.random(len(BLIST)), bgrp))
    gs = bgrp[order]
    starts = np.r_[0, np.where(np.diff(gs) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(order)]
    for s_, e_ in zip(starts, ends):
        if e_ - s_ >= 2:
            tgt[order[s_:e_]] = np.roll(order[s_:e_], 1)
    return tgt


TER_B = first_b["ter"].to_numpy()
FE_B = first_b["fe"].to_numpy()
TER_ROW = TER_B[np.searchsorted(BLIST, ub)]
FE_ROW = FE_B[np.searchsorted(BLIST, ub)]


def donor_shape(v_, tgt):
    donor = BLIST[tgt][np.searchsorted(BLIST, ub)]
    # G13 on the array actually used: another batter, same tercile and first-eligible bucket, fixed over dates
    dpos = np.searchsorted(BLIST, donor)
    assert np.all(BLIST[dpos] == donor), "G13: donor not a listed batter"
    assert np.all(TER_B[dpos] == TER_ROW) and np.all(FE_B[dpos] == FE_ROW), "G13: stratum"
    assert pd.Series(donor).groupby(ub).nunique().max() == 1, "G13: donor not fixed over dates"
    moved_b = pd.Series(donor != ub).groupby(ub).first()
    assert moved_b.mean() >= 0.9, f"G13: only {moved_b.mean():.3f} of batters moved"
    shp = shape_of(look_asof(BP_RES if v_ == "res" else BP_RAW, donor, ud))
    bad = ~np.isfinite(shp).all(1)
    shp[bad] = FILL[v_]
    return shp, donor, bad.mean()


DON_RNG = np.random.default_rng(31)
DMAPS = [donor_map(DON_RNG) for _ in range(max(N_DON, N_DON_RAW, N_PL))]
DON = {"res": [], "raw": []}
for v_, k_ in (("res", N_DON), ("raw", N_DON_RAW)):
    for r_ in range(k_):
        shp, donor, fill_frac = donor_shape(v_, DMAPS[r_])
        same_rows = (donor == ub).mean()
        assert same_rows <= 0.1, f"G13: {same_rows:.3f} of rows keep their own batter's shape"
        DON[v_].append(run_world(design(v_, shp), ["whiff|swing", "xwOBA|contact"]))
        log(f"donor world {v_} {r_ + 1}/{k_}: rows filled with the mean shape {fill_frac:.3f}, own shape {same_rows:.3f}")

# ---------------------------------------------------------------- report
print("\nD3 results on 2025 pitches (models fitted on 2024; positive = S predicts better than Q+)")
for v_ in ("res", "raw"):
    for nm in OUT:
        R0 = REAL[v_][nm]
        dv = qs(REAL[v_], nm)
        bs = boot(dv, R0["rows"])
        line = (f"[{v_} | {nm}] n {len(dv)} | loss Q+ {R0['loss']['Q'].mean():.6f} S {R0['loss']['S'].mean():.6f} | "
                f"rounds {R0['iters']} | Q+ - S {dv.mean():+.3e} 95% [{np.quantile(bs, .025):+.3e}, "
                f"{np.quantile(bs, .975):+.3e}] ({dv.mean() / R0['loss']['Q'].mean() * 100:+.3f}%)")
        print(line)
        if "A" in R0["loss"]:
            da = R0["loss"]["A"] - R0["loss"]["Q"]
            ba = boot(da, R0["rows"])
            print(f"    quality+ x pitch (A - Q+) {da.mean():+.3e} 95% [{np.quantile(ba, .025):+.3e}, "
                  f"{np.quantile(ba, .975):+.3e}]")
        if nm in DON[v_][0]:
            fl = np.array([qs(W_, nm).mean() for W_ in DON[v_]])
            p_ = (1 + (fl >= dv.mean()).sum()) / (1 + len(fl))
            print(f"    donor floor (n {len(fl)}) mean {fl.mean():+.3e} sd {fl.std():+.3e} max {fl.max():+.3e} | p {p_:.3f}")
            sd_seed = np.std([dv.mean()] + SEED_QS[v_][nm], ddof=1) if N_SEED > 1 else np.nan
            print(f"    seeds: Q+ - S over {N_SEED} seeds {np.round([dv.mean()] + SEED_QS[v_][nm], 7).tolist()} "
                  f"sd {sd_seed:.3e}")
            adj = R0["pred"]["S"] - R0["pred"]["Q"]
            rq = R0["y"] - R0["pred"]["Q"]
            cd = [np.corrcoef(W_[nm]["pred"]["S"] - W_[nm]["pred"]["Q"], W_[nm]["y"] - W_[nm]["pred"]["Q"])[0, 1]
                  for W_ in DON[v_]]
            print(f"    corr(S - Q+, y - Q+) {np.corrcoef(adj, rq)[0, 1]:+.4f} | donor worlds "
                  f"{np.min(cd):+.4f}..{np.max(cd):+.4f}")
            um = SUB_UNSEEN(R0["rows"])
            if um.sum() > 100:
                du = dv[um]
                bu = boot(du, R0["rows"][um])
                flu = np.array([qs(W_, nm, SUB_UNSEEN).mean() for W_ in DON[v_]])
                print(f"    unseen batters: n {int(um.sum())} ({len(np.unique(ub[R0['rows'][um]]))} batters) | "
                      f"Q+ - S {du.mean():+.3e} 95% [{np.quantile(bu, .025):+.3e}, {np.quantile(bu, .975):+.3e}] | "
                      f"donor p {(1 + (flu >= du.mean()).sum()) / (1 + len(flu)):.3f}")

# ---------------------------------------------------------------- level check (labels with no shape x pitch term)
XR_ = X_REAL["res"]
LV = np.random.default_rng(77)
for nm, nset in (("whiff|swing", N_LEV_W), ("xwOBA|contact", N_LEV_X)):
    msk, y_, kind = OUT[nm]
    rf = tr_l[msk[tr_l]]
    mQ = fit_one(XR_, y_, rf, kind, CONS["Q"])
    allr = np.where(msk)[0]
    p0 = np.zeros(L)
    p0[allr] = mQ.predict(XR_[allr])
    if kind != "binary":
        resid = (y_ - p0)[rf]
    passes, means = 0, []
    for t in range(nset):
        ys = np.full(L, np.nan)
        if kind == "binary":
            ys[allr] = (LV.random(len(allr)) < p0[allr]).astype(float)
        else:
            ys[allr] = p0[allr] + LV.choice(resid, size=len(allr))
        w_ = run_world(XR_, [nm], ylab={nm: ys})
        dvl = qs(w_, nm)
        lb = np.quantile(boot(dvl, w_[nm]["rows"], nb=300), 0.025)
        passes += lb > 0
        means.append(dvl.mean())
    if nset:
        print(f"level [{nm}]: bootstrap lower bound > 0 in {passes}/{nset} null sets; null Q+ - S mean "
              f"{np.mean(means):+.3e} sd {np.std(means):.3e}")
log("level check done")

# ---------------------------------------------------------------- planted effects (whiff, residualised shape)
msk, y_, kind = OUT["whiff|swing"]
rf = tr_l[msk[tr_l]]
mQ = fit_one(XR_, y_, rf, kind, CONS["Q"])
allr = np.where(msk)[0]
lg0 = np.zeros(L)
lg0[allr] = np.log(np.clip(mQ.predict(XR_[allr]), 1e-6, 1 - 1e-6) / np.clip(1 - mQ.predict(XR_[allr]), 1e-6, 1))
aa = SH["res"][urow][:, 2]
za = (aa - aa[tr_l].mean()) / aa[tr_l].std()
vv = PITX[:, PIT.index("vaa")]
zv = (vv - vv[tr_l].mean()) / vv[tr_l].std()
regu = d["reg"].to_numpy()[urow]
HB = np.random.default_rng(91).normal(size=(len(BLIST), NREG))
hot = np.where(regu >= 0, HB[np.searchsorted(BLIST, ub), np.maximum(regu, 0)], 0.0)
PL_RNG = np.random.default_rng(93)
for nm_pl, eff, should in (("attack angle x VAA", BETA_PL * za * zv, "PASS"),
                           ("per-batter zone effect, no shape", GAMMA_PL * hot, "FAIL")):
    ys = np.full(L, np.nan)
    ys[allr] = (PL_RNG.random(len(allr)) < 1 / (1 + np.exp(-(lg0 + eff)[allr]))).astype(float)
    w_ = run_world(XR_, ["whiff|swing"], ylab={"whiff|swing": ys})
    dvp = qs(w_, "whiff|swing")
    lb = np.quantile(boot(dvp, w_["whiff|swing"]["rows"]), 0.025)
    fl = []
    for r_ in range(N_PL):
        shp, _, _ = donor_shape("res", DMAPS[r_])
        fl.append(qs(run_world(design("res", shp), ["whiff|swing"], ylab={"whiff|swing": ys}), "whiff|swing").mean())
    fl = np.array(fl)
    p_ = (1 + (fl >= dvp.mean()).sum()) / (1 + len(fl))
    verdict = "PASS" if (lb > 0 and p_ <= 0.1) else "FAIL"
    print(f"planted [{nm_pl}]: Q+ - S {dvp.mean():+.3e} lower {lb:+.3e} | donor max {fl.max():+.3e} p {p_:.3f} | "
          f"criterion {verdict} (should {should})")
log("D3 done")
