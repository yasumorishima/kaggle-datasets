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
        "pfx_x", "pfx_z", "vx0", "vy0", "vz0", "ax", "ay", "az", "arm_angle", "plate_x", "plate_z"]


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
# ## Gates G2 / G3: brute-force recomputation of sampled features from earlier pitches only

# %%
def close(a_, b_, what):
    a_, b_ = float(a_), float(b_)
    if not (np.isnan(a_) and np.isnan(b_)) and not np.isclose(a_, b_, rtol=1e-7, atol=1e-9):
        raise AssertionError(f"{what}: vectorized {a_!r} vs brute force {b_!r}")


samp = RNG.choice(np.where(elig)[0], size=min(120, int(elig.sum())), replace=False)
D_ = d[["batter", "pitcher", "date_i", "game_pk", "at_bat_number", "stand", "slot", "velo", "tracked", "bat_speed",
        "whiff", "reg"]].copy()
D_["lab"] = lab_sh
for i in samp:
    b, p, dt, gk, ab = (int(pa.at[i, c]) for c in ("batter", "pitcher", "date_i", "game_pk", "at_bat_number"))
    st = pa.at[i, "stand"]
    pb = pa[(pa["batter"] == b) & (pa["date_i"] < dt) & pa["counted"]]
    close(bpa[i, 0], (pb["cls"] == 0).sum(), "G2 batter strikeouts before")
    db = D_[(D_["batter"] == b) & (D_["date_i"] < dt)]
    if ntrk[i] >= 20:
        close(F["bs_bat_speed_mean"][i], db.loc[db["tracked"], "bat_speed"].mean(), "G2 batter bat speed")
    c = int(np.argmax(nC[i]))
    close(whC[i, c], (db["whiff"] & (db["lab"] == c)).sum(), "G2 batter whiffs in a shape cluster")
    dpp = D_[(D_["pitcher"] == p) & (D_["date_i"] < dt)]
    for s in range(NSLOT):
        ns = (dpp["slot"] == s).sum()
        close(n_s[i, s], ns, "G2 pitcher pitches of a slot")
        if ns >= 10:
            close(mean["velo"][i, s], dpp.loc[dpp["slot"] == s, "velo"].mean(), "G2 pitcher slot velocity")
    vs = dpp[(dpp["stand"] == st) & (dpp["slot"] >= 0)]
    cnts = np.array([((vs["slot"] == s) & valid[i, s]).sum() for s in range(NSLOT)], float)
    if cnts.sum() > 0:
        for s in range(NSLOT):
            close(use[i, s], cnts[s] / cnts.sum(), "G2 pitcher usage against this side")
    g_ = D_[(D_["game_pk"] == gk) & (D_["pitcher"] == p) & (D_["at_bat_number"] < ab)]
    close(pa.at[i, "game_pitches_before"], len(g_), "G3 pitches earlier in the game")
    fbm = g_.loc[g_["slot"].isin([0, 1]), "velo"].mean()
    close(pa.at[i, "_fbv_today"], fbm, "G3 fastball velocity earlier in the game")
    gb = D_[(D_["game_pk"] == gk) & (D_["batter"] == b) & (D_["at_bat_number"] < ab) & D_["tracked"]]
    close(pa.at[i, "_bs_today"], gb["bat_speed"].mean(), "G3 batter bat speed earlier in the game")
# one centring group recomputed directly
gsel = pd.Series(grp_key[elig]).value_counts()
gk0 = gsel.index[0]
rows = np.where(elig & (grp_key == gk0))[0]
Am, Dm = A_real[rows], D_r[rows]
direct = ((Am - Am.mean(0)) * (Dm - Dm.mean(0))).sum(1)
for j, r_ in enumerate(rows[:20]):
    close(XR.at[r_, "x1_rv"], direct[j], "G2 centred pairing feature")
log("G2/G3 brute-force recomputation passed on", len(samp), "PAs")

# %% [markdown]
# ## Stage 1 (rival R) and stage 2 (pairing arm T)

# %%
import lightgbm as lgb  # noqa: E402

y = pa["xrv"].to_numpy(float)
yact = pa["woba_value"].to_numpy(float)
tr_idx = np.where(elig & is_train)[0]
ev_idx = np.where(elig & (pa["season"].to_numpy() == EVAL_SEASON))[0]
log("train rows", len(tr_idx), "eval rows", len(ev_idx))
BASE = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
SMALLP = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=4)
MAXR = 60 if SMALL else 3000
SEEDS = (1, 2, 3)


def date_blocks(idx, k):
    dd = pa["date_i"].to_numpy()[idx]
    qs = np.quantile(dd, np.linspace(0, 1, k + 1))
    b = np.clip(np.searchsorted(qs, dd, side="right") - 1, 0, k - 1)
    return b


def choose_rounds(Xtr, ytr, params, init=None):
    """Rehearsed rule: first 70% of training dates -> early stopping on the last 30%, then scale by size."""
    dd = pa["date_i"].to_numpy()[tr_idx]
    cut = np.quantile(dd, 0.7)
    a_, b_ = dd <= cut, dd > cut
    ds = lgb.Dataset(Xtr[a_], ytr[a_], init_score=None if init is None else init[a_], free_raw_data=False)
    dv = lgb.Dataset(Xtr[b_], ytr[b_], init_score=None if init is None else init[b_], reference=ds)
    m = lgb.train({**params, "seed": 1}, ds, MAXR, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return max(10, int(round(m.best_iteration * a_.size / a_.sum())))


def fit_pred(Xtr, ytr, Xte, params, rounds, init=None, constraints=None):
    out = np.zeros(len(Xte))
    for s in SEEDS:
        p = {**params, "seed": s}
        if constraints is not None:
            p["interaction_constraints"] = constraints
        m = lgb.train(p, lgb.Dataset(Xtr, ytr, init_score=init), rounds)
        out += m.predict(Xte) / len(SEEDS)
    return out, m


def oof(Xtr, ytr, params, rounds, init=None, constraints=None, k=5):
    b = date_blocks(tr_idx, k)
    o = np.zeros(len(ytr))
    for j in range(k):
        te = b == j
        o[te], _ = fit_pred(Xtr[~te], ytr[~te], Xtr[te], params, max(10, int(rounds * (~te).mean())),
                            None if init is None else init[~te], constraints)
    return o


FEc = FE.copy()
cols_b = CX + BQ + BS
cols_p = CX + PQ + PS
ytr = y[tr_idx]
Xb_tr, Xb_ev = FEc[cols_b].to_numpy()[tr_idx], FEc[cols_b].to_numpy()[ev_idx]
Xp_tr, Xp_ev = FEc[cols_p].to_numpy()[tr_idx], FEc[cols_p].to_numpy()[ev_idx]
rb = choose_rounds(Xb_tr, ytr, BASE)
rp = choose_rounds(Xp_tr, ytr, BASE)
bscore_tr = oof(Xb_tr, ytr, BASE, rb)
pscore_tr = oof(Xp_tr, ytr, BASE, rp)
bscore_ev, _ = fit_pred(Xb_tr, ytr, Xb_ev, BASE, rb)
pscore_ev, _ = fit_pred(Xp_tr, ytr, Xp_ev, BASE, rp)
log("own-side scores: rounds", rb, rp)

cols_r = CX + BQ + PQ + BS + PS + ["bscore", "pscore"]
ix = {c: i for i, c in enumerate(cols_r)}
CONS = [[ix[c] for c in CX + BQ + PQ + ["bscore", "pscore"]],
        [ix[c] for c in CX + BQ + BS],
        [ix[c] for c in CX + PQ + PS]]


def rmat(rows_idx, bs, ps):
    M = FEc[CX + BQ + PQ + BS + PS].to_numpy()[rows_idx]
    return np.column_stack([M, bs, ps])


XR_tr, XR_ev = rmat(tr_idx, bscore_tr, pscore_tr), rmat(ev_idx, bscore_ev, pscore_ev)
rr = choose_rounds(XR_tr, ytr, {**BASE, "interaction_constraints": CONS})
R_ev, R_model = fit_pred(XR_tr, ytr, XR_ev, BASE, rr, constraints=CONS)
R_tr = oof(XR_tr, ytr, BASE, rr, constraints=CONS)
log("R rounds", rr)

# G4: every root-to-leaf path of R uses features from a single constraint set
sets = [set(s) for s in CONS]


def paths(node, acc, outl):
    if "split_feature" in node:
        a2 = acc | {node["split_feature"]}
        paths(node["left_child"], a2, outl)
        paths(node["right_child"], a2, outl)
    else:
        outl.append(acc)


def check_constraints(model, sets_):
    bad = 0
    for t in model.dump_model()["tree_info"]:
        pl_ = []
        paths(t["tree_structure"], set(), pl_)
        bad += sum(1 for p_ in pl_ if not any(p_ <= s for s in sets_))
    return bad


bad = check_constraints(R_model, sets)
assert bad == 0, f"G4: {bad} paths of R cross the constraint sets"
probe = lgb.train({**BASE, "seed": 1}, lgb.Dataset(XR_tr, ytr), 30)          # unconstrained: must be caught
assert check_constraints(probe, sets) > 0, "G4: the path check did not catch an unconstrained model"
log("G4 constraint gate passed (and catches an unconstrained model)")

# stage 2
cols_t = XN + CX


def tmat(Xdf, rows_idx):
    return np.column_stack([Xdf[XN].to_numpy()[rows_idx], FEc[CX].to_numpy()[rows_idx]])


T_tr, T_ev = tmat(XR, tr_idx), tmat(XR, ev_idx)
rt = choose_rounds(T_tr, ytr, SMALLP, init=R_tr)
T_delta, T_model = fit_pred(T_tr, ytr, T_ev, SMALLP, rt, init=R_tr)
T_ev_pred = R_ev + T_delta
log("T rounds", rt)

# ---------------------------------------------------------------- statistics and bootstrap
ye, ya = y[ev_idx], yact[ev_idx]
bat_i = pd.factorize(pa["batter"].to_numpy()[ev_idx])[0]
pit_i = pd.factorize(pa["pitcher"].to_numpy()[ev_idx])[0]
gam_i = pd.factorize(pa["game_pk"].to_numpy()[ev_idx])[0]


def stats(Tp, w=None):
    w = np.ones(len(ye)) if w is None else w
    sw_ = w.sum()
    h1 = (w * ((ye - R_ev) ** 2 - (ye - Tp) ** 2)).sum() / sw_
    dl = Tp - R_ev
    r_ = ya - R_ev
    md, mr = (w * dl).sum() / sw_, (w * r_).sum() / sw_
    vd = (w * (dl - md) ** 2).sum()
    h2 = (w * (dl - md) * (r_ - mr)).sum() / vd if vd > 0 else np.nan
    return h1, h2


H1, H2 = stats(T_ev_pred)
boot = np.empty((N_BOOT, 2))
for b_ in range(N_BOOT):
    w = (RNG.exponential(size=bat_i.max() + 1)[bat_i] * RNG.exponential(size=pit_i.max() + 1)[pit_i]
         * RNG.exponential(size=gam_i.max() + 1)[gam_i])
    boot[b_] = stats(T_ev_pred, w)
lo = np.nanquantile(boot, 0.025, axis=0)
log(f"H1 MSE(R)-MSE(T) = {H1:.3e}  bootstrap 2.5% {lo[0]:.3e} | H2 slope = {H2:.3f}  2.5% {lo[1]:.3f}")
print("MSE R", float(((ye - R_ev) ** 2).mean()), "MSE T", float(((ye - T_ev_pred) ** 2).mean()),
      "| sd of delta", float(np.std(T_ev_pred - R_ev)))

# ---------------------------------------------------------------- decoy floor
role = pa["starter_p"].to_numpy()
qual = (np.stack([F[f"pq_all{k}"] for k in range(6)], 1) * np.array([0, .7, 0, .9, 1.3, 2.0])).sum(1)
app = pa[["game_pk", "pitcher", "date_i", "p_throws"]].copy()
app["role"], app["q"] = role, qual
apps = app[elig].groupby(["game_pk", "pitcher"], sort=False).agg(date_i=("date_i", "first"), hand=("p_throws", "first"),
                                                                 role=("role", "max"), q=("q", "first")).reset_index()
QCUT = np.quantile(apps.loc[apps["date_i"] < pa.loc[ev_idx, "date_i"].min(), "q"], [1 / 3, 2 / 3])
apps["tq"] = np.searchsorted(QCUT, apps["q"])


def draw_decoy(rng):
    dec = np.full(len(apps), -1, np.int64)
    fallback = 0
    for keys in (["date_i", "hand", "role", "tq"], ["date_i", "hand", "role"], ["date_i", "hand"], ["date_i"]):
        todo = dec < 0
        if not todo.any():
            break
        sub = apps[todo]
        for _, gi in sub.groupby(keys, sort=False).indices.items():
            rows = sub.index.to_numpy()[gi]
            if len(rows) < 2:
                continue
            perm = rows[rng.permutation(len(rows))]
            cand = apps["pitcher"].to_numpy()[np.roll(perm, 1)]
            same = cand == apps["pitcher"].to_numpy()[perm]          # same pitcher twice on a date (doubleheader)
            dec[perm[~same]] = cand[~same]                          # those few go to the next, coarser stratum
        if keys != ["date_i", "hand", "role", "tq"]:
            fallback += int((todo & (dec >= 0)).sum())
    return dec, fallback


def decoy_pitchers(rng):
    dec, fb_ = draw_decoy(rng)
    m = pd.Series(dec, index=pd.MultiIndex.from_frame(apps[["game_pk", "pitcher"]]))
    k = pd.MultiIndex.from_arrays([pa["game_pk"].to_numpy(), pa["pitcher"].to_numpy()])
    dp = m.reindex(k).to_numpy()
    return dp, fb_


dec_stats = []
for j in range(N_DECOY):
    dp, fb_ = decoy_pitchers(RNG)
    ok = elig & (dp >= 0)
    assert np.all(dp[elig & (dp >= 0)] != pa["pitcher"].to_numpy()[elig & (dp >= 0)]), "G5: decoy equals the real pitcher"
    Ad, BLd, LOCd, VAAd, _ = pitcher_vectors(np.where(ok, dp, pa["pitcher"].to_numpy()).astype(np.int64),
                                             pa["date_i"].to_numpy(), sg_pa)   # as of the REAL PA's date
    Xd = pairing_features(Ad, BLd, LOCd, VAAd)
    Xd.loc[~ok] = 0.0                       # no decoy: no pairing content (never the real pitcher's)
    Xd = Xd.fillna(0.0)
    Td, _ = fit_pred(tmat(Xd, tr_idx), ytr, tmat(Xd, ev_idx), SMALLP, rt, init=R_tr)
    dec_stats.append(stats(R_ev + Td))
    log(f"decoy {j + 1}/{N_DECOY}: H1 {dec_stats[-1][0]:.3e} H2 {dec_stats[-1][1]:.3f} (fallback strata {fb_}, "
        f"PAs without a decoy {int((elig & (dp < 0)).sum())})")
dec_stats = np.array(dec_stats)
p1 = (1 + (dec_stats[:, 0] >= H1).sum()) / (1 + N_DECOY)
p2 = (1 + (dec_stats[:, 1] >= H2).sum()) / (1 + N_DECOY)
print(f"decoy floor: H1 p = {p1:.3f} (decoy mean {dec_stats[:, 0].mean():.3e}, max {dec_stats[:, 0].max():.3e}) | "
      f"H2 p = {p2:.3f} (decoy mean {dec_stats[:, 1].mean():.3f}, max {dec_stats[:, 1].max():.3f})")

# ---------------------------------------------------------------- reported only
imp = pd.Series(T_model.feature_importance("gain"), index=cols_t).sort_values(ascending=False)
print("T gain importance:\n", imp.head(12).round(1).to_string())
dl = T_ev_pred - R_ev
dec_ = pd.qcut(dl, 10, labels=False, duplicates="drop")
tab = pd.DataFrame({"decile": dec_, "delta": dl, "resid_actual": ya - R_ev, "resid_x": ye - R_ev})
print(tab.groupby("decile").agg(n=("delta", "size"), delta=("delta", "mean"), resid_actual=("resid_actual", "mean"),
                                 resid_x=("resid_x", "mean")).round(4).to_string())
sub_m = (pa["lineup_starter"].to_numpy()[ev_idx] == 1) & (pa["starter_p"].to_numpy()[ev_idx] == 1) & (pa["tto"].to_numpy()[ev_idx] == 1)
print("starting batters vs starter, first time through: n", int(sub_m.sum()),
      "H1", float((((ye - R_ev) ** 2 - (ye - T_ev_pred) ** 2)[sub_m]).mean()))
print("pairing feature correlations with own-side quantities (eval rows):")
own = pd.DataFrame({"bscore": bscore_ev, "pscore": pscore_ev, "R": R_ev})
print(pd.concat([XR.iloc[ev_idx].reset_index(drop=True), own], axis=1).corr().loc[XN, ["bscore", "pscore", "R"]].round(3))
log("done")
