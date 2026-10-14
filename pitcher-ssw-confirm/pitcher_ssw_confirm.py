# %% [markdown]
# # Confirming the Spin-Axis Gap Before Arm Placements on Unseen Seasons (MLB 2021-2023)
#
# The sixth pre-registered pitcher family ([FREEZE_pitch_ssw.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_ssw.md), run once on MLB 2024-2026) was NOT SUPPORTED on all four primaries. Two of its numbers were picked out only after that run: on arm placements its AUC was above all 200 rotation floors, and the reported-only fastball-gap score on arm placements was 0.559 (Bonferroni lower bound 0.516, floor max 0.559). A target picked after seeing the data can only be **confirmed** on seasons no study has read: MLB 2021-2023 (Statcast reports the spin axis from 2020).
#
# - **C1** (the only primary): MLB 2021-2023 pooled, arm placements, the daily within-role rank of the fastball-gap-only score, injured list within 14 days. SUPPORTED if the bootstrap-over-pitchers 95% lower bound is above 0.5 **and** the rotation p-value is at most 0.025, p = (1 + #{floor >= AUC}) / 1001 over 1,000 rotations drawn uniformly from each pitcher-season's whole cyclic group (shift 0 included). The frozen families' rotation rule excludes shifts near 0, which made its floor too narrow under random labels (found before freezing; see the freeze document).
# - Everything else is reported, not tested.
#
# The notebook runs the frozen sixth-family notebook (commit `e83a56e`, checked by md5), which itself runs the frozen pitcher notebook with its 47 replacements. On top of that it only changes the seasons and removes Triple-A pitch data (none exists for 2021). **Before any 2021-2023 row meets a label, the same code is run on the frozen 2024-2026 files and must reproduce the frozen MLB numbers exactly** (rows, labels, the rival's AUC, S1, S2 and the fastball-only scores with their floors). Everything was fixed before any 2021-2023 pitch feature was compared with a placement: [FREEZE_pitch_ssw_confirm.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_ssw_confirm.md).

# %%
import gc
import hashlib
import os

import requests

SSW_SHA = "e83a56e32bf183495d5a0fe728c5da63d2ef35d9"
SSW_URL = f"https://raw.githubusercontent.com/yasumorishima/kaggle-datasets/{SSW_SHA}/pitcher-ssw-il/pitcher_ssw_il.py"
ssw = requests.get(SSW_URL, timeout=60).text
assert hashlib.md5(ssw.encode()).hexdigest() == "5fa9c4a0e362f6618bd1f02e1a067afc", "not the frozen sixth-family notebook"
LAST = 'exec(compile(src, "frozen_pitcher_ssw", "exec"))\n'
assert ssw.endswith(LAST) and ssw.count(LAST) == 1
_ns = {"__name__": "frozen_ssw_wrapper"}
exec(compile(ssw[: -len(LAST)], "frozen_ssw_wrapper", "exec"), _ns)  # fetches the frozen pitcher notebook, applies 47 replacements
BASE = _ns["src"]
assert len(_ns["REPL"]) == 47
del _ns
print("frozen base:", hashlib.md5(BASE.encode()).hexdigest(), len(BASE), "characters")

# yasunorim/mlb-statcast-2021-2023-pitch-subset, built by GHA 38483678623 (pitcher-confirm-statcast-2021-2023/build.py at 492e865)
CONFIRM_MD5 = {2021: "51c22a049453bb1ba5102d66e728a304", 2022: "ac2f365b4d00f0ab139b0a844615a3fc", 2023: "4bba35933470f5a8d6549059a486d59a"}
# Frozen sixth-family run (GHA 38435663378, ssw_verdict.csv and run.txt), printed beside this session's values (drift)
FROZEN = {"rows": (61966, 1180, 53384, 3553), "b0": 0.5644912856858865,
          "S1": (1000, "U2d", 0.5126959331991916, 0.5111601933691299, 0.48120304500968153, 1827, 1e-12),
          "S2": (2000, "U2d", 0.5327550222479216, 0.5275247509574993, 0.48882916827537265, 1045, 1e-12),
          "S1 fastball only": (1000, "U2_deld", 0.5230, 0.5208, 0.4920, 1928, 5e-5),
          "S2 fastball only": (2000, "U2_deld", 0.5586, 0.5587, 0.5162, 1092, 5e-5)}
# Triple-A first and last regular-season dates the frozen code took from Triple-A Statcast (2023-2026), printed
AAA_DATES = {2023: ("2023-03-31", "2023-09-24"), 2024: ("2024-03-29", "2024-09-22"),
             2025: ("2025-03-28", "2025-09-21"), 2026: ("2026-03-27", "2026-09-20")}

# %% [markdown]
# ## The replacements (MLB only; the same code for the regression run and the confirmation run)

# %%
AAA_SCHEDULE = '''def aaa_season_dates(years):
    """First and last date of completed Triple-A regular-season games from the StatsAPI schedule. The frozen code took
    them from Triple-A Statcast pitches (none exist for 2021); on 2023-2026 the two are equal (checked below)."""
    first, last = {}, {}
    for y in years:
        for k in range(4):
            try:
                r = requests.get("https://statsapi.mlb.com/api/v1/schedule", params={"sportId": 11, "season": y, "gameType": "R"},
                                 headers={"User-Agent": "kaggle-notebook-research/1.0"}, timeout=60)
                r.raise_for_status()
                js = r.json()
                break
            except requests.RequestException:
                time.sleep(5 * (k + 1))
        else:
            raise RuntimeError(f"StatsAPI schedule {y} failed")
        ds = [d["date"] for d in js.get("dates", []) for g in d["games"]
              if g["status"]["abstractGameState"] == "Final" and g["status"]["detailedState"] in ("Final", "Completed Early", "Game Over")]
        if ds:
            first[y], last[y] = pd.Timestamp(min(ds)), pd.Timestamp(max(ds))
    return (pd.Series(first, dtype="datetime64[ns]").rename_axis("season"),
            pd.Series(last, dtype="datetime64[ns]").rename_axis("season"))


_aaa_first, _aaa_last = aaa_season_dates([ROW_SEASONS[0] - 1] + ROW_SEASONS)
print("Triple-A regular season (StatsAPI schedule):", {y: (f"{_aaa_first[y]:%Y-%m-%d}", f"{_aaa_last[y]:%Y-%m-%d}") for y in _aaa_last.index})
season_last = {"mlb": mlb.groupby("season").game_date.max(), "aaa": _aaa_last}
season_first = {"mlb": mlb.groupby("season").game_date.min(), "aaa": _aaa_first}
'''


def between(s, a, b):
    i = s.index(a)
    return s[i:s.index(b, i)]


def make(mode):
    """The frozen base with the confirmation replacements. Modes: 'reference' (no replacement: the frozen sixth-family
    code with Triple-A, 2024-2026), 'regression' (MLB only, 2024-2026), 'confirm' (MLB only, 2021-2023)."""
    assert mode in ("reference", "regression", "confirm")
    rep = []
    if mode == "confirm":
        rep += [
            ("seasons", "ROW_SEASONS, TRAIN, TEST = [2024, 2025, 2026], [2024, 2025], 2026",
             "ROW_SEASONS, TRAIN, TEST = [2021, 2022, 2023], [2021, 2022], 2023"),
            ("MLB md5", between(BASE, "MLB_MD5 = {2024:", "assert FAKE or not SMOKE"), f"MLB_MD5 = {CONFIRM_MD5!r}\n"),
            ("league season", 'rows_m = build_rows(mlb, "mlb", 2025, "aaa")', 'rows_m = build_rows(mlb, "mlb", 2022, "aaa")'),
        ]
    rep += [] if mode == "reference" else [
        ("no Triple-A pitches", 'aaa = load("aaa_pitching_{y}.parquet", [2023] + ROW_SEASONS, AAA_MD5)\n',
         "aaa = mlb.iloc[:0].copy()  # MLB only: no Triple-A pitch data (none exists for 2021)\n"),
        ("fill table MLB only", 'for name, d in (("MLB", mlb), ("Triple-A", aaa)):\n    fill =', 'for name, d in (("MLB", mlb),):\n    fill ='),
        ("Triple-A season dates", between(BASE, 'season_last = {"mlb":', "if not SMOKE:\n    for lv in season_last:"), AAA_SCHEDULE),
        ("history years", "HIST_YEARS = list(range(2022, TEST + 1))", "HIST_YEARS = list(range(ROW_SEASONS[0] - 2, TEST + 1))"),
        ("game-log years", "    for y in [2023] + ROW_SEASONS:\n", "    for y in [ROW_SEASONS[0] - 1] + ROW_SEASONS:\n"),
        ("empty rehab table", "no_rehab_tx = ~pd.Series([k in rehab_seasons for k in rem.index], index=rem.index)",
         "no_rehab_tx = ~pd.Series([k in rehab_seasons for k in rem.index], index=rem.index, dtype=bool)"),
        ("convention gate MLB only", 'for name, d in (("MLB", mlb), ("Triple-A", aaa)):\n    a_, x_, z_', 'for name, d in (("MLB", mlb),):\n    a_, x_, z_'),
        ("fill gates MLB only", 'for name, d in (("mlb", mlb), ("aaa", aaa)):', 'for name, d in (("mlb", mlb),):'),
        ("no Triple-A rows", 'rows_a = build_rows(aaa, "aaa", 2023, "mlb")\n', ""),
        ("fake labels MLB only", "    for rows_x in (rows_m, rows_a):\n", "    for rows_x in (rows_m,):\n"),
        ("no Triple-A labels", 'rows_a = attach_labels(rows_a, "mlb")\n', ""),
        ("placement counts MLB only", 'for name, rows_x in (("MLB", rows_m), ("Triple-A", rows_a)):',
         'for name, rows_x in (("MLB", rows_m),):'),
        ("league-scale gate in the code test too",
         "    assert FAKE or all(np.isfinite(v) for v in lscale.values()), ('a league-typical scale is missing', lscale)\n",
         "    assert all(np.isfinite(v) for v in lscale.values()), ('a league-typical scale is missing', lscale)  # MLB only\n"),
        ("placement count text", 'print(f"{name}: placements of these pitchers in 2024-2026 {len(lab_y):,};',
         'print(f"{name}: placements of these pitchers in {ROW_SEASONS[0]}-{TEST} {len(lab_y):,};'),
    ]
    src = BASE
    for name, old, new in rep:
        assert src.count(old) == 1, f"replacement '{name}' matches {src.count(old)} times"
        src = src.replace(old, new)
    model_defs = between(src, "def rotate_z(df, rng):", "\n\nP2 = model_test(rows_m")
    cut = src.index("\nif not FAKE:  # equality gates")
    src = src[:cut] + "\n\n" + model_defs + "\n\n" + COMMON + (CONFIRM_TAIL if mode == "confirm" else METRICS)
    for word in ("rows_a", "P4", "aaa_pitching"):
        assert word not in src[cut:], f"{word} after the cut"
    print(f"{mode}: {len(rep)} replacements applied")
    return src


COMMON = '''
# Season gates (label-free; also run on 2024-2026, where they must pass on the frozen data)
_yrs = tx.loc[tx.kind == "placed", "effective"].dt.year.value_counts().sort_index()
print("placements read by effective year (all levels):", _yrs.to_dict())
# a few placements carry an effective date years before their transaction date, so presence alone proves nothing
assert all(_yrs.get(y, 0) >= 300 for y in range(ROW_SEASONS[0] - 2, TEST + 1)), "placement history does not start two seasons before the rows"
assert sorted(rows_m.season.unique()) == ROW_SEASONS, "row seasons"
_days = rows_m.day.to_numpy().astype("datetime64[D]")
_hit = np.zeros(len(rows_m), bool)
for _b, _ix in rows_m.groupby("pitcher").indices.items():
    _hit[_ix] = np.isin(_days[_ix], app_days["mlb"].get(_b, EMPTY).astype("datetime64[D]"))
_share = pd.Series(_hit).groupby(rows_m.season.to_numpy()).mean()
print("Statcast appearance days found in the StatsAPI MLB game logs, by season:", _share.round(4).to_dict())
assert (_share >= 0.99).all(), "Statcast days and MLB game logs disagree"
for _y in ROW_SEASONS + [ROW_SEASONS[0] - 1]:
    _n = sum(bool(((v >= np.datetime64(f"{_y}-01-01")) & (v < np.datetime64(f"{_y + 1}-01-01"))).any()) for v in app_days["aaa"].values())
    _last = max((v[(v >= np.datetime64(f"{_y}-01-01")) & (v < np.datetime64(f"{_y + 1}-01-01"))].max()
                 for v in app_days["aaa"].values() if ((v >= np.datetime64(f"{_y}-01-01")) & (v < np.datetime64(f"{_y + 1}-01-01"))).any()),
                default=None)
    print(f"pitchers with Triple-A game-log days in {_y}: {_n:,}; last such day {str(_last)[:10] if _last is not None else None}"
          f" (schedule's last {season_last['aaa'][_y].date() if _y in season_last['aaa'].index else None})")
    assert _n > 0 or _y == 2020, ("no Triple-A game logs", _y)  # no minor-league season in 2020
p1_df = rows_m[rows_m.use14].reset_index(drop=True)
arm_df = p1_df[~p1_df.il14 | p1_df.next_arm].reset_index(drop=True)
'''

METRICS = '''
# The MLB numbers the regression gate compares (2024-2026 runs only)
_tr = (rows_m.season.isin(TRAIN) & rows_m.use14).to_numpy()
_te = ((rows_m.season == TEST) & rows_m.use14).to_numpy()
OUT = {"rows": (len(rows_m), rows_m.pitcher.nunique(), int(rows_m.use14.sum()), int(rows_m.il14[rows_m.use14].sum())),
       "b0": auc(rows_m.il14[_te].to_numpy(), predict(fit(CTX, rows_m[_tr], rows_m.il14[_tr].to_numpy()), CTX, rows_m[_te])),
       "arm positives": int((rows_m.use14 & rows_m.il14 & rows_m.next_arm).sum()),
       "aaa dates": {int(y): (str(season_first["aaa"][y].date()), str(season_last["aaa"][y].date())) for y in season_last["aaa"].index}}
for key in ("S1", "S2", "S1 fastball only", "S2 fastball only"):
    sd, sc = (1000 if key.startswith("S1") else 2000), ("U2_deld" if "fastball" in key else "U2d")
    r_ = unsupervised_test(p1_df if key.startswith("S1") else arm_df, key, sd, sc)
    OUT[key] = (r_["auc"], r_["floor"].max(), r_["lo"], int(r_["y"].sum()))
print("OUT", OUT)
'''

CONFIRM_TAIL = '''
assert ROW_SEASONS == [2021, 2022, 2023] and TEST == 2023


def boot_dist(df, y, s, seed=2, n=N_BOOT):
    """The frozen boot() draws for one score, returned as the array of AUCs (pitchers resampled)."""
    idx_by = list(df.groupby("pitcher").indices.values())
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        idx = np.concatenate([idx_by[i] for i in rng.integers(0, len(idx_by), len(idx_by))])
        v.append(auc(y[idx], s[idx]))
    v = np.array(v)
    return v[np.isfinite(v)]


C1 = unsupervised_test(arm_df, "C1 data (verdict at the end of this line: the frozen families' rule, not C1's)", 2000, "U2_deld", gate=True)
_v = boot_dist(C1["df"], C1["y"], C1["s"])
assert abs(np.percentile(_v, LO) - C1["lo"]) < 1e-12, "the bootstrap is not the frozen one"
lo95, lo99 = np.percentile(_v, 2.5), np.percentile(_v, 0.5)
f975 = np.percentile(C1["floor"], 97.5)


def rotated_group(vals, groups, sd):
    """C1's floor: each pitcher-season with at least 7 values (as frozen) is rolled by k drawn uniformly from the whole
    cyclic group 0..n-1, shift 0 included. The frozen rule (k on [n/6, 5n/6]) never draws the shifts near 0, whose
    positive rows overlap the real alignment's; its rotations are therefore less like the real alignment than a random
    member of the whole group, and the real AUC is too extreme against them under random labels (pre-freeze diagnostic)."""
    rng = np.random.default_rng(sd)
    rot = vals.copy()
    for f in groups:
        if len(f) >= 7:
            rot[f] = np.roll(vals[f], int(rng.integers(0, len(f))))
    return rot


N_ROT_C1 = 1000
_gix = list(C1["df"].groupby(["pitcher", "season"]).indices.values())
floor_c1 = np.array([auc(C1["y"], rotated_group(C1["s"], _gix, sd)) for sd in range(5000, 5000 + N_ROT_C1)])
p_rot = (1 + int((floor_c1 >= C1["auc"]).sum())) / (N_ROT_C1 + 1)
assert np.array_equal(np.sort(np.concatenate(_gix)), np.arange(len(C1["df"]))), "C1's groups do not cover its rows once"
_key, _days = C1["df"][["pitcher", "season"]].to_numpy(), C1["df"].day.to_numpy()
for _f in _gix:
    assert (_key[_f] == _key[_f][0]).all() and (np.diff(_days[_f]) > np.timedelta64(0, "D")).all(), "C1's groups are not pitcher-seasons in time order"
assert all(floor_c1[i] == auc(C1["y"], rotated_group(C1["s"], _gix, 5000 + i)) for i in range(3)), "C1's floor is not the whole-group rotation"
_ks, _out = [], []
for _sd in range(5000, 5050):
    _ri = rotated_group(np.arange(len(C1["df"]), dtype=float), _gix, _sd).astype(int)
    for _f in _gix:
        _r = _ri[_f]
        if len(_f) < 7:
            assert np.array_equal(_r, _f), "a pitcher-season with fewer than 7 values was rotated"
            continue
        _k = (-int(np.flatnonzero(_f == _r[0])[0])) % len(_f)
        assert np.array_equal(_r, np.roll(_f, _k)), "C1's floor is not a rotation within the pitcher-season"
        _m = max(2, int(np.ceil(len(_f) / 6)))
        _ks.append(_k)
        _out.append(_k < _m or _k > len(_f) - _m)
assert 0 in _ks, "shift 0 is never drawn"
assert np.mean(_out) > 0.2, "C1's shifts avoid the frozen rule's excluded band (it holds about a third of the group)"
print(f"C1 floor structure: {len(_ks):,} group draws, share of shifts outside the frozen band {np.mean(_out):.3f}, "
      f"shift 0 drawn {_ks.count(0):,} times")
if FAKE:  # C1's floor must be beatable by a planted signal
    _pl = np.clip(C1["s"] + 0.3 * C1["y"], 0, 1.3)
    _pf = np.array([auc(C1["y"], rotated_group(_pl, _gix, sd)) for sd in range(5000, 5000 + N_ROT_C1)])
    assert (1 + int((_pf >= auc(C1["y"], _pl)).sum())) / (N_ROT_C1 + 1) <= 0.025, "C1's floor cannot be beaten by a planted signal"
c1_ok = bool(lo95 > 0.5 and p_rot <= 0.025)
print(f"C1: AUC {C1['auc']:.4f}; 95% lower {lo95:.4f}, 99% lower {lo99:.4f}; whole-group rotation floor ({N_ROT_C1}): mean "
      f"{floor_c1.mean():.4f}, 97.5th percentile {np.percentile(floor_c1, 97.5):.4f}, p {p_rot:.4f}; frozen-rule floor ({len(C1['floor'])}, "
      f"printed only) 97.5th percentile {f975:.4f}, max {C1['floor'].max():.4f}; positives {int(C1['y'].sum()):,}; "
      f"pitchers {C1['df'].pitcher.nunique():,} -> {'SUPPORTED' if c1_ok else 'NOT SUPPORTED'}")

# Reported, not tested
# Series position: a real placement often ends a pitcher's season, and a rotation wraps that block onto the season's
# start. If the score drifts within a pitcher-season for other reasons, the rotation floor is biased. Fixed before the run:
# (a) labels only: AUC of the position in the series, with the same whole-group p (cyclic uniformity of the labels);
# (b) features only: mean score by position decile (all placements' usable rows, no label);
# (c) C1 with the score minus its decile mean from (b), with the same whole-group p.
_pp = p1_df[np.isfinite(p1_df["U2_deld"])][["pitcher", "season", "day", "U2_deld"]].copy()
_gp = _pp.groupby(["pitcher", "season"]).day
_pp["pos"] = (_gp.rank(method="first") - 1) / (_gp.transform("size") - 1).clip(lower=1)
_pp["dec"] = np.minimum((_pp.pos * 10).astype(int), 9)
_dm = _pp.groupby("dec").U2_deld.mean()
print("label-free: mean fastball-gap daily rank by position in the pitcher-season (deciles)\\n" + _dm.round(4).to_string())
_cp = C1["df"][["pitcher", "season", "day"]].merge(_pp[["pitcher", "season", "day", "pos", "dec"]], how="left",
                                                    on=["pitcher", "season", "day"], validate="one_to_one")
assert len(_cp) == len(C1["df"]) and _cp.pos.notna().all(), "C1 rows missing from the position table"


def p_group(sc):
    a = auc(C1["y"], sc)
    fl = np.array([auc(C1["y"], rotated_group(sc, _gix, sd)) for sd in range(5000, 5000 + N_ROT_C1)])
    return a, fl.mean(), (1 + int((fl >= a).sum())) / (N_ROT_C1 + 1)


POS = {"position in the series (labels only)": p_group(_cp.pos.to_numpy()),
       "C1 score minus its position-decile mean": p_group(C1["s"] - _dm.reindex(_cp.dec).to_numpy())}
for _k, (_a, _fm, _p) in POS.items():
    print(f"reported: {_k}: AUC {_a:.4f}; whole-group floor mean {_fm:.4f}; p {_p:.4f}")
unsupervised_test(arm_df, "reported: full U2, arm placements", 2000)
unsupervised_test(p1_df, "reported: full U2, all placements", 1000)
unsupervised_test(p1_df, "reported: fastball gap only, all placements", 1000, "U2_deld")
u30_m = rows_m[rows_m.use30].reset_index(drop=True)
unsupervised_test(u30_m[~u30_m.il30 | u30_m.next_arm].reset_index(drop=True), "reported: fastball gap only, arm, IL30", 2000,
                  "U2_deld", "il30")
_c = C1["df"]
_sub = {**subsets_of(_c),
        "without 2021 from 3 June (sticky-stuff checks)": (~(_c.season.eq(2021) & (_c.day >= pd.Timestamp(2021, 6, 3)))).to_numpy(),
        "2022 and 2023": _c.season.isin([2022, 2023]).to_numpy()}
C1_SUB = by_subsets(C1, _sub)
print("C1 subsets (the frozen-rule rotations, reported)\\n" + C1_SUB.round(4).to_string(index=False))
P2c = model_test(rows_m, "MLB 2023 model over the rival (reported)")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(np.zeros(len(floor_c1)), floor_c1, color="#bbbbbb", s=12, label="rotations (whole group)")
ax.scatter([0], [C1["auc"]], color="#c0392b", s=110, zorder=3, label="real")
ax.axhline(0.5, color="#cccccc", lw=1)
ax.axhline(np.percentile(floor_c1, 97.5), color="#888888", lw=1, ls="--", label="97.5th percentile of the rotations")
ax.set_xticks([0], ["C1 MLB 2021-2023, arm"])
ax.set_ylabel("AUC")
ax.set_title("Fastball spin-axis gap, unseen seasons")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("confirm_floor.png", dpi=120)
plt.show()

summary = pd.DataFrame([{
    "test": "C1 MLB 2021-2023 arm placements, fastball-gap-only U2 daily rank, IL14", "AUC": C1["auc"], "lo95": lo95,
    "lo99": lo99, "floor_mean": floor_c1.mean(), "floor_p975": np.percentile(floor_c1, 97.5), "p_rotation": p_rot,
    "frozen_rule_floor_p975": f975, "frozen_rule_floor_max": C1["floor"].max(),
    "positives": int(C1["y"].sum()), "verdict": "SUPPORTED" if c1_ok else "NOT SUPPORTED",
    "reported_position_auc": POS["position in the series (labels only)"][0],
    "reported_position_p": POS["position in the series (labels only)"][2],
    "reported_adjusted_auc": POS["C1 score minus its position-decile mean"][0],
    "reported_adjusted_p": POS["C1 score minus its position-decile mean"][2]}])
if FAKE:
    print("FAKE LABELS: code test only, no verdict written")
else:
    summary.to_csv("confirm_verdict.csv", index=False)
    C1_SUB.to_csv("confirm_subsets.csv", index=False)
print(summary.to_string(index=False))
'''

# %% [markdown]
# ## 1. Regression gate: in this session, the frozen code (with Triple-A) and the MLB-only code on 2024-2026 must agree exactly
#
# Both read the StatsAPI minutes apart. A pre-freeze check found that the frozen code's arm labels can move between runs
# hours apart: 15 placement groups (same pitcher, same effective date) disagree on the arm flag, and the frozen merge keeps
# the one the API lists first. Comparing two runs of this session separates the effect of the replacements (must be
# zero) from that drift (printed against the frozen run's values, not gated).

# %%
def run_mode(mode):
    ns = {"__name__": "__main__"}
    exec(compile(make(mode), f"confirm_{mode}", "exec"), ns)
    out = ns["OUT"]
    del ns
    gc.collect()
    return out


FAKE_RUN = os.environ.get("SMOKE_FAKE_LABELS") == "1"
REF = run_mode("reference")
REG = run_mode("regression")
for k in REF:
    print(f"{k}: frozen code with Triple-A {REF[k]} | MLB only {REG[k]}")
for k in ("S1", "S2", "S1 fastball only", "S2 fastball only"):
    print(f"{k} drift since the frozen run (AUC, floor max, lower bound, positives): "
          f"{tuple(round(float(a - b), 6) for a, b in zip(REF[k], FROZEN[k][2:6]))}")
print(f"rows {REF['rows']} against the frozen run {FROZEN['rows']}; B0 {REF['b0'] - FROZEN['b0']:+.2e}; "
      f"Triple-A dates of the frozen run {AAA_DATES == {y: REF['aaa dates'][y] for y in AAA_DATES}}")
# label-free parts (rows, pitchers, usable rows, Triple-A dates) are compared in the code test too
assert REG["rows"][:3] == REF["rows"][:3], "MLB rows differ"
assert REG["aaa dates"] == REF["aaa dates"], "the Triple-A dates from the schedule differ from those of Triple-A Statcast"
if not FAKE_RUN:  # with random labels the two runs differ (the Triple-A rows draw from the same generator)
    assert REG["rows"] == REF["rows"] and REG["arm positives"] == REF["arm positives"], "MLB labels differ"
    assert abs(REG["b0"] - REF["b0"]) < 1e-12, "the rival differs"
    for k in ("S1", "S2", "S1 fastball only", "S2 fastball only"):
        assert all(abs(a - b) < 1e-12 for a, b in zip(REG[k], REF[k])), (k, REG[k], REF[k])
    print("REGRESSION GATE PASSED: the MLB-only replacements change no MLB number of the frozen code")

# %% [markdown]
# ## 2. Confirmation run: MLB 2021-2023

# %%
ns = {"__name__": "__main__"}
exec(compile(make("confirm"), "confirm_2021_2023", "exec"), ns)
