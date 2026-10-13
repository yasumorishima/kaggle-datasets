# %% [markdown]
# # Pre-freeze check: the regression gate of the 2021-2023 confirmation notebook
#
# Runs only section 1 of the draft confirmation notebook (the frozen sixth-family code with Triple-A and the MLB-only code on 2024-2026 must agree exactly). No 2021-2023 file is attached or read.

# %% [markdown]
# # Confirming the Spin-Axis Gap Before Arm Placements on Unseen Seasons (MLB 2021-2023)
#
# The sixth pre-registered pitcher family ([FREEZE_pitch_ssw.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_ssw.md), run once on MLB 2024-2026) was NOT SUPPORTED on all four primaries. Two of its numbers were picked out only after that run: on arm placements its AUC was above all 200 rotation floors, and the reported-only fastball-gap score on arm placements was 0.559 (Bonferroni lower bound 0.516, floor max 0.559). A target picked after seeing the data can only be **confirmed** on seasons no study has read: MLB 2021-2023 (Statcast reports the spin axis from 2020).
#
# - **C1** (the only primary): MLB 2021-2023 pooled, arm placements, the daily within-role rank of the fastball-gap-only score, injured list within 14 days. SUPPORTED if the bootstrap-over-pitchers 95% lower bound is above 0.5 **and** the AUC is above the 97.5th percentile of 200 rotation floors.
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

CONFIRM_MD5 = {2021: "@@MD5_2021@@", 2022: "@@MD5_2022@@", 2023: "@@MD5_2023@@"}
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
    print(f"pitchers with Triple-A game-log days in {_y}: {_n:,}")
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


C1 = unsupervised_test(arm_df, "C1 MLB 2021-2023, arm placements, fastball gap only", 2000, "U2_deld", gate=True)
_v = boot_dist(C1["df"], C1["y"], C1["s"])
assert abs(np.percentile(_v, LO) - C1["lo"]) < 1e-12, "the bootstrap is not the frozen one"
lo95, lo99 = np.percentile(_v, 2.5), np.percentile(_v, 0.5)
f975 = np.percentile(C1["floor"], 97.5)
c1_ok = bool(lo95 > 0.5 and C1["auc"] > f975)
print(f"C1: AUC {C1['auc']:.4f}; 95% lower {lo95:.4f}, 99% lower {lo99:.4f}; rotation floor 97.5th percentile {f975:.4f}, "
      f"max {C1['floor'].max():.4f} (above all 200: {bool(C1['auc'] > C1['floor'].max())}); positives {int(C1['y'].sum()):,}; "
      f"pitchers {C1['df'].pitcher.nunique():,} -> {'SUPPORTED' if c1_ok else 'NOT SUPPORTED'}")

# Reported, not tested
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
print("C1 subsets (same rotations)\\n" + C1_SUB.round(4).to_string(index=False))
P2c = model_test(rows_m, "MLB 2023 model over the rival (reported)")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(np.zeros(len(C1["floor"])), C1["floor"], color="#bbbbbb", s=12, label="rotations")
ax.scatter([0], [C1["auc"]], color="#c0392b", s=110, zorder=3, label="real")
ax.axhline(0.5, color="#cccccc", lw=1)
ax.set_xticks([0], ["C1 MLB 2021-2023, arm"])
ax.set_ylabel("AUC")
ax.set_title("Fastball spin-axis gap, unseen seasons")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("confirm_floor.png", dpi=120)
plt.show()

summary = pd.DataFrame([{
    "test": "C1 MLB 2021-2023 arm placements, fastball-gap-only U2 daily rank, IL14", "AUC": C1["auc"], "lo95": lo95,
    "lo99": lo99, "floor_p975": f975, "floor_max": C1["floor"].max(), "above_all_200": bool(C1["auc"] > C1["floor"].max()),
    "positives": int(C1["y"].sum()), "verdict": "SUPPORTED" if c1_ok else "NOT SUPPORTED"}])
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
if not FAKE_RUN:  # with random labels the two runs differ (the Triple-A rows draw from the same generator)
    assert REG["rows"] == REF["rows"] and REG["arm positives"] == REF["arm positives"], "MLB rows or labels differ"
    assert REG["aaa dates"] == REF["aaa dates"], "the Triple-A dates from the schedule differ from those of Triple-A Statcast"
    assert abs(REG["b0"] - REF["b0"]) < 1e-12, "the rival differs"
    for k in ("S1", "S2", "S1 fastball only", "S2 fastball only"):
        assert all(abs(a - b) < 1e-12 for a, b in zip(REG[k], REF[k])), (k, REG[k], REF[k])
    print("REGRESSION GATE PASSED: the MLB-only replacements change no MLB number of the frozen code")

