# %% [markdown]
# # Does a Pitcher's Two-Strike Reach-Back Fade Before the Injured List?
#
# Four [pre-registered](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch.md) [pitcher](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_inouting.md) [studies](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_mix.md) [found](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_command.md) that neither the window levels of the fastball and delivery, nor the fade and scatter inside an outing, nor the pitch mix and secondary-pitch quality, nor command drift from a pitcher's own normal before the injured list. None of them asked what a pitcher **keeps in reserve**: most pitchers throw their fastball harder once they have two strikes. This notebook measures that reach-back:
#
# - **Velocity reserve**: how much faster his fastball is with two strikes than with fewer, and in put-away counts (0-2, 1-2) than on 0-0, each pitch measured against the mean of its own outing.
# - **Delivery effort**: the same two-strike difference in extension and in arm angle.
#
# It runs the frozen pitcher notebook (commit `2c6b01d`, checked by md5) with **only its parameters replaced** by these four; rows, labels, the rival model, floors and tests are unchanged, and equality gates check that. The count before each pitch comes from the frozen MLB files and, for Triple-A, from the [separate location fetch](https://www.kaggle.com/code/yasunorim/triple-a-statcast-pitch-location-2023-2026-fetch) joined by pitch key; a gate checks it against a count rebuilt from the pitch descriptions. Everything was fixed before any new parameter was compared with a placement: [FREEZE_pitch_reach.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_reach.md).
#
# Names in the printed output are the frozen code's: **P1 = V1** (MLB, all placements), **P3 = V2** (MLB, arm placements), **P4 = V3** (Triple-A), **P2 = V4** (MLB 2026 model over the rival). "Delivery" in the frozen printouts is now the **velocity reserve** group (the two velocity parameters); "results" is the **delivery effort** group.
#
# This is the fifth pre-registered family of tests on the same pitcher placements, so the summary also shows each verdict at a 20-way Bonferroni reading across the five families (0.125% and 99.875%); the 4-way verdict is "supported within this family", not across the five.

# %%
import hashlib

import requests

SHA = "2c6b01de48f5fd00dba18a5731404b40d4ddd707"
URL = f"https://raw.githubusercontent.com/yasumorishima/kaggle-datasets/{SHA}/pitcher-anomaly-il/pitcher_anomaly_il.py"
src = requests.get(URL, timeout=60).text
assert hashlib.md5(src.encode()).hexdigest() == "100c853ccb8e42f386f547ead97f0b95", "not the frozen pitcher notebook"

import numpy
import pandas
import sklearn

print("versions: numpy", numpy.__version__, "| pandas", pandas.__version__, "| scikit-learn", sklearn.__version__)

LOCATION = '''LOC = ["stand", "plate_x", "plate_z", "sz_top", "sz_bot", "balls", "strikes"]
LOC_MD5 = {2023: "ddc88cbdb25d87cd356890b5934a1f69", 2024: "0eb84f0af52298597404accf72a318a5",
           2025: "9652d14a943e76d3d772437554110dc4", 2026: "921600911144b7e89c2fe58d8f87aef3"}
LOC_KEYS = ["game_pk", "at_bat_number", "pitch_number"]
STRIKE = {"called_strike", "swinging_strike", "swinging_strike_blocked", "missed_bunt", "foul", "foul_bunt", "foul_tip",
          "bunt_foul_tip", "automatic_strike", "foul_pitchout", "swinging_pitchout"}
BALL = {"ball", "blocked_ball", "pitchout", "hit_by_pitch", "intent_ball", "automatic_ball"}


def join_location(d, y):
    """Left-join the Triple-A location fetch to a frozen Triple-A pitching file by pitch key (FREEZE_pitch_command.md)."""
    p = glob.glob(os.path.join(INPUT_ROOT, "**", f"aaa_location_{y}.parquet"), recursive=True)
    assert len(p) == 1, (y, p)
    m = hashlib.md5(open(p[0], "rb").read()).hexdigest()
    print(f"{os.path.basename(p[0])} md5 {m}")
    assert FAKE or m == LOC_MD5.get(y), "not the location file recorded in FREEZE_pitch_command.md"
    lo = pd.read_parquet(p[0], columns=LOC_KEYS + ["pitcher", "game_date"] + LOC).rename(
        columns={"pitcher": "pitcher_loc", "stand": "stand_loc", "game_date": "date_loc"})
    lo["date_loc"] = pd.to_datetime(lo["date_loc"])
    assert not lo.duplicated(LOC_KEYS).any(), "duplicate pitch keys in the location file"
    n0 = len(d)
    d = d.merge(lo, on=LOC_KEYS, how="left", validate="many_to_one")
    assert len(d) == n0, "the join changed the number of pitches"
    hit = d.pitcher_loc.notna().to_numpy()
    same = ((d.pitcher_loc.to_numpy() == d.pitcher.to_numpy()) & (d.stand_loc.to_numpy() == d.stand.to_numpy())
            & (d.date_loc.to_numpy() == pd.to_datetime(d.game_date).to_numpy()))
    print(f"  Triple-A {y}: location joined to {hit.mean():.4%} of pitches, same pitcher, batter side and date on "
          f"{same[hit].mean():.4%} of those")
    assert hit.mean() >= 0.99, "location missing for more than 1% of Triple-A pitches"
    assert same[hit].mean() >= 0.999, "location rows disagree with the frozen file on pitcher, batter side or date"
    d.loc[~same, ["plate_x", "plate_z", "sz_top", "sz_bot", "balls", "strikes"]] = np.nan
    ok = (d.zone.notna() & d.plate_x.notna() & d.plate_z.notna() & d.sz_top.notna() & d.sz_bot.notna()).to_numpy()
    box = ((d.plate_x.abs() <= 0.83) & (d.plate_z >= d.sz_bot) & (d.plate_z <= d.sz_top)).to_numpy()
    agree = (d.zone.between(1, 9).to_numpy() == box)[ok].mean()
    print(f"  Triple-A {y}: frozen zone agrees with the joined plate location on {agree:.4%} of {ok.sum():,} pitches")
    assert agree >= 0.93, "joined locations do not match the frozen zone (pitches shifted inside a plate appearance?)"
    return d.drop(columns=["pitcher_loc", "stand_loc", "date_loc"])


def count_check(d, name):
    """Label-free gate (FREEZE_pitch_reach.md): the count before each pitch agrees with a count rebuilt from the
    descriptions of the earlier pitches of its plate appearance (strikes capped at 2, balls at 3)."""
    e = d.sort_values(LOC_KEYS, kind="stable")
    k = [e.game_pk, e.at_bat_number]
    s = e.description.isin(STRIKE).astype(int)
    b = e.description.isin(BALL).astype(int)
    st = np.minimum(s.groupby(k).cumsum() - s, 2).to_numpy()
    bl = np.minimum(b.groupby(k).cumsum() - b, 3).to_numpy()
    ok = (e.strikes.notna() & e.balls.notna()).to_numpy()
    first = (~e.duplicated(["game_pk", "at_bat_number"])).to_numpy()
    two = ((e.strikes.to_numpy() == 2) == (st == 2))[ok].mean()
    agree = (e.strikes.to_numpy() == st)[ok].mean()
    agree_b = (e.balls.to_numpy() == bl)[ok].mean()
    zero = ((e.strikes.to_numpy() == 0) & (e.balls.to_numpy() == 0))[ok & first].mean()
    print(f"  {name}: count filled {ok.mean():.4%}; strikes agree {agree:.4%}, two-strike agree {two:.4%}, "
          f"balls agree {agree_b:.4%}; first pitch 0-0 {zero:.4%}")
    assert ok.mean() >= 0.98, "count missing for more than 2% of pitches"
    assert min(agree, two, agree_b, zero) >= 0.995, "the count does not follow the pitch descriptions"
    assert (e.strikes.dropna() <= 2).all() and (e.balls.dropna() <= 3).all(), "count out of range"


def load(pattern, years, md5s=None):'''

REACH = '''
def group_key(*cols):
    """Integer group id per pitch (empty for no pitches: pandas 2 cannot factorize an empty MultiIndex)."""
    if len(cols[0]) == 0:
        return np.zeros(0, dtype=np.int64)
    return pd.factorize(pd.MultiIndex.from_arrays(list(cols)))[0]


def outing_dev(gd, gp, v):
    """Deviation of each pitch from the mean of its outing (game_pk, game_date) over the pitches of this type with a
    value; NaN without a value or in an outing with fewer than IN_MIN such pitches."""
    out = np.full(len(v), np.nan)
    pos = np.flatnonzero(np.isfinite(v))
    if len(pos) == 0:
        return out
    key = group_key(gp[pos], gd[pos])
    for ii in pd.Series(np.arange(len(pos))).groupby(key).indices.values():
        ix = pos[ii]
        if len(ix) >= IN_MIN:
            out[ix] = v[ix] - v[ix].mean()
    return out


def wgap(dates, dev, a, b, w, days, mn):
    """Mean deviation of the set-a pitches minus that of the set-b pitches among the last w pitches strictly before
    each day (needs w pitches, half of them with a deviation, and mn deviations in each set)."""
    k = np.searchsorted(dates, days, side="left")
    f = np.isfinite(dev)
    cum = lambda x: np.r_[0.0, np.cumsum(x, dtype=float)]
    cn = cum(f)
    sa, na = cum(np.where(f & a, dev, 0.0)), cum(f & a)
    sb, nb = cum(np.where(f & b, dev, 0.0)), cum(f & b)
    out = np.full(len(days), np.nan)
    ok = k >= w
    hi, lo = k[ok], k[ok] - w
    n, n_a, n_b = cn[hi] - cn[lo], na[hi] - na[lo], nb[hi] - nb[lo]
    good = (n >= w / 2) & (n_a >= mn) & (n_b >= mn)
    val = (sa[hi] - sa[lo]) / np.maximum(n_a, 1) - (sb[hi] - sb[lo]) / np.maximum(n_b, 1)
    out[ok] = np.where(good, val, np.nan)
    return out


def reach_series(g, pt, days):
    """Four reach-back parameters per fastball type (FREEZE_pitch_reach.md). g is sorted by date, game, at-bat, pitch."""
    gd, gp = g.game_date.to_numpy(), g.game_pk.to_numpy()
    st, bl = g.strikes.to_numpy(float), g.balls.to_numpy(float)
    fb = {}
    for t in FB:
        m = pt == t
        two, under = st[m] == 2, st[m] < 2
        put, first = two & (bl[m] <= 1), (st[m] == 0) & (bl[m] == 0)
        for n, c in FBCOL.items():
            dv = outing_dev(gd[m], gp[m], g[c].to_numpy(float)[m])
            a, b, j = (put, first, 1) if n == "FB put-away velocity" else (two, under, 0)
            for sfx, w, mn in zip(SFX, (LONG[0], SHORT[0]), REACH_MIN):
                fb[(t, n + sfx)] = wgap(gd[m], dv, a, b, w, days, mn[j])
    return fb, {}


def reach_plain(g, days):
    """The same four parameters by a plain loop over days (a second implementation; the gate in build_rows compares)."""
    out = {}
    for t in FB:
        s = g[g.pitch_type.fillna("").to_numpy() == t].reset_index(drop=True)
        stv, blv = s.strikes.astype(float), s.balls.astype(float)
        for n, c in FBCOL.items():
            v = s[c].astype(float)
            grp = [s.game_pk, s.game_date]
            cnt = v.notna().groupby(grp).transform("sum")
            dev = (v - v.groupby(grp).transform("mean")).where(v.notna() & (cnt >= IN_MIN))
            if n == "FB put-away velocity":
                A, B, j = (stv == 2) & (blv <= 1), (stv == 0) & (blv == 0), 1
            else:
                A, B, j = stv == 2, stv < 2, 0
            for sfx, w, mn in zip(SFX, (LONG[0], SHORT[0]), REACH_MIN):
                res = np.full(len(days), np.nan)
                for i, day in enumerate(days):
                    before = s.index[(s.game_date < day).to_numpy()]
                    if len(before) < w:
                        continue
                    dv = dev.loc[before[-w:]]
                    ok = dv.notna()
                    xa, xb = dv[ok & A.loc[dv.index]], dv[ok & B.loc[dv.index]]
                    if ok.sum() >= w / 2 and len(xa) >= mn[j] and len(xb) >= mn[j]:
                        res[i] = xa.mean() - xb.mean()
                out[(t, n + sfx)] = res
    return out


def params_for(g, prev_type):'''

PARAMS = '''FBP = ["FB velocity reach-back", "FB put-away velocity", "Extension reach-back", "Arm-angle reach-back"]
FBCOL = {"FB velocity reach-back": "release_speed", "FB put-away velocity": "release_speed",
         "Extension reach-back": "release_extension", "Arm-angle reach-back": "arm_angle"}
OTHER = []
OTHERCOL = {}
FBSRC = {"FB velocity reach-back": ["release_speed", "strikes"], "FB put-away velocity": ["release_speed", "strikes", "balls"],
         "Extension reach-back": ["release_extension", "strikes"], "Arm-angle reach-back": ["arm_angle", "strikes"]}
BASE14 = FBP + OTHER
DELIVERY = ["FB velocity reach-back", "FB put-away velocity"]  # the velocity reserve group
'''

GATES = '''
if not FAKE:  # equality gates: rows, labels and the rival do not depend on the parameters (FREEZE_pitch_reach.md)
    _got = (len(rows_m), rows_m.pitcher.nunique(), len(rows_a), rows_a.pitcher.nunique(), int(rows_m.use14.sum()),
            int(rows_m.il14[rows_m.use14].sum()), int(rows_a.use14.sum()), int(rows_a.il14[rows_a.use14].sum()))
    print("equality gates, rows and labels:", _got)
    assert _got == (61966, 1180, 55686, 1853, 53384, 3553, 45453, 1884), "rows or labels differ from the frozen run"
    _tr = (rows_m.season.isin(TRAIN) & rows_m.use14).to_numpy()
    _te = ((rows_m.season == TEST) & rows_m.use14).to_numpy()
    _b0 = auc(rows_m.il14[_te].to_numpy(), predict(fit(CTX, rows_m[_tr], rows_m.il14[_tr].to_numpy()), CTX, rows_m[_te]))
    print(f"equality gate, MLB 2026 AUC(B0): {_b0:.16f} (frozen run 0.5644912856858865)")
    assert abs(_b0 - 0.5644912856858865) < 1e-6, "the rival differs from the frozen run"
p1_df = rows_m[rows_m.use14].reset_index(drop=True)
'''

DIAG = '''          f"mean {np.mean(changes):.2f}, max {max(changes)}; rows without a primary fastball: {int(rows.no_primary.sum()):,}")
    print("  short z defined, by role (0 reliever, 1 starter):\\n"
          + rows.groupby("role")[Z14S].apply(lambda d: d.notna().mean()).T.round(3).to_string())
    print("  short values by role (median):\\n"
          + rows.groupby("role")[P14S].median().T.round(3).to_string())
'''

ROLE = "same role as the previous appearance"
NWAY = 20

HEADER_END = src.index("# %%\nimport glob")
REPL = [
    ("header", src[:HEADER_END], "# %% [markdown]\n# Frozen pitcher notebook (commit 2c6b01d) with the reach-back parameters (see above).\n\n"),
    ("smoke MIN_Z", "BASE_LAG, MIN_BASE, MIN_SCALE, MIN_Z, MIN_D = (0, 2, 3, 4, 2) if SMOKE else (30, 10, 20, 7, 5)",
     "BASE_LAG, MIN_BASE, MIN_SCALE, MIN_Z, MIN_D = (0, 2, 3, 2, 2) if SMOKE else (30, 10, 20, 3, 5)\nIN_MIN = 4\n"
     "REACH_MIN = ((2, 1), (1, 1)) if SMOKE else ((15, 9), (5, 3))  # (two-strike, put-away) for the long and the short window"),
    ("location join", "def load(pattern, years, md5s=None):", LOCATION),
    ("read columns", "        d = pd.read_parquet(p[0], columns=PCOLS)\n",
     '        d = pd.read_parquet(p[0], columns=PCOLS + (["stand", "balls", "strikes"] if pattern.startswith("statcast") else ["stand"]))\n'
     '        if pattern.startswith("aaa"):\n'
     '            d = join_location(d, y)\n'
     '        count_check(d[d.game_type == "R"], pattern.format(y=y))\n'),
    ("fill table", '"arm_angle", "pfx_x", "pfx_z", "pitch_type", "zone"]].apply',
     '"arm_angle", "pfx_x", "pfx_z", "pitch_type", "zone", "balls", "strikes"]].apply'),
    ("section title", "# ## Fourteen parameters per appearance (only earlier pitches, same season and level)",
     "# ## Four reach-back parameters per appearance (only earlier pitches, same season and level)"),
    ("parameter lists", src[src.index('FBP = ["FB velocity", "FB top velocity"'):src.index('SFX = ("", " (short)")')], PARAMS),
    ("fill columns", 'fr = d.groupby("season")[sorted(set(FBCOL.values()) | {"zone", "pitch_type", "description"})]',
     'fr = d.groupby("season")[sorted({c for v in FBSRC.values() for c in v} | {"zone", "pitch_type", "description"})]'),
    ("fill rule", "    DROPPED[name] = sorted(n for n in BASE14 if fr[src[n]] < 0.90)",
     "    DROPPED[name] = sorted(n for n in BASE14 if min(fr[c] for c in FBSRC[n]) < 0.90)"),
    ("new function", "\ndef params_for(g, prev_type):", REACH),
    ("parameter loop", src[src.index("    fb = {}\n    for t in FB:\n"):src.index("    per_day = pd.Series(1, index=gd)")],
     "    fb, oth = reach_series(g, pt, days)\n"),
    ("velocity U2 need", 'df["U2_del"] = u_scores(df[ZDEL].to_numpy(), 5)[1]', 'df["U2_del"] = u_scores(df[ZDEL].to_numpy(), 2)[1]'),
    ("look-ahead columns", '"pfx_x", "pfx_z", "launch_speed"]:\n        g0.loc[late, c] = 999.0',
     '"pfx_x", "pfx_z", "launch_speed"]:\n        g0.loc[late, c] = 999.0\n'
     '    for c in ["release_speed", "release_extension", "arm_angle"]:  # not constant inside an outing\n'
     '        g0.loc[late, c] = 999.0 + g0.loc[late, "pitch_number"].to_numpy(float)\n'
     '    g0.loc[late, "strikes"] = np.where(g0.loc[late, "pitch_number"].to_numpy() % 2 == 1, 2.0, 0.0)  # pitch 1 is not 0-0\n'
     '    g0.loc[late, "balls"] = 1.0'),
    ("look-ahead assert", "    assert any(not np.array_equal(s0[2][n][i0 + 1:], s0b[2][n][i0 + 1:], equal_nan=True) for n in s0[2])\n",
     "    assert not s0[2] and not s0b[2], 'no non-fastball series in this study'\n"
     "    for _sh in (False, True):  # the plant must move a long and a short series after t0\n"
     "        assert any(not np.array_equal(s0[1][n][i0 + 1:], s0b[1][n][i0 + 1:], equal_nan=True)\n"
     "                   for n in s0[1] if n[1].endswith(' (short)') == _sh), 'the plant moves nothing after t0'\n"
     "    # the vectorised parameters equal a plain loop (the count use within a day is not covered by the plant):\n"
     "    # the plant's pitcher-season, up to five with two outings on one date, for each series the first with a value,\n"
     "    # and 60 drawn with seed 0 (all among this level's pitcher-seasons of the row seasons; repeats counted once)\n"
     "    _ks = sorted(k for k in series if k[1] in ROW_SEASONS)\n"
     "    _dh = [k for k in _ks if groups[k].groupby('game_date').game_pk.nunique().max() > 1][:5]\n"
     "    _cv = [next((k for k in _ks if np.isfinite(series[k][1][n]).any()), k0) for n in sorted(series[k0][1])]\n"
     "    _pick = list(dict.fromkeys([k0] + _dh + _cv + [_ks[i] for i in np.random.default_rng(0).choice(len(_ks), min(60, len(_ks)), replace=False)]))\n"
     "    _nd = {}\n"
     "    for _k in _pick:\n"
     "        _p = reach_plain(groups[_k], series[_k][0])\n"
     "        assert set(_p) == set(series[_k][1]), ('reach-back differs from the plain loop', _k, 'parameter names')\n"
     "        for n in series[_k][1]:\n"
     "            _a, _b = series[_k][1][n], _p[n]\n"
     "            assert np.array_equal(np.isnan(_a), np.isnan(_b)), ('reach-back differs from the plain loop', _k, n)\n"
     "            assert np.allclose(_a[~np.isnan(_a)], _b[~np.isnan(_b)], rtol=0, atol=1e-9), ('reach-back differs from the plain loop', _k, n)\n"
     "            _nd[n] = _nd.get(n, 0) + int((~np.isnan(_a)).sum())\n"
     "    print(f'  {level}: reach-back equals the plain loop on {len(_pick)} pitcher-seasons ({len(_dh)} with two outings on one date; '\n"
     "          f'{sum(_nd.values()):,} defined values, fewest for one parameter {min(_nd.values()):,})')\n"
     "    assert len(_nd) == len(FB) * len(FBCOL) * len(SFX) and min(_nd.values()) > 0, ('the plain-loop check left a parameter uncompared', _nd)\n"),
    ("centre kind", '            if n == "Zone rate (short)":', '            if n == "FB velocity reach-back (short)":'),
    ("diagnostics",
     '          f"mean {np.mean(changes):.2f}, max {max(changes)}; rows without a primary fastball: {int(rows.no_primary.sum()):,}")\n',
     DIAG),
    ("gates", "\np1_df = rows_m[rows_m.use14].reset_index(drop=True)\n", GATES),
    ("planted column", 'c0 = Z14S.index("z_FB velocity (short)")', 'c0 = Z14S.index("z_FB velocity reach-back (short)")'),
    ("note P1-P4", 'n_.append("results, not delivery" + (" (a roster move by results)" if key == "P4" else ""))',
     'n_.append("delivery effort, not velocity reserve" + (" (a roster move by delivery effort)" if key == "P4" else ""))'),
    ("groups", 'return "delivery" if base in DELIVERY else "results"', 'return "velocity reserve" if base in DELIVERY else "delivery effort"'),
    ("note P2", '    if grp.loc["delivery", "MLB"] <= 0:\n        n_.append("results, not delivery")',
     f'    if sp.loc["{ROLE}", "M2 - B0"] <= sp.loc["{ROLE}", "floor mean"]:\n        n_.append("role change")\n'
     '    if grp.loc["velocity reserve", "MLB"] <= 0:\n        n_.append("delivery effort, not velocity reserve")'),
    ("note role P1-P4", '    if DEL[key]["auc"] <= DEL[key]["floor"].mean():\n',
     f'    if sub.loc["{ROLE}", "AUC"] <= sub.loc["{ROLE}", "floor mean"]:\n        n_.append("role change")\n'
     '    if DEL[key]["auc"] <= DEL[key]["floor"].mean():\n'),
    ("role subset", '"starters": df.role.eq(1.0).to_numpy(), "relievers": df.role.eq(0.0).to_numpy(),',
     f'"starters": df.role.eq(1.0).to_numpy(), "relievers": df.role.eq(0.0).to_numpy(),\n            "{ROLE}": df.prev_start.eq(df.role).to_numpy(),'),
    ("role subset P2", '    for sn, m_ in (("starters", test_.role.eq(1.0).to_numpy()), ("relievers", test_.role.eq(0.0).to_numpy()),',
     f'    for sn, m_ in (("starters", test_.role.eq(1.0).to_numpy()), ("relievers", test_.role.eq(0.0).to_numpy()),\n'
     f'                   ("{ROLE}", test_.prev_start.eq(test_.role).to_numpy()),'),
    ("20-way bounds", '"lo_bonf": np.percentile(v, LO), "hi_bonf": np.percentile(v, HI)})',
     '"lo_bonf": np.percentile(v, LO), "hi_bonf": np.percentile(v, HI),\n'
     f'                    "lo_n": np.percentile(v, 100 * 0.05 / {NWAY} / 2), "hi_n": np.percentile(v, 100 - 100 * 0.05 / {NWAY} / 2)}})'),
    ("20-way return", '    return {"auc": r.loc["U", "AUC"], "lo": r.loc["U", "lo_bonf"], "hi": r.loc["U", "hi_bonf"], "floor": floor, "ok": ok,',
     '    return {"auc": r.loc["U", "AUC"], "lo": r.loc["U", "lo_bonf"], "hi": r.loc["U", "hi_bonf"], "floor": floor, "ok": ok,\n'
     '            "lo_n": r.loc["U", "lo_n"],'),
    ("20-way summary", '    "floor_max": [P1["floor"].max(), P2["floor"].max(), P3["floor"].max(), P4["floor"].max()],',
     '    "floor_max": [P1["floor"].max(), P2["floor"].max(), P3["floor"].max(), P4["floor"].max()],\n'
     f'    "lo_{NWAY}way": [P1["lo_n"], P2["main"].loc["M2 - B0", "lo_n"], P3["lo_n"], P4["lo_n"]],\n'
     f'    "across five pitcher families ({NWAY}-way)": ["SUPPORTED" if o and l > t else "NOT SUPPORTED" for o, l, t in (\n'
     '        (P1["ok"], P1["lo_n"], 0.5), (P2["ok"], P2["main"].loc["M2 - B0", "lo_n"], 0.0), (P3["ok"], P3["lo_n"], 0.5),\n'
     '        (P4["ok"], P4["lo_n"], 0.5))],'),
    ("section text",
     "# Fastball parameters are separate series per fastball type (FF, SI, FC), each against its own normal; the z used on a day is the one of that day's primary fastball.",
     "# All four are separate series per fastball type (FF, SI, FC), each against its own normal; the z used on a day is the one of that day's primary fastball. Each pitch is measured against the mean of its own outing, so park and tracking shifts between outings cancel."),
    ("docstring", "    Returns days, per-type fastball series {(type, name): values}",
     "    Returns days, per-type fastball reach-back series {(type, name): values}"),
    ("velocity-only labels",
     '"P1 delivery only", 1000, "U2_deld"),\n       "P3": unsupervised_test(arm_df, "P3 delivery only", 2000, "U2_deld"),\n       "P4": unsupervised_test(p4_df, "P4 delivery only"',
     '"V1 velocity reserve only", 1000, "U2_deld"),\n       "P3": unsupervised_test(arm_df, "V2 velocity reserve only", 2000, "U2_deld"),\n       "P4": unsupervised_test(p4_df, "V3 velocity reserve only"'),
    ("roc labels", '(("P1 MLB", "#2471a3", P1), ("P3 MLB arm", "#c0392b", P3), ("P4 Triple-A", "#8a8a8a", P4))',
     '(("V1 MLB", "#2471a3", P1), ("V2 MLB arm", "#c0392b", P3), ("V3 Triple-A", "#8a8a8a", P4))'),
    ("roc title", 'ax.set_title(f"Pitchers: injured list within {H} days")', 'ax.set_title(f"Reach-back: injured list within {H} days")'),
    ("roc file", 'plt.savefig("pitcher_roc.png", dpi=120)', 'plt.savefig("reach_roc.png", dpi=120)'),
    ("floor labels", '["P1 MLB", "P3 MLB arm", "P4 Triple-A"]', '["V1 MLB", "V2 MLB arm", "V3 Triple-A"]'),
    ("floor dots", 'enumerate((("P1", P1), ("P3", P3), ("P4", P4)))', 'enumerate((("V1", P1), ("V2", P3), ("V3", P4)))'),
    ("floor file", 'plt.savefig("pitcher_floors.png", dpi=120)', 'plt.savefig("reach_floors.png", dpi=120)'),
    ("summary labels",
     '"test": ["P1 MLB U2 daily rank, 2024-2026", "P2 MLB M2 over B0, 2026", "P3 MLB arm, U2 daily rank", "P4 Triple-A U2 daily rank"]',
     '"test": ["V1 MLB U2 daily rank, 2024-2026", "V4 MLB M2 over B0, 2026", "V2 MLB arm, U2 daily rank", "V3 Triple-A U2 daily rank"]'),
    ("verdict file", 'summary.to_csv("pitcher_verdict.csv", index=False)', 'summary.to_csv("reach_verdict.csv", index=False)'),
    ("auc file", 'P2["main"].to_csv("pitcher_auc_mlb_2026.csv")', 'P2["main"].to_csv("reach_auc_mlb_2026.csv")'),
]
for name, old, new in REPL:
    assert src.count(old) == 1, f"replacement '{name}' matches {src.count(old)} times"
    src = src.replace(old, new)
print(f"{len(REPL)} replacements applied")
exec(compile(src, "frozen_pitcher_reach", "exec"))
