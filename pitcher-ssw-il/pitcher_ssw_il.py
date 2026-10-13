# %% [markdown]
# # Does a Pitcher's Spin-Axis Gap Drift Before the Injured List?
#
# Five [pre-registered](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch.md) [pitcher](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_inouting.md) [studies](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_mix.md) [found](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_command.md) [nothing](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_reach.md) before the injured list in the fastball's levels, its fade and scatter inside an outing, the pitch mix, command or two-strike reach-back. None of them measured the part of a pitch's movement that the spin does not explain. Hawk-Eye reports the spin axis it observes; the movement points to the axis that spin alone would give. The difference, the **spin-axis gap** (seam-shifted wake), depends on grip, seam orientation and release:
#
#     gap = wrap(spin_axis - (180 - degrees(atan2(pfx_x, pfx_z))))   in [-180, 180)
#
# - **Fastball**: mean gap over his last 150 / 50 pitches of the type (FF, SI or FC).
# - **Secondary**: the same over his last 100 / 30 changeups or splitters (CH, FS, FO) and breaking balls (SL, ST, SV, CU, KC, CS).
#
# Each pitch type is its own series against his own normal; the z of a day is the one of that family's primary type (most pitches this season before the day). It runs the frozen pitcher notebook (commit `2c6b01d`, checked by md5) with **only its parameters replaced**; rows, labels, the rival model, floors and tests are unchanged, and equality gates check that. Everything was fixed before any new parameter was compared with a placement: [FREEZE_pitch_ssw.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_ssw.md).
#
# Names in the printed output are the frozen code's: **P1 = S1** (MLB, all placements), **P3 = S2** (MLB, arm placements), **P4 = S3** (Triple-A), **P2 = S4** (MLB 2026 model over the rival). "Delivery" in the frozen printouts is now the **fastball** group; "results" is the **secondary** group.
#
# This is the sixth pre-registered family of tests on the same pitcher placements, so the summary also shows each verdict at a 24-way Bonferroni reading across the six families (0.104% and 99.896%); the 4-way verdict is "supported within this family", not across the six.

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

PARAMS = '''FBP = []  # no parameter of the frozen per-fastball kind in this study
FAM = {"Fastball": ("FF", "SI", "FC"), "Changeup": ("CH", "FS", "FO"), "Breaking": ("SL", "ST", "SV", "CU", "KC", "CS")}
FAMW = {"Fastball": 0, "Changeup": 1, "Breaking": 1}  # index into LONG / SHORT: 150 / 50 fastballs, 100 / 30 of the others
FAMCOL = {f + " gap": "spin_axis" for f in FAM}
FAMP = {n: n.split(" ")[0] for n in FAMCOL}
FBCOL = dict(FAMCOL)
OTHER = []
OTHERCOL = {}
FBSRC = {n: ["spin_axis", "pfx_x", "pfx_z"] for n in FAMCOL}
BASE14 = list(FAMCOL)
DELIVERY = ["Fastball gap"]  # the fastball group; changeup and breaking are the secondary group


def ssw_gap(axis, px, pz):
    """Observed spin axis minus the axis implied by the movement direction, wrapped to [-180, 180) degrees."""
    inferred = 180.0 - np.degrees(np.arctan2(px, pz))
    return (axis - inferred + 180.0) % 360.0 - 180.0


# Convention gate (label-free): with the right sign the typical gap of a pitcher's pitch type is small; the mirrored
# convention (180 + atan2) is far off. Code-test run: 15.7 and 86.6 degrees (MLB), 14.9 and 91.9 (Triple-A).
for name, d in (("MLB", mlb), ("Triple-A", aaa)):
    a_, x_, z_ = d.spin_axis.to_numpy(float), d.pfx_x.to_numpy(float), d.pfx_z.to_numpy(float)
    gg = ssw_gap(a_, x_, z_)
    fin = np.isfinite(gg)
    assert ((gg[fin] >= -180.0) & (gg[fin] < 180.0)).all(), f"{name}: gap outside [-180, 180)"
    mirror = (a_ - (180.0 + np.degrees(np.arctan2(x_, z_))) + 180.0) % 360.0 - 180.0
    key = [d.pitcher.to_numpy()[fin], d.season.to_numpy()[fin], d.pitch_type.fillna("").to_numpy()[fin]]
    per = pd.DataFrame({"g": gg[fin], "m": mirror[fin]}).groupby(key).agg(["median", "size"])
    per = per[per[("g", "size")] >= 30]
    conv, mirr = per[("g", "median")].abs().median(), per[("m", "median")].abs().median()
    print(f"{name}: gap defined on {fin.mean():.4f} of pitches; median over {len(per):,} pitcher-season-types of "
          f"|median gap| {conv:.1f} deg (mirrored convention {mirr:.1f})")
    assert conv <= 25.0 and mirr >= 40.0, f"{name}: the gap convention is not the expected one"
'''

LOOP = '''    fb = {}  # per type of each family: (type, name) -> series; ("choice", family) -> primary type index per day
    gap = ssw_gap(g.spin_axis.to_numpy(float), g.pfx_x.to_numpy(float), g.pfx_z.to_numpy(float))
    for fam, types in FAM.items():
        bef = np.vstack([np.searchsorted(gd[pt == t], days, side="left") for t in types])
        fb[("choice", fam)] = np.where(bef.max(0) > 0, bef.argmax(0), -1)
        for t in types:
            m_ = pt == t
            for sfx, w in zip(SFX, (LONG[FAMW[fam]], SHORT[FAMW[fam]])):
                fb[(t, fam + " gap" + sfx)] = wmean(gd[m_], gap[m_], w, days)
    oth = {}
'''

PLAIN = '''
def ssw_plain(g, days):
    """The six gap series and the three primary-type choices by a plain loop over days (a second implementation with
    its own trigonometry and its own window table; the gate in build_rows compares)."""
    assert SMOKE or (LONG[0], SHORT[0], LONG[1], SHORT[1]) == (150, 50, 100, 30), "not the frozen windows"
    windows = {"Fastball": (LONG[0], SHORT[0]), "Changeup": (LONG[1], SHORT[1]), "Breaking": (LONG[1], SHORT[1])}
    out = {}
    gd = list(g.game_date)
    pt = list(g.pitch_type.fillna(""))
    ax, px, pz = list(g.spin_axis.astype(float)), list(g.pfx_x.astype(float)), list(g.pfx_z.astype(float))

    def one(a, x, z):
        if math.isnan(a) or math.isnan(x) or math.isnan(z):
            return float("nan")
        return ((a - (180.0 - math.degrees(math.atan2(x, z))) + 180.0) % 360.0) - 180.0

    gaps = [one(a, x, z) for a, x, z in zip(ax, px, pz)]
    for fam, types in FAM.items():
        nbefore = {}  # type -> number of its pitches strictly before each day (pitches are in date order)
        for t in types:
            idx = [i for i in range(len(gd)) if pt[i] == t]
            counts, j = [], 0
            for day in days:
                while j < len(idx) and gd[idx[j]] < day:
                    j += 1
                counts.append(j)
            nbefore[t] = counts
            for sfx, w in zip(SFX, windows[fam]):
                res = []
                for c in counts:
                    if c < w:
                        res.append(float("nan"))
                        continue
                    v = [gaps[i] for i in idx[c - w:c] if not math.isnan(gaps[i])]
                    res.append(sum(v) / len(v) if len(v) >= w / 2 else float("nan"))
                out[(t, fam + " gap" + sfx)] = np.array(res)
        ch = []
        for d_ in range(len(days)):
            cnt = [nbefore[t][d_] for t in types]
            ch.append(cnt.index(max(cnt)) if max(cnt) > 0 else -1)
        out[("choice", fam)] = np.array(ch)
    return out


def params_for(g, prev_type):'''

ZSEL = '''            if base in FAMP:
                types = FAM[FAMP[base]]
                ch = fb[("choice", FAMP[base])]
                zt = []
                for t in types:
                    x = fb[(t, n)]
                    pv = prev[1].get((t, n)) if prev is not None else None
                    own = mad(pv[np.isfinite(pv)]) if pv is not None and np.isfinite(pv).sum() >= MIN_SCALE else np.nan
                    sc = np.where(np.isfinite(own) & (own > 0) & (role == prole), own, lsc)
                    zt.append(own_z(days, x, pv, sc))
                r[n] = np.where(ch >= 0, np.vstack([fb[(t, n)] for t in types])[np.maximum(ch, 0), np.arange(len(days))], np.nan)
                zz = np.vstack([z_[0] for z_ in zt])
                kk = np.vstack([z_[1] for z_ in zt])
                r["z_" + n] = np.where(ch >= 0, zz[np.maximum(ch, 0), np.arange(len(days))], np.nan)
                kind = np.where(ch >= 0, kk[np.maximum(ch, 0), np.arange(len(days))], 0)
'''

SELGATE = '''    # each row's value of a family is the plain loop's series of that day's primary type (plain-loop pitcher-seasons)
    for _k in _pick:
        if _k[0] == OHTANI and _k[1] == TEST:
            continue
        _p = ssw_plain(groups[_k], series[_k][0])
        _r = rows[(rows.pitcher == _k[0]) & (rows.season == _k[1])]
        assert len(_r) == len(series[_k][0]), ("selection differs", _k, "rows")
        for n in P14 + P14S:
            _f = FAMP[n.replace(" (short)", "")]
            _e = np.array([_p[(FAM[_f][c], n)][i] if c >= 0 else np.nan for i, c in enumerate(_p[("choice", _f)])])
            assert np.allclose(_r[n].to_numpy(float), _e, rtol=0, atol=1e-9, equal_nan=True), ("selection differs", _k, n)
            assert not (np.isfinite(_r["z_" + n].to_numpy(float)) & ~np.isfinite(_r[n].to_numpy(float))).any(), ("selection differs", _k, "z without value " + n)
    print(f"  {level}: each family's row value equals the plain loop's primary-type series on {len(_pick)} pitcher-seasons")
    rows = recompute_scores(rows)
'''

GATES = '''
if not FAKE:  # equality gates: rows, labels and the rival do not depend on the parameters (FREEZE_pitch_ssw.md)
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
    print("  short values by role (median, degrees):\\n"
          + rows.groupby("role")[P14S].median().T.round(3).to_string())
'''

ROLE = "same role as the previous appearance"
NWAY = 24

HEADER_END = src.index("# %%\nimport glob")
REPL = [
    ("header", src[:HEADER_END], "# %% [markdown]\n# Frozen pitcher notebook (commit 2c6b01d) with the spin-axis gap parameters (see above).\n\n"),
    ("imports", "# %%\nimport glob", "# %%\nimport math\nimport glob"),
    ("smoke MIN_Z", "BASE_LAG, MIN_BASE, MIN_SCALE, MIN_Z, MIN_D = (0, 2, 3, 4, 2) if SMOKE else (30, 10, 20, 7, 5)",
     "BASE_LAG, MIN_BASE, MIN_SCALE, MIN_Z, MIN_D = (0, 2, 3, 2, 2) if SMOKE else (30, 10, 20, 2, 5)"),
    ("read columns", "        d = pd.read_parquet(p[0], columns=PCOLS)\n",
     '        d = pd.read_parquet(p[0], columns=PCOLS + ["spin_axis"])\n'),
    ("fill table", '"arm_angle", "pfx_x", "pfx_z", "pitch_type", "zone"]].apply',
     '"arm_angle", "pfx_x", "pfx_z", "pitch_type", "zone", "spin_axis"]].apply'),
    ("section title", "# ## Fourteen parameters per appearance (only earlier pitches, same season and level)",
     "# ## Six spin-axis gap parameters per appearance (only earlier pitches, same season and level)"),
    ("parameter lists", src[src.index('FBP = ["FB velocity", "FB top velocity"'):src.index('SFX = ("", " (short)")')], PARAMS),
    ("fill columns", 'fr = d.groupby("season")[sorted(set(FBCOL.values()) | {"zone", "pitch_type", "description"})]',
     'fr = d.groupby("season")[sorted({c for v in FBSRC.values() for c in v} | {"zone", "pitch_type", "description"})]'),
    ("fill rule", "    DROPPED[name] = sorted(n for n in BASE14 if fr[src[n]] < 0.90)",
     "    DROPPED[name] = sorted(n for n in BASE14 if min(fr[c] for c in FBSRC[n]) < 0.90)"),
    ("new function", "\ndef params_for(g, prev_type):", PLAIN),
    ("parameter loop", src[src.index("    fb = {}\n    for t in FB:\n"):src.index("    per_day = pd.Series(1, index=gd)")], LOOP),
    ("fastball U2 need", 'df["U2_del"] = u_scores(df[ZDEL].to_numpy(), 5)[1]', 'df["U2_del"] = u_scores(df[ZDEL].to_numpy(), 1)[1]'),
    ("family modes init", "    modes, fbvel = {}, {}\n", "    modes, fbvel, fmode = {}, {}, {}\n"),
    ("family modes", "        modes[k] = FB[int(np.argmax(cnt))] if max(cnt) > 0 else None\n",
     "        modes[k] = FB[int(np.argmax(cnt))] if max(cnt) > 0 else None\n"
     "        for fam, types in FAM.items():\n"
     "            cf = [c.get(t, 0) for t in types]\n"
     "            fmode[(k, fam)] = types[int(np.argmax(cf))] if max(cf) > 0 else None\n"),
    ("league scale", 'v = s[1].get((modes[k], n)) if n.replace(" (short)", "") in FBP else s[2][n]',
     'v = s[1].get((fmode[(k, FAMP[n.replace(" (short)", "")])], n)) if n.replace(" (short)", "") in FAMP else s[2][n]'),
    ("z by primary type", src[src.index("            if base in FBP:\n"):src.index("            else:\n                x = oth[n]")], ZSEL),
    ("look-ahead columns", '"pfx_x", "pfx_z", "launch_speed"]:\n        g0.loc[late, c] = 999.0',
     '"pfx_x", "pfx_z", "launch_speed"]:\n        g0.loc[late, c] = 999.0\n'
     '    g0.loc[late, "spin_axis"] = 30.0 + g0.loc[late, "pitch_number"].to_numpy(float)  # not constant inside an outing'),
    ("look-ahead assert", "    assert any(not np.array_equal(s0[2][n][i0 + 1:], s0b[2][n][i0 + 1:], equal_nan=True) for n in s0[2])\n",
     "    assert not s0[2] and not s0b[2], 'no other-series parameters in this study'\n"
     "    for _sh in (False, True):  # the plant must move a long and a short gap series after t0\n"
     "        assert any(not np.array_equal(s0[1][n][i0 + 1:], s0b[1][n][i0 + 1:], equal_nan=True)\n"
     "                   for n in s0[1] if n[0] != 'choice' and n[1].endswith(' (short)') == _sh), 'the plant moves nothing after t0'\n"
     "    # the vectorised series equal a plain loop: the plant's pitcher-season, up to five with two outings on one\n"
     "    # date, for each series the first pitcher-season with a value, and 60 drawn with seed 0 (row seasons only)\n"
     "    _ks = sorted(k for k in series if k[1] in ROW_SEASONS)\n"
     "    _dh = [k for k in _ks if groups[k].groupby('game_date').game_pk.nunique().max() > 1][:5]\n"
     "    _names = sorted(n for n in series[k0][1] if n[0] != 'choice')\n"
     "    _cv = [next((k for k in _ks if np.isfinite(series[k][1][n]).any()), k0) for n in _names]\n"
     "    _pick = list(dict.fromkeys([k0] + _dh + _cv + [_ks[i] for i in np.random.default_rng(0).choice(len(_ks), min(60, len(_ks)), replace=False)]))\n"
     "    _nd = {}\n"
     "    for _k in _pick:\n"
     "        _p = ssw_plain(groups[_k], series[_k][0])\n"
     "        assert set(_p) == set(series[_k][1]), ('the gap differs from the plain loop', _k, 'series names')\n"
     "        for n in series[_k][1]:\n"
     "            _a, _b = series[_k][1][n], _p[n]\n"
     "            if n[0] == 'choice':\n"
     "                assert np.array_equal(_a, _b), ('the gap differs from the plain loop', _k, n)\n"
     "                continue\n"
     "            assert np.array_equal(np.isnan(_a), np.isnan(_b)), ('the gap differs from the plain loop', _k, n)\n"
     "            assert np.allclose(_a[~np.isnan(_a)], _b[~np.isnan(_b)], rtol=0, atol=1e-9), ('the gap differs from the plain loop', _k, n)\n"
     "            _nd[n] = _nd.get(n, 0) + int((~np.isnan(_a)).sum())\n"
     "    _any = {n for k in _ks for n in _names if np.isfinite(series[k][1][n]).any()}\n"
     "    print(f'  {level}: gap equals the plain loop on {len(_pick)} pitcher-seasons ({len(_dh)} with two outings on one date; '\n"
     "          f'{sum(_nd.values()):,} defined values; series never defined at this level: {sorted(set(_names) - _any) or \"none\"})')\n"
     "    assert {(t, f + ' gap' + s) for t, f in (('FF', 'Fastball'), ('CH', 'Changeup'), ('SL', 'Breaking')) for s in SFX} <= _any, \\\n"
     "        'a main series is never defined'\n"),
    ("league scale init", "    lscale = {}\n", "    lscale, lcount = {}, {}\n"),
    ("league scale count", "            lscale[(n, role)] = float(np.median(ms)) if ms else np.nan\n",
     "            lscale[(n, role)] = float(np.median(ms)) if ms else np.nan\n"
     "            lcount[(n, role)] = len(ms)\n"),
    ("league scale gate", "    recs, changes = [], []\n",
     "    print(f'  {level}: pitcher-seasons behind each league-typical scale (reliever / starter):',\n"
     "          {n: (lcount[(n, 0.0)], lcount[(n, 1.0)]) for n in ZPARAMS})\n"
     "    # real run only, like the equality gates: the 100-pitcher code-test sample has no Triple-A 2023 starter with a\n"
     "    # long changeup series (the full file has six before the roster and rehab filters)\n"
     "    assert FAKE or all(np.isfinite(v) for v in lscale.values()), ('a league-typical scale is missing', lscale)\n"
     "    recs, changes = [], []\n"),
    ("selection gate", "    rows = recompute_scores(rows)\n", SELGATE),
    ("centre kind", '            if n == "Zone rate (short)":', '            if n == "Fastball gap (short)":'),
    ("diagnostics",
     '          f"mean {np.mean(changes):.2f}, max {max(changes)}; rows without a primary fastball: {int(rows.no_primary.sum()):,}")\n',
     DIAG),
    ("gates", "\np1_df = rows_m[rows_m.use14].reset_index(drop=True)\n", GATES),
    ("planted column", 'c0 = Z14S.index("z_FB velocity (short)")', 'c0 = Z14S.index("z_Fastball gap (short)")'),
    ("note P1-P4", 'n_.append("results, not delivery" + (" (a roster move by results)" if key == "P4" else ""))',
     'n_.append("secondary, not fastball" + (" (a roster move by secondary pitches)" if key == "P4" else ""))'),
    ("groups", 'return "delivery" if base in DELIVERY else "results"', 'return "fastball" if base in DELIVERY else "secondary"'),
    ("note P2", '    if grp.loc["delivery", "MLB"] <= 0:\n        n_.append("results, not delivery")',
     f'    if sp.loc["{ROLE}", "M2 - B0"] <= sp.loc["{ROLE}", "floor mean"]:\n        n_.append("role change")\n'
     '    if grp.loc["fastball", "MLB"] <= 0:\n        n_.append("secondary, not fastball")'),
    ("note role P1-P4", '    if DEL[key]["auc"] <= DEL[key]["floor"].mean():\n',
     f'    if sub.loc["{ROLE}", "AUC"] <= sub.loc["{ROLE}", "floor mean"]:\n        n_.append("role change")\n'
     '    if DEL[key]["auc"] <= DEL[key]["floor"].mean():\n'),
    ("role subset", '"starters": df.role.eq(1.0).to_numpy(), "relievers": df.role.eq(0.0).to_numpy(),',
     f'"starters": df.role.eq(1.0).to_numpy(), "relievers": df.role.eq(0.0).to_numpy(),\n            "{ROLE}": df.prev_start.eq(df.role).to_numpy(),'),
    ("role subset P2", '    for sn, m_ in (("starters", test_.role.eq(1.0).to_numpy()), ("relievers", test_.role.eq(0.0).to_numpy()),',
     f'    for sn, m_ in (("starters", test_.role.eq(1.0).to_numpy()), ("relievers", test_.role.eq(0.0).to_numpy()),\n'
     f'                   ("{ROLE}", test_.prev_start.eq(test_.role).to_numpy()),'),
    ("24-way bounds", '"lo_bonf": np.percentile(v, LO), "hi_bonf": np.percentile(v, HI)})',
     '"lo_bonf": np.percentile(v, LO), "hi_bonf": np.percentile(v, HI),\n'
     f'                    "lo_n": np.percentile(v, 100 * 0.05 / {NWAY} / 2), "hi_n": np.percentile(v, 100 - 100 * 0.05 / {NWAY} / 2)}})'),
    ("24-way return", '    return {"auc": r.loc["U", "AUC"], "lo": r.loc["U", "lo_bonf"], "hi": r.loc["U", "hi_bonf"], "floor": floor, "ok": ok,',
     '    return {"auc": r.loc["U", "AUC"], "lo": r.loc["U", "lo_bonf"], "hi": r.loc["U", "hi_bonf"], "floor": floor, "ok": ok,\n'
     '            "lo_n": r.loc["U", "lo_n"],'),
    ("24-way summary", '    "floor_max": [P1["floor"].max(), P2["floor"].max(), P3["floor"].max(), P4["floor"].max()],',
     '    "floor_max": [P1["floor"].max(), P2["floor"].max(), P3["floor"].max(), P4["floor"].max()],\n'
     f'    "lo_{NWAY}way": [P1["lo_n"], P2["main"].loc["M2 - B0", "lo_n"], P3["lo_n"], P4["lo_n"]],\n'
     f'    "across six pitcher families ({NWAY}-way)": ["SUPPORTED" if o and l > t else "NOT SUPPORTED" for o, l, t in (\n'
     '        (P1["ok"], P1["lo_n"], 0.5), (P2["ok"], P2["main"].loc["M2 - B0", "lo_n"], 0.0), (P3["ok"], P3["lo_n"], 0.5),\n'
     '        (P4["ok"], P4["lo_n"], 0.5))],'),
    ("section text",
     "# Fastball parameters are separate series per fastball type (FF, SI, FC), each against its own normal; the z used on a day is the one of that day's primary fastball.",
     "# All six are separate series per pitch type, each against its own normal; the z used on a day is the one of that family's primary type (most pitches this season before the day). The rival's primary fastball is the frozen one."),
    ("docstring", "    Returns days, per-type fastball series {(type, name): values}",
     "    Returns days, per-type gap series {(type, name): values} with each family's primary-type choice"),
    ("fastball-only labels",
     '"P1 delivery only", 1000, "U2_deld"),\n       "P3": unsupervised_test(arm_df, "P3 delivery only", 2000, "U2_deld"),\n       "P4": unsupervised_test(p4_df, "P4 delivery only"',
     '"S1 fastball only", 1000, "U2_deld"),\n       "P3": unsupervised_test(arm_df, "S2 fastball only", 2000, "U2_deld"),\n       "P4": unsupervised_test(p4_df, "S3 fastball only"'),
    ("roc labels", '(("P1 MLB", "#2471a3", P1), ("P3 MLB arm", "#c0392b", P3), ("P4 Triple-A", "#8a8a8a", P4))',
     '(("S1 MLB", "#2471a3", P1), ("S2 MLB arm", "#c0392b", P3), ("S3 Triple-A", "#8a8a8a", P4))'),
    ("roc title", 'ax.set_title(f"Pitchers: injured list within {H} days")', 'ax.set_title(f"Spin-axis gap: injured list within {H} days")'),
    ("roc file", 'plt.savefig("pitcher_roc.png", dpi=120)', 'plt.savefig("ssw_roc.png", dpi=120)'),
    ("floor labels", '["P1 MLB", "P3 MLB arm", "P4 Triple-A"]', '["S1 MLB", "S2 MLB arm", "S3 Triple-A"]'),
    ("floor dots", 'enumerate((("P1", P1), ("P3", P3), ("P4", P4)))', 'enumerate((("S1", P1), ("S2", P3), ("S3", P4)))'),
    ("floor file", 'plt.savefig("pitcher_floors.png", dpi=120)', 'plt.savefig("ssw_floors.png", dpi=120)'),
    ("summary labels",
     '"test": ["P1 MLB U2 daily rank, 2024-2026", "P2 MLB M2 over B0, 2026", "P3 MLB arm, U2 daily rank", "P4 Triple-A U2 daily rank"]',
     '"test": ["S1 MLB U2 daily rank, 2024-2026", "S4 MLB M2 over B0, 2026", "S2 MLB arm, U2 daily rank", "S3 Triple-A U2 daily rank"]'),
    ("verdict file", 'summary.to_csv("pitcher_verdict.csv", index=False)', 'summary.to_csv("ssw_verdict.csv", index=False)'),
    ("auc file", 'P2["main"].to_csv("pitcher_auc_mlb_2026.csv")', 'P2["main"].to_csv("ssw_auc_mlb_2026.csv")'),
]
for name, old, new in REPL:
    assert src.count(old) == 1, f"replacement '{name}' matches {src.count(old)} times"
    src = src.replace(old, new)
print(f"{len(REPL)} replacements applied")
exec(compile(src, "frozen_pitcher_ssw", "exec"))
