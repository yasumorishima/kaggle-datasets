# %% [markdown]
# # Does a Pitcher's Pitch Mix or Secondary-Pitch Quality Drift Before the Injured List?
#
# Two [pre-registered](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch.md) [pitcher studies](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_inouting.md) found that neither the window levels of the fastball and delivery nor the fade and scatter inside an outing drift from a pitcher's own normal before the injured list. Both measured the fastball; the other pitches entered only as one pooled off-speed velocity. This notebook measures what they did not:
#
# - **Pitch mix**: the share of breaking balls and the share of changeups and splitters among a pitcher's pitches.
# - **Secondary-pitch quality**: velocity, spin and movement of the pitcher's primary breaking ball and primary changeup or splitter, each type against its own normal.
#
# It runs the frozen pitcher notebook (commit `2c6b01d`, checked by md5) with **only its parameters replaced** by these ten; rows, labels, the rival model, floors and tests are unchanged, and equality gates check that. Everything was fixed before any new parameter was compared with a placement: [FREEZE_pitch_mix.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_mix.md).
#
# Names in the printed output are the frozen code's: **P1 = R1** (MLB, all placements), **P3 = R2** (MLB, arm placements), **P4 = R3** (Triple-A), **P2 = R4** (MLB 2026 model over the rival). "Delivery" in the frozen printouts is now the **pitch quality** group (the eight secondary-pitch parameters); "results" is the **mix** group.
#
# This is the third pre-registered family of tests on the same pitcher placements, so the summary also shows each verdict at a 12-way Bonferroni reading across the three families (0.208% and 99.79%); the 4-way verdict is "supported within this family", not across the three.

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

PARAMS = '''FBP = []  # no fastball parameters in this study
FAM = {"Breaking": ("SL", "ST", "SV", "CU", "KC", "CS"), "Changeup": ("CH", "FS", "FO")}
FAMCOL = {"Breaking velocity": "release_speed", "Breaking spin": "release_spin_rate",
          "Breaking vertical movement": "pfx_z", "Breaking horizontal movement": "pfx_x",
          "Changeup velocity": "release_speed", "Changeup spin": "release_spin_rate",
          "Changeup vertical movement": "pfx_z", "Changeup horizontal movement": "pfx_x"}
FAMP = {n: n.split(" ")[0] for n in FAMCOL}
FBCOL = dict(FAMCOL)
OTHER = ["Breaking share", "Changeup share"]
OTHERCOL = {"Breaking share": "pitch_type", "Changeup share": "pitch_type"}
FBSRC = {**{n: [c] for n, c in FAMCOL.items()}, **{n: [c] for n, c in OTHERCOL.items()}}
BASE14 = OTHER + list(FAMCOL)
DELIVERY = list(FAMCOL)
'''

LOOP = '''    fb = {}  # per type of each family: (type, name) -> series; ("choice", family) -> primary type index per day
    for fam, types in FAM.items():
        bef = np.vstack([np.searchsorted(gd[pt == t], days, side="left") for t in types])
        fb[("choice", fam)] = np.where(bef.max(0) > 0, bef.argmax(0), -1)
        for t in types:
            s_ = g[pt == t]
            sd = s_.game_date.to_numpy()
            for sfx, w in zip(SFX, (LONG[1], SHORT[1])):
                for n, c in FAMCOL.items():
                    if FAMP[n] == fam:
                        fb[(t, n + sfx)] = wmean(sd, s_[c].to_numpy(float), w, days)
    oth = {}
    known = (pt != "") & ~np.isin(pt, NOT_OFFSPEED)
    for fam, types in FAM.items():
        v = np.where(known, np.isin(pt, types).astype(float), np.nan)
        for sfx, w in zip(SFX, (LONG[2], SHORT[2])):
            oth[fam + " share" + sfx] = wmean(gd, v, w, days)
'''

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

GATES = '''
if not FAKE:  # equality gates: rows, labels and the rival do not depend on the parameters (FREEZE_pitch_mix.md)
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

LOOKAHEAD = '''    assert any(not np.array_equal(s0[2][n][i0 + 1:], s0b[2][n][i0 + 1:], equal_nan=True) for n in s0[2])
    # family primary-type choice recomputed independently on every pitcher-season and day (FREEZE_pitch_mix.md)
    n_choice = 0
    for k, g in groups.items():
        gd_, pt_ = g.game_date.to_numpy(), g.pitch_type.fillna("").to_numpy(dtype=object)
        for fam, types in FAM.items():
            exp = []
            for d in series[k][0]:
                cnt = [int(np.sum((pt_ == t) & (gd_ < d))) for t in types]
                exp.append(int(np.argmax(cnt)) if max(cnt) > 0 else -1)
            assert np.array_equal(np.array(exp, dtype=int), series[k][1][("choice", fam)]), (k, fam)
            n_choice += len(exp)
    print(f"  {level}: family primary-type choice recomputed on {n_choice:,} pitcher-season-days, identical")
    # second planted look-ahead check: from t0 on, pitches alternate between each family's least-used type before t0
    both = sorted((k for k in series if k[1] in ROW_SEASONS and len(series[k][0]) >= 10
                   and all(np.isin(groups[k].pitch_type, types).any() for types in FAM.values())),
                  key=lambda k: (-len(series[k][0]), k))[:20]
    assert len(both) == 20, "fewer than 20 pitcher-seasons with both families for the second look-ahead check"
    for k in both:
        s1 = series[k]
        t1 = s1[0][len(s1[0]) // 2]
        i1 = list(s1[0]).index(t1)
        g1 = groups[k].copy()
        late1 = (g1.game_date >= t1).to_numpy()
        pre1 = g1.pitch_type[~late1].value_counts()
        rare = [min(types, key=lambda q: (pre1.get(q, 0), types.index(q))) for types in FAM.values()]
        for c in ["release_speed", "release_spin_rate", "pfx_x", "pfx_z"]:
            g1.loc[late1, c] = 999.0
        li = np.flatnonzero(late1)
        g1.iloc[li[0::2], g1.columns.get_loc("pitch_type")] = rare[0]
        g1.iloc[li[1::2], g1.columns.get_loc("pitch_type")] = rare[1]
        s1b = params_for(g1, modes.get((k[0], k[1] - 1)))
        for key in s1[1]:
            assert np.array_equal(s1[1][key][: i1 + 1], s1b[1][key][: i1 + 1], equal_nan=True), (k, key)
        for key in s1[2]:
            assert np.array_equal(s1[2][key][: i1 + 1], s1b[2][key][: i1 + 1], equal_nan=True), (k, key)
        for fam, types in FAM.items():
            assert any(not np.array_equal(s1[1][key][i1 + 1:], s1b[1][key][i1 + 1:], equal_nan=True)
                       for key in s1[1] if key[0] in types), (k, fam)
'''

FAMDIAG = '''          f"mean {np.mean(changes):.2f}, max {max(changes)}; rows without a primary fastball: {int(rows.no_primary.sum()):,}")
    for fam in FAM:
        fc = [int(np.sum(np.diff(s[1][("choice", fam)][s[1][("choice", fam)] >= 0]) != 0))
              for k, s in series.items() if k[1] in ROW_SEASONS]
        print(f"  {fam}: primary-type changes per pitcher-season mean {np.mean(fc):.2f}, max {max(fc)}; "
              f"rows without a short {fam.lower()} velocity z: {int(rows['z_' + fam + ' velocity (short)'].isna().sum()):,}")
'''

ROLE = "same role as the previous appearance"

HEADER_END = src.index("# %%\nimport glob")
REPL = [
    ("header", src[:HEADER_END], "# %% [markdown]\n# Frozen pitcher notebook (commit 2c6b01d) with the pitch-mix and secondary-pitch parameters (see above).\n\n"),
    ("smoke MIN_Z", "BASE_LAG, MIN_BASE, MIN_SCALE, MIN_Z, MIN_D = (0, 2, 3, 4, 2) if SMOKE else (30, 10, 20, 7, 5)",
     "BASE_LAG, MIN_BASE, MIN_SCALE, MIN_Z, MIN_D = (0, 2, 3, 3, 2) if SMOKE else (30, 10, 20, 5, 5)"),
    ("section title", "# ## Fourteen parameters per appearance (only earlier pitches, same season and level)",
     "# ## Ten pitch-mix and secondary-pitch parameters per appearance (only earlier pitches, same season and level)"),
    ("parameter lists", src[src.index('FBP = ["FB velocity", "FB top velocity"'):src.index("SFX = (\"\", \" (short)\")")], PARAMS),
    ("fill rule", "    DROPPED[name] = sorted(n for n in BASE14 if fr[src[n]] < 0.90)",
     "    DROPPED[name] = sorted(n for n in BASE14 if min(fr[c] for c in FBSRC[n]) < 0.90)"),
    ("parameter loop", src[src.index("    fb = {}\n    for t in FB:\n"):src.index("    per_day = pd.Series(1, index=gd)")], LOOP),
    ("quality U2 need", 'df["U2_del"] = u_scores(df[ZDEL].to_numpy(), 5)[1]', 'df["U2_del"] = u_scores(df[ZDEL].to_numpy(), 4)[1]'),
    ("family modes init", "    modes, fbvel = {}, {}\n", "    modes, fbvel, fmode = {}, {}, {}\n"),
    ("family modes", "        modes[k] = FB[int(np.argmax(cnt))] if max(cnt) > 0 else None\n",
     "        modes[k] = FB[int(np.argmax(cnt))] if max(cnt) > 0 else None\n"
     "        for fam, types in FAM.items():\n"
     "            cf = [c.get(t, 0) for t in types]\n"
     "            fmode[(k, fam)] = types[int(np.argmax(cf))] if max(cf) > 0 else None\n"),
    ("league scale", 'v = s[1].get((modes[k], n)) if n.replace(" (short)", "") in FBP else s[2][n]',
     'v = s[1].get((fmode[(k, FAMP[n.replace(" (short)", "")])], n)) if n.replace(" (short)", "") in FAMP else s[2][n]'),
    ("z by primary type", src[src.index("            if base in FBP:\n"):src.index("            else:\n                x = oth[n]")], ZSEL),
    ("centre kind", '            if n == "Zone rate (short)":', '            if n == "Breaking share (short)":'),
    ("gates", "\np1_df = rows_m[rows_m.use14].reset_index(drop=True)\n", GATES),
    ("planted column", 'c0 = Z14S.index("z_FB velocity (short)")', 'c0 = Z14S.index("z_Breaking share (short)")'),
    ("note P1-P4", 'n_.append("results, not delivery" + (" (a roster move by results)" if key == "P4" else ""))',
     'n_.append("mix, not pitch quality" + (" (a roster move by mix)" if key == "P4" else ""))'),
    ("groups", 'return "delivery" if base in DELIVERY else "results"', 'return "pitch quality" if base in DELIVERY else "mix"'),
    ("note P2", '    if grp.loc["delivery", "MLB"] <= 0:\n        n_.append("results, not delivery")',
     f'    if sp.loc["{ROLE}", "M2 - B0"] <= sp.loc["{ROLE}", "floor mean"]:\n        n_.append("role change")\n'
     '    if grp.loc["pitch quality", "MLB"] <= 0:\n        n_.append("mix, not pitch quality")'),
    ("note role P1-P4", '    if DEL[key]["auc"] <= DEL[key]["floor"].mean():\n',
     f'    if sub.loc["{ROLE}", "AUC"] <= sub.loc["{ROLE}", "floor mean"]:\n        n_.append("role change")\n'
     '    if DEL[key]["auc"] <= DEL[key]["floor"].mean():\n'),
    ("role subset", '"starters": df.role.eq(1.0).to_numpy(), "relievers": df.role.eq(0.0).to_numpy(),',
     f'"starters": df.role.eq(1.0).to_numpy(), "relievers": df.role.eq(0.0).to_numpy(),\n            "{ROLE}": df.prev_start.eq(df.role).to_numpy(),'),
    ("role subset P2", '    for sn, m_ in (("starters", test_.role.eq(1.0).to_numpy()), ("relievers", test_.role.eq(0.0).to_numpy()),',
     f'    for sn, m_ in (("starters", test_.role.eq(1.0).to_numpy()), ("relievers", test_.role.eq(0.0).to_numpy()),\n'
     f'                   ("{ROLE}", test_.prev_start.eq(test_.role).to_numpy()),'),
    ("look-ahead", "    assert any(not np.array_equal(s0[2][n][i0 + 1:], s0b[2][n][i0 + 1:], equal_nan=True) for n in s0[2])\n", LOOKAHEAD),
    ("family diagnostics",
     '          f"mean {np.mean(changes):.2f}, max {max(changes)}; rows without a primary fastball: {int(rows.no_primary.sum()):,}")\n',
     FAMDIAG),
    ("12-way bounds", '"lo_bonf": np.percentile(v, LO), "hi_bonf": np.percentile(v, HI)})',
     '"lo_bonf": np.percentile(v, LO), "hi_bonf": np.percentile(v, HI),\n'
     '                    "lo_12": np.percentile(v, 100 * 0.05 / 12 / 2), "hi_12": np.percentile(v, 100 - 100 * 0.05 / 12 / 2)})'),
    ("12-way return", '    return {"auc": r.loc["U", "AUC"], "lo": r.loc["U", "lo_bonf"], "hi": r.loc["U", "hi_bonf"], "floor": floor, "ok": ok,',
     '    return {"auc": r.loc["U", "AUC"], "lo": r.loc["U", "lo_bonf"], "hi": r.loc["U", "hi_bonf"], "floor": floor, "ok": ok,\n'
     '            "lo12": r.loc["U", "lo_12"],'),
    ("12-way summary", '    "floor_max": [P1["floor"].max(), P2["floor"].max(), P3["floor"].max(), P4["floor"].max()],',
     '    "floor_max": [P1["floor"].max(), P2["floor"].max(), P3["floor"].max(), P4["floor"].max()],\n'
     '    "lo_12way": [P1["lo12"], P2["main"].loc["M2 - B0", "lo_12"], P3["lo12"], P4["lo12"]],\n'
     '    "across three pitcher families (12-way)": ["SUPPORTED" if o and l > t else "NOT SUPPORTED" for o, l, t in (\n'
     '        (P1["ok"], P1["lo12"], 0.5), (P2["ok"], P2["main"].loc["M2 - B0", "lo_12"], 0.0), (P3["ok"], P3["lo12"], 0.5),\n'
     '        (P4["ok"], P4["lo12"], 0.5))],'),
    ("section text",
     "# Fastball parameters are separate series per fastball type (FF, SI, FC), each against its own normal; the z used on a day is the one of that day's primary fastball.",
     "# Secondary-pitch parameters are separate series per type of each family, each against its own normal; the z used on a day is the one of that family's primary type that day (most pitches before the day in the season). The two shares are all-pitch windows."),
    ("docstring", "    Returns days, per-type fastball series {(type, name): values}",
     "    Returns days, per-type series of each family {(type, name): values} with (\"choice\", family) primary-type indices"),
    ("quality-only labels",
     '"P1 delivery only", 1000, "U2_deld"),\n       "P3": unsupervised_test(arm_df, "P3 delivery only", 2000, "U2_deld"),\n       "P4": unsupervised_test(p4_df, "P4 delivery only"',
     '"R1 pitch quality only", 1000, "U2_deld"),\n       "P3": unsupervised_test(arm_df, "R2 pitch quality only", 2000, "U2_deld"),\n       "P4": unsupervised_test(p4_df, "R3 pitch quality only"'),
    ("roc labels", '(("P1 MLB", "#2471a3", P1), ("P3 MLB arm", "#c0392b", P3), ("P4 Triple-A", "#8a8a8a", P4))',
     '(("R1 MLB", "#2471a3", P1), ("R2 MLB arm", "#c0392b", P3), ("R3 Triple-A", "#8a8a8a", P4))'),
    ("roc title", 'ax.set_title(f"Pitchers: injured list within {H} days")', 'ax.set_title(f"Pitch mix and secondary pitches: injured list within {H} days")'),
    ("roc file", 'plt.savefig("pitcher_roc.png", dpi=120)', 'plt.savefig("mix_roc.png", dpi=120)'),
    ("floor labels", '["P1 MLB", "P3 MLB arm", "P4 Triple-A"]', '["R1 MLB", "R2 MLB arm", "R3 Triple-A"]'),
    ("floor dots", 'enumerate((("P1", P1), ("P3", P3), ("P4", P4)))', 'enumerate((("R1", P1), ("R2", P3), ("R3", P4)))'),
    ("floor file", 'plt.savefig("pitcher_floors.png", dpi=120)', 'plt.savefig("mix_floors.png", dpi=120)'),
    ("summary labels",
     '"test": ["P1 MLB U2 daily rank, 2024-2026", "P2 MLB M2 over B0, 2026", "P3 MLB arm, U2 daily rank", "P4 Triple-A U2 daily rank"]',
     '"test": ["R1 MLB U2 daily rank, 2024-2026", "R4 MLB M2 over B0, 2026", "R2 MLB arm, U2 daily rank", "R3 Triple-A U2 daily rank"]'),
    ("verdict file", 'summary.to_csv("pitcher_verdict.csv", index=False)', 'summary.to_csv("mix_verdict.csv", index=False)'),
    ("auc file", 'P2["main"].to_csv("pitcher_auc_mlb_2026.csv")', 'P2["main"].to_csv("mix_auc_mlb_2026.csv")'),
]
for name, old, new in REPL:
    assert src.count(old) == 1, f"replacement '{name}' matches {src.count(old)} times"
    src = src.replace(old, new)
print(f"{len(REPL)} replacements applied")
exec(compile(src, "frozen_pitcher_mix", "exec"))
