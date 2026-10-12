# %% [markdown]
# # Does a Pitcher's Command Drift Before the Injured List?
#
# Three [pre-registered](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch.md) [pitcher](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_inouting.md) [studies](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_mix.md) found that neither the window levels of the fastball and delivery, nor the fade and scatter inside an outing, nor the pitch mix and secondary-pitch quality drift from a pitcher's own normal before the injured list. Where the pitches **go** entered only as one zone rate. This notebook measures command:
#
# - **Location scatter**: how far a pitcher's fastballs miss, sideways and in height, around where his fastballs of that type went to that side of the plate in that outing, and the same for his other pitches.
# - **Zone outcomes**: the share of pitches over the heart of the plate, the share of pitches far outside the zone, and the share of plate appearances that reach a three-ball count.
#
# It runs the frozen pitcher notebook (commit `2c6b01d`, checked by md5) with **only its parameters replaced** by these six; rows, labels, the rival model, floors and tests are unchanged, and equality gates check that. Triple-A pitch locations come from a [separate fetch](https://www.kaggle.com/code/yasunorim/triple-a-statcast-pitch-location-2023-2026-fetch) joined to the frozen pitching files by pitch key. Everything was fixed before any new parameter was compared with a placement: [FREEZE_pitch_command.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_command.md).
#
# Names in the printed output are the frozen code's: **P1 = C1** (MLB, all placements), **P3 = C2** (MLB, arm placements), **P4 = C3** (Triple-A), **P2 = C4** (MLB 2026 model over the rival). "Delivery" in the frozen printouts is now the **scatter** group (the three location-scatter parameters); "results" is the **zone outcomes** group.
#
# This is the fourth pre-registered family of tests on the same pitcher placements, so the summary also shows each verdict at a 16-way Bonferroni reading across the four families (0.156% and 99.84%); the 4-way verdict is "supported within this family", not across the four.

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

LOCATION = '''LOC = ["stand", "plate_x", "plate_z", "sz_top", "sz_bot", "balls"]
LOC_MD5 = {2023: "ddc88cbdb25d87cd356890b5934a1f69", 2024: "0eb84f0af52298597404accf72a318a5",
           2025: "9652d14a943e76d3d772437554110dc4", 2026: "921600911144b7e89c2fe58d8f87aef3"}
LOC_KEYS = ["game_pk", "at_bat_number", "pitch_number"]


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
    d.loc[~same, ["plate_x", "plate_z", "sz_top", "sz_bot", "balls"]] = np.nan
    ok = (d.zone.notna() & d.plate_x.notna() & d.plate_z.notna() & d.sz_top.notna() & d.sz_bot.notna()).to_numpy()
    box = ((d.plate_x.abs() <= 0.83) & (d.plate_z >= d.sz_bot) & (d.plate_z <= d.sz_top)).to_numpy()
    agree = (d.zone.between(1, 9).to_numpy() == box)[ok].mean()
    print(f"  Triple-A {y}: frozen zone agrees with the joined plate location on {agree:.4%} of {ok.sum():,} pitches")
    assert agree >= 0.93, "joined locations do not match the frozen zone (pitches shifted inside a plate appearance?)"
    return d.drop(columns=["pitcher_loc", "stand_loc", "date_loc"])


def load(pattern, years, md5s=None):'''

COMMAND = '''
def group_dev2(keys, vals):
    """Squared distance of each pitch from the mean of its group, times k / (k - 1) (vals: one column per coordinate);
    NaN for a pitch with a missing coordinate or in a group with fewer than IN_MIN such pitches."""
    out = np.full(len(keys), np.nan)
    pos = np.flatnonzero(np.isfinite(vals).all(1))
    for ii in pd.Series(pos).groupby(keys[pos]).indices.values():
        ix = pos[ii]
        k = len(ix)
        if k >= IN_MIN:
            dv = vals[ix] - vals[ix].mean(0)
            out[ix] = (dv ** 2).sum(1) * k / (k - 1)
    return out


def group_key(*cols):
    """Integer group id per pitch (empty for no pitches: pandas 2 cannot factorize an empty MultiIndex)."""
    if len(cols[0]) == 0:
        return np.zeros(0, dtype=np.int64)
    return pd.factorize(pd.MultiIndex.from_arrays(list(cols)))[0]


def command_series(g, pt, days):
    """Six command parameters (FREEZE_pitch_command.md). g is sorted by date, game, at-bat, pitch; a group is the
    pitches of one appearance (game_pk, game_date) to one batter side, and for other pitches also of one type."""
    gd = g.game_date.to_numpy()
    gp = g.game_pk.to_numpy()
    st = g.stand.fillna("").to_numpy(dtype=object)
    px, pz = g.plate_x.to_numpy(float), g.plate_z.to_numpy(float)
    sx, sz = np.where(st == "", np.nan, px), np.where(st == "", np.nan, pz)
    fb = {}
    for t in FB:
        m = pt == t
        key = group_key(gp[m], gd[m], st[m])
        hx, hz = group_dev2(key, sx[m][:, None]), group_dev2(key, sz[m][:, None])
        for sfx, w in zip(SFX, (LONG[0], SHORT[0])):
            fb[(t, "FB horizontal location scatter" + sfx)] = np.sqrt(wmean(gd[m], hx, w, days))
            fb[(t, "FB vertical location scatter" + sfx)] = np.sqrt(wmean(gd[m], hz, w, days))
    oth = {}
    sec = (pt != "") & ~np.isin(pt, FB + NOT_OFFSPEED)
    key = group_key(gp[sec], gd[sec], st[sec], pt[sec])
    s2 = group_dev2(key, np.c_[sx[sec], sz[sec]])
    heart = np.where(g.zone.isna(), np.nan, g.zone.eq(5).astype(float))
    top, bot = g.sz_top.to_numpy(float), g.sz_bot.to_numpy(float)
    far = (np.abs(px) > 0.83 + 1.0) | (pz > top + 1.0) | (pz < bot - 1.0)
    waste = np.where(np.isfinite(px) & np.isfinite(pz) & np.isfinite(top) & np.isfinite(bot), far.astype(float), np.nan)
    ev = g.events.fillna("").astype(str)
    is_pa = ((ev != "") & ~ev.str.startswith(NOT_PA) & (ev != "intent_walk")).to_numpy()
    pa_max = g.groupby(["game_pk", "at_bat_number"]).balls.transform("max").to_numpy(float)
    three = np.where(np.isfinite(pa_max), (pa_max >= 3).astype(float), np.nan)[is_pa]
    for sfx, (w_fb, w_nfb, w_all, w_sw, w_bf, w_bip) in zip(SFX, (LONG, SHORT)):
        oth["Secondary location scatter" + sfx] = np.sqrt(wmean(gd[sec], s2, w_nfb, days))
        oth["Heart share" + sfx] = wmean(gd, heart, w_all, days)
        oth["Waste share" + sfx] = wmean(gd, waste, w_all, days)
        oth["Three-ball share" + sfx] = wmean(gd[is_pa], three, w_bf, days)
    return fb, oth


def params_for(g, prev_type):'''

PARAMS = '''FBP = ["FB horizontal location scatter", "FB vertical location scatter"]
FBCOL = {"FB horizontal location scatter": "plate_x", "FB vertical location scatter": "plate_z"}
OTHER = ["Secondary location scatter", "Heart share", "Waste share", "Three-ball share"]
OTHERCOL = {"Secondary location scatter": "plate_x", "Heart share": "zone", "Waste share": "plate_x",
            "Three-ball share": "balls"}
FBSRC = {"FB horizontal location scatter": ["plate_x", "stand"], "FB vertical location scatter": ["plate_z", "stand"],
         "Secondary location scatter": ["plate_x", "plate_z", "stand"], "Heart share": ["zone"],
         "Waste share": ["plate_x", "plate_z", "sz_top", "sz_bot"], "Three-ball share": ["balls"]}
BASE14 = FBP + OTHER
DELIVERY = ["FB horizontal location scatter", "FB vertical location scatter", "Secondary location scatter"]
'''

GATES = '''
if not FAKE:  # equality gates: rows, labels and the rival do not depend on the parameters (FREEZE_pitch_command.md)
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
'''

ROLE = "same role as the previous appearance"

HEADER_END = src.index("# %%\nimport glob")
REPL = [
    ("header", src[:HEADER_END], "# %% [markdown]\n# Frozen pitcher notebook (commit 2c6b01d) with the command parameters (see above).\n\n"),
    ("smoke MIN_Z", "BASE_LAG, MIN_BASE, MIN_SCALE, MIN_Z, MIN_D = (0, 2, 3, 4, 2) if SMOKE else (30, 10, 20, 7, 5)",
     "BASE_LAG, MIN_BASE, MIN_SCALE, MIN_Z, MIN_D = (0, 2, 3, 2, 2) if SMOKE else (30, 10, 20, 3, 5)\nIN_MIN = 3"),
    ("location join", "def load(pattern, years, md5s=None):", LOCATION),
    ("read columns", "        d = pd.read_parquet(p[0], columns=PCOLS)\n",
     '        d = pd.read_parquet(p[0], columns=PCOLS + (LOC if pattern.startswith("statcast") else ["stand"]))\n'
     '        if pattern.startswith("aaa"):\n'
     '            d = join_location(d, y)\n'),
    ("fill table", '"arm_angle", "pfx_x", "pfx_z", "pitch_type", "zone"]].apply',
     '"arm_angle", "pfx_x", "pfx_z", "pitch_type", "zone"] + LOC].apply'),
    ("section title", "# ## Fourteen parameters per appearance (only earlier pitches, same season and level)",
     "# ## Six command parameters per appearance (only earlier pitches, same season and level)"),
    ("parameter lists", src[src.index('FBP = ["FB velocity", "FB top velocity"'):src.index("SFX = (\"\", \" (short)\")")], PARAMS),
    ("fill columns", 'fr = d.groupby("season")[sorted(set(FBCOL.values()) | {"zone", "pitch_type", "description"})]',
     'fr = d.groupby("season")[sorted({c for v in FBSRC.values() for c in v} | {"zone", "pitch_type", "description"})]'),
    ("fill rule", "    DROPPED[name] = sorted(n for n in BASE14 if fr[src[n]] < 0.90)",
     "    DROPPED[name] = sorted(n for n in BASE14 if min(fr[c] for c in FBSRC[n]) < 0.90)"),
    ("new function", "\ndef params_for(g, prev_type):", COMMAND),
    ("parameter loop", src[src.index("    fb = {}\n    for t in FB:\n"):src.index("    per_day = pd.Series(1, index=gd)")],
     "    fb, oth = command_series(g, pt, days)\n"),
    ("scatter U2 need", 'df["U2_del"] = u_scores(df[ZDEL].to_numpy(), 5)[1]', 'df["U2_del"] = u_scores(df[ZDEL].to_numpy(), 2)[1]'),
    ("look-ahead columns", '"pfx_x", "pfx_z", "launch_speed"]:\n        g0.loc[late, c] = 999.0',
     '"pfx_x", "pfx_z", "launch_speed", "plate_x", "plate_z", "sz_top", "sz_bot"]:\n        g0.loc[late, c] = 999.0\n    g0.loc[late, "balls"] = 3'),
    ("look-ahead assert", "    assert any(not np.array_equal(s0[2][n][i0 + 1:], s0b[2][n][i0 + 1:], equal_nan=True) for n in s0[2])\n",
     "    assert any(not np.array_equal(s0[2][n][i0 + 1:], s0b[2][n][i0 + 1:], equal_nan=True) for n in s0[2])\n"
     "    assert any(not np.array_equal(s0[1][n][i0 + 1:], s0b[1][n][i0 + 1:], equal_nan=True) for n in s0[1])\n"),
    ("centre kind", '            if n == "Zone rate (short)":', '            if n == "Heart share (short)":'),
    ("diagnostics",
     '          f"mean {np.mean(changes):.2f}, max {max(changes)}; rows without a primary fastball: {int(rows.no_primary.sum()):,}")\n',
     DIAG),
    ("gates", "\np1_df = rows_m[rows_m.use14].reset_index(drop=True)\n", GATES),
    ("planted column", 'c0 = Z14S.index("z_FB velocity (short)")', 'c0 = Z14S.index("z_FB horizontal location scatter (short)")'),
    ("note P1-P4", 'n_.append("results, not delivery" + (" (a roster move by results)" if key == "P4" else ""))',
     'n_.append("zone outcomes, not scatter" + (" (a roster move by zone outcomes)" if key == "P4" else ""))'),
    ("groups", 'return "delivery" if base in DELIVERY else "results"', 'return "scatter" if base in DELIVERY else "zone outcomes"'),
    ("note P2", '    if grp.loc["delivery", "MLB"] <= 0:\n        n_.append("results, not delivery")',
     f'    if sp.loc["{ROLE}", "M2 - B0"] <= sp.loc["{ROLE}", "floor mean"]:\n        n_.append("role change")\n'
     '    if grp.loc["scatter", "MLB"] <= 0:\n        n_.append("zone outcomes, not scatter")'),
    ("note role P1-P4", '    if DEL[key]["auc"] <= DEL[key]["floor"].mean():\n',
     f'    if sub.loc["{ROLE}", "AUC"] <= sub.loc["{ROLE}", "floor mean"]:\n        n_.append("role change")\n'
     '    if DEL[key]["auc"] <= DEL[key]["floor"].mean():\n'),
    ("role subset", '"starters": df.role.eq(1.0).to_numpy(), "relievers": df.role.eq(0.0).to_numpy(),',
     f'"starters": df.role.eq(1.0).to_numpy(), "relievers": df.role.eq(0.0).to_numpy(),\n            "{ROLE}": df.prev_start.eq(df.role).to_numpy(),'),
    ("role subset P2", '    for sn, m_ in (("starters", test_.role.eq(1.0).to_numpy()), ("relievers", test_.role.eq(0.0).to_numpy()),',
     f'    for sn, m_ in (("starters", test_.role.eq(1.0).to_numpy()), ("relievers", test_.role.eq(0.0).to_numpy()),\n'
     f'                   ("{ROLE}", test_.prev_start.eq(test_.role).to_numpy()),'),
    ("16-way bounds", '"lo_bonf": np.percentile(v, LO), "hi_bonf": np.percentile(v, HI)})',
     '"lo_bonf": np.percentile(v, LO), "hi_bonf": np.percentile(v, HI),\n'
     '                    "lo_16": np.percentile(v, 100 * 0.05 / 16 / 2), "hi_16": np.percentile(v, 100 - 100 * 0.05 / 16 / 2)})'),
    ("16-way return", '    return {"auc": r.loc["U", "AUC"], "lo": r.loc["U", "lo_bonf"], "hi": r.loc["U", "hi_bonf"], "floor": floor, "ok": ok,',
     '    return {"auc": r.loc["U", "AUC"], "lo": r.loc["U", "lo_bonf"], "hi": r.loc["U", "hi_bonf"], "floor": floor, "ok": ok,\n'
     '            "lo16": r.loc["U", "lo_16"],'),
    ("16-way summary", '    "floor_max": [P1["floor"].max(), P2["floor"].max(), P3["floor"].max(), P4["floor"].max()],',
     '    "floor_max": [P1["floor"].max(), P2["floor"].max(), P3["floor"].max(), P4["floor"].max()],\n'
     '    "lo_16way": [P1["lo16"], P2["main"].loc["M2 - B0", "lo_16"], P3["lo16"], P4["lo16"]],\n'
     '    "across four pitcher families (16-way)": ["SUPPORTED" if o and l > t else "NOT SUPPORTED" for o, l, t in (\n'
     '        (P1["ok"], P1["lo16"], 0.5), (P2["ok"], P2["main"].loc["M2 - B0", "lo_16"], 0.0), (P3["ok"], P3["lo16"], 0.5),\n'
     '        (P4["ok"], P4["lo16"], 0.5))],'),
    ("section text",
     "# Fastball parameters are separate series per fastball type (FF, SI, FC), each against its own normal; the z used on a day is the one of that day's primary fastball.",
     "# Fastball location scatter is a separate series per fastball type (FF, SI, FC), each against its own normal; the z used on a day is the one of that day's primary fastball. The other four are windows over all secondary pitches, all pitches or all plate appearances."),
    ("docstring", "    Returns days, per-type fastball series {(type, name): values}",
     "    Returns days, per-type fastball location-scatter series {(type, name): values}"),
    ("scatter-only labels",
     '"P1 delivery only", 1000, "U2_deld"),\n       "P3": unsupervised_test(arm_df, "P3 delivery only", 2000, "U2_deld"),\n       "P4": unsupervised_test(p4_df, "P4 delivery only"',
     '"C1 scatter only", 1000, "U2_deld"),\n       "P3": unsupervised_test(arm_df, "C2 scatter only", 2000, "U2_deld"),\n       "P4": unsupervised_test(p4_df, "C3 scatter only"'),
    ("roc labels", '(("P1 MLB", "#2471a3", P1), ("P3 MLB arm", "#c0392b", P3), ("P4 Triple-A", "#8a8a8a", P4))',
     '(("C1 MLB", "#2471a3", P1), ("C2 MLB arm", "#c0392b", P3), ("C3 Triple-A", "#8a8a8a", P4))'),
    ("roc title", 'ax.set_title(f"Pitchers: injured list within {H} days")', 'ax.set_title(f"Command: injured list within {H} days")'),
    ("roc file", 'plt.savefig("pitcher_roc.png", dpi=120)', 'plt.savefig("command_roc.png", dpi=120)'),
    ("floor labels", '["P1 MLB", "P3 MLB arm", "P4 Triple-A"]', '["C1 MLB", "C2 MLB arm", "C3 Triple-A"]'),
    ("floor dots", 'enumerate((("P1", P1), ("P3", P3), ("P4", P4)))', 'enumerate((("C1", P1), ("C2", P3), ("C3", P4)))'),
    ("floor file", 'plt.savefig("pitcher_floors.png", dpi=120)', 'plt.savefig("command_floors.png", dpi=120)'),
    ("summary labels",
     '"test": ["P1 MLB U2 daily rank, 2024-2026", "P2 MLB M2 over B0, 2026", "P3 MLB arm, U2 daily rank", "P4 Triple-A U2 daily rank"]',
     '"test": ["C1 MLB U2 daily rank, 2024-2026", "C4 MLB M2 over B0, 2026", "C2 MLB arm, U2 daily rank", "C3 Triple-A U2 daily rank"]'),
    ("verdict file", 'summary.to_csv("pitcher_verdict.csv", index=False)', 'summary.to_csv("command_verdict.csv", index=False)'),
    ("auc file", 'P2["main"].to_csv("pitcher_auc_mlb_2026.csv")', 'P2["main"].to_csv("command_auc_mlb_2026.csv")'),
]
for name, old, new in REPL:
    assert src.count(old) == 1, f"replacement '{name}' matches {src.count(old)} times"
    src = src.replace(old, new)
print(f"{len(REPL)} replacements applied")
exec(compile(src, "frozen_pitcher_command", "exec"))
