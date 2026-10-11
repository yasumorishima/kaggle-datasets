# %% [markdown]
# # Does a Pitcher's In-Outing Fade and Scatter Drift Before the Injured List?
#
# A [pre-registered pitcher study](https://www.kaggle.com/code/yasunorim/do-pitchers-drift-before-the-il) found that the drift of 14 window **levels** (velocity, spin, extension, release point, arm angle, movement, results) from a pitcher's own normal does not come before the injured list. A level can stay normal while the delivery **inside an outing** changes:
#
# - **Velocity fade**: how fast fastball velocity falls as the outing goes on, and the scatter of velocity around that line.
# - **Repeatability**: the pitch-to-pitch scatter of the release point, the arm angle and the spin axis within an outing.
#
# This notebook runs the frozen pitcher notebook (commit `2c6b01d`, checked by md5) with **only its parameters replaced** by these five; rows, labels, the rival model, floors and tests are unchanged, and equality gates check that. Everything was fixed before any new parameter was compared with a placement: [FREEZE_pitch_inouting.md](https://github.com/yasumorishima/kaggle-datasets/blob/main/ohtani-2026-early-warning/FREEZE_pitch_inouting.md).
#
# Names in the printed output are the frozen code's: **P1 = Q1** (MLB, all placements), **P3 = Q2** (MLB, arm placements), **P4 = Q3** (Triple-A), **P2 = Q4** (MLB 2026 model over the rival). "Delivery" in the frozen printouts is now the **mechanics** group (release point, arm angle and spin axis scatter).

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

INOUTING = '''
def inouting_series(g, pt, days):
    """Five in-outing parameters per fastball type and window (FREEZE_pitch_inouting.md). g is sorted by date, game,
    at-bat, pitch; an appearance is (game_pk, game_date); order counts all of the pitcher's pitches in it."""
    app = pd.factorize(pd.MultiIndex.from_arrays([g.game_pk.to_numpy(), g.game_date.to_numpy()]))[0]
    order = pd.Series(np.ones(len(g))).groupby(app).cumsum().to_numpy() - 1.0
    out = {}
    for t in FB:
        m = pt == t
        s_ = g[m]
        sd = s_.game_date.to_numpy()
        a, o = app[m], order[m]
        spd = s_.release_speed.to_numpy(float)
        px, pz = s_.release_pos_x.to_numpy(float), s_.release_pos_z.to_numpy(float)
        arm, sax = s_.arm_angle.to_numpy(float), s_.spin_axis.to_numpy(float)
        n_ = len(s_)
        dxdy, dx2, r2, rp2, ar2, sa2 = (np.full(n_, np.nan) for _ in range(6))
        for ix in pd.Series(np.arange(n_)).groupby(a).indices.values():
            ii = ix[np.isfinite(spd[ix])]
            k = len(ii)
            if k >= IN_MIN:
                dx, dy = o[ii] - o[ii].mean(), spd[ii] - spd[ii].mean()
                dxdy[ii], dx2[ii] = dx * dy, dx ** 2
                b = (dx * dy).sum() / (dx ** 2).sum()
                r2[ii] = (dy - b * dx) ** 2 * k / (k - 2)
            ii = ix[np.isfinite(px[ix]) & np.isfinite(pz[ix])]
            k = len(ii)
            if k >= IN_MIN:
                rp2[ii] = ((px[ii] - px[ii].mean()) ** 2 + (pz[ii] - pz[ii].mean()) ** 2) * k / (k - 1)
            ii = ix[np.isfinite(arm[ix])]
            k = len(ii)
            if k >= IN_MIN:
                ar2[ii] = (arm[ii] - arm[ii].mean()) ** 2 * k / (k - 1)
            ii = ix[np.isfinite(sax[ix])]
            k = len(ii)
            if k >= IN_MIN:
                rad = np.deg2rad(sax[ii])
                cm = np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()))
                sa2[ii] = ((sax[ii] - cm + 180.0) % 360.0 - 180.0) ** 2 * k / (k - 1)
        for sfx, w in zip(SFX, (LONG[0], SHORT[0])):
            out[(t, "FB velocity fade" + sfx)] = 25.0 * wmean(sd, dxdy, w, days) / wmean(sd, dx2, w, days)
            out[(t, "FB velocity scatter" + sfx)] = np.sqrt(wmean(sd, r2, w, days))
            out[(t, "Release point scatter" + sfx)] = np.sqrt(wmean(sd, rp2, w, days))
            out[(t, "Arm angle scatter" + sfx)] = np.sqrt(wmean(sd, ar2, w, days))
            out[(t, "Spin axis scatter" + sfx)] = np.sqrt(wmean(sd, sa2, w, days))
    return out


def params_for(g, prev_type):'''

GATES = '''
if not FAKE:  # equality gates: rows, labels and the rival do not depend on the parameters (FREEZE_pitch_inouting.md)
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

DIRECTIONAL = '''
# %% [markdown]
# ### Reported, not tested: directional fade (a steeper fade than normal), overall and by role

# %%
for _r in (rows_m, rows_a):
    _r["fade_steeper"] = -_r["z_FB velocity fade (short)"]
    _r["fade_steeperd"] = daily_rank(_r, "fade_steeper")
_p1 = rows_m[rows_m.use14].reset_index(drop=True)
for _key, _d, _sd in (("Q1 MLB", _p1, 1000), ("Q2 MLB arm", _p1[~_p1.il14 | _p1.next_arm].reset_index(drop=True), 2000),
                      ("Q3 Triple-A", rows_a[rows_a.use14].reset_index(drop=True), 3000)):
    _res = unsupervised_test(_d, f"{_key} directional fade (reported)", _sd, "fade_steeperd")
    print(by_subsets(_res, {"starters": _res["df"].role.eq(1.0).to_numpy(),
                            "relievers": _res["df"].role.eq(0.0).to_numpy()}).round(4).to_string(index=False))
'''

HEADER_END = src.index("# %%\nimport glob")
REPL = [
    ("header", src[:HEADER_END], "# %% [markdown]\n# Frozen pitcher notebook (commit 2c6b01d) with the in-outing parameters (see above).\n\n"),
    ("smoke MIN_Z", "BASE_LAG, MIN_BASE, MIN_SCALE, MIN_Z, MIN_D = (0, 2, 3, 4, 2) if SMOKE else (30, 10, 20, 7, 5)",
     "BASE_LAG, MIN_BASE, MIN_SCALE, MIN_Z, MIN_D = (0, 2, 3, 2, 2) if SMOKE else (30, 10, 20, 3, 5)\nIN_MIN = 4"),
    ("PCOLS", '"zone", "description", "events", "type", "launch_speed", "woba_denom", "inning", "game_type"]\nassert not',
     '"zone", "description", "events", "type", "launch_speed", "woba_denom", "inning", "game_type", "spin_axis"]\nassert not'),
    ("fill table", '"arm_angle", "pfx_x", "pfx_z", "pitch_type", "zone"]].apply',
     '"arm_angle", "pfx_x", "pfx_z", "pitch_type", "zone", "spin_axis"]].apply'),
    ("section title", "# ## Fourteen parameters per appearance (only earlier pitches, same season and level)",
     "# ## Five in-outing parameters per appearance (only earlier pitches, same season and level)"),
    ("parameter lists",
     src[src.index('FBP = ["FB velocity", "FB top velocity"'):src.index("SFX = (\"\", \" (short)\")")],
     'FBP = ["FB velocity fade", "FB velocity scatter", "Release point scatter", "Arm angle scatter", "Spin axis scatter"]\n'
     'FBCOL = {"FB velocity fade": "release_speed", "FB velocity scatter": "release_speed",\n'
     '         "Release point scatter": "release_pos_x", "Arm angle scatter": "arm_angle", "Spin axis scatter": "spin_axis"}\n'
     'FBSRC = {**{n: [c] for n, c in FBCOL.items()}, "Release point scatter": ["release_pos_x", "release_pos_z"]}\n'
     'OTHER = []\n'
     'OTHERCOL = {"Off-speed velocity": "release_speed", "Zone rate": "zone", "Whiff": "description", "K - BB": "events",\n'
     '            "Hard-hit allowed": "launch_speed"}\n'
     'BASE14 = FBP + OTHER\n'
     'DELIVERY = ["Release point scatter", "Arm angle scatter", "Spin axis scatter"]\n'),
    ("fill columns", 'fr = d.groupby("season")[sorted(set(FBCOL.values()) | {"zone", "pitch_type", "description"})]',
     'fr = d.groupby("season")[sorted(set(FBCOL.values()) | {"release_pos_z", "zone", "pitch_type", "description"})]'),
    ("fill rule", "    DROPPED[name] = sorted(n for n in BASE14 if fr[src[n]] < 0.90)",
     "    DROPPED[name] = sorted(n for n in BASE14 if min(fr[c] for c in FBSRC[n]) < 0.90)"),
    ("new function", "\ndef params_for(g, prev_type):", INOUTING),
    ("parameter loop",
     src[src.index("    fb = {}\n    for t in FB:\n"):src.index("    per_day = pd.Series(1, index=gd)")],
     "    fb = inouting_series(g, pt, days)\n    oth = {}\n"),
    ("mechanics U2 need", 'df["U2_del"] = u_scores(df[ZDEL].to_numpy(), 5)[1]', 'df["U2_del"] = u_scores(df[ZDEL].to_numpy(), 2)[1]'),
    ("look-ahead columns", '"pfx_x", "pfx_z", "launch_speed"]:\n        g0.loc[late, c] = 999.0',
     '"pfx_x", "pfx_z", "launch_speed", "spin_axis"]:\n        g0.loc[late, c] = 999.0'),
    ("look-ahead assert", "    assert any(not np.array_equal(s0[2][n][i0 + 1:], s0b[2][n][i0 + 1:], equal_nan=True) for n in s0[2])",
     "    assert any(not np.array_equal(s0[1][n][i0 + 1:], s0b[1][n][i0 + 1:], equal_nan=True) for n in s0[1])"),
    ("centre kind", '            if n == "Zone rate (short)":', '            if n == "FB velocity scatter (short)":'),
    ("gates", "\np1_df = rows_m[rows_m.use14].reset_index(drop=True)\n", GATES),
    ("planted column", 'c0 = Z14S.index("z_FB velocity (short)")', 'c0 = Z14S.index("z_FB velocity scatter (short)")'),
    ("note P1-P4", 'n_.append("results, not delivery" + (" (a roster move by results)" if key == "P4" else ""))',
     'n_.append("velocity, not mechanics" + (" (a roster move by velocity)" if key == "P4" else ""))'),
    ("groups", 'return "delivery" if base in DELIVERY else "results"', 'return "mechanics" if base in DELIVERY else "velocity"'),
    ("note P2", '    if grp.loc["delivery", "MLB"] <= 0:\n        n_.append("results, not delivery")',
     '    if grp.loc["mechanics", "MLB"] <= 0:\n        n_.append("velocity, not mechanics")'),
    ("directional", "\nprint(summary.to_string(index=False))\n", "\nprint(summary.to_string(index=False))\n" + DIRECTIONAL),
    ("roc labels", '(("P1 MLB", "#2471a3", P1), ("P3 MLB arm", "#c0392b", P3), ("P4 Triple-A", "#8a8a8a", P4))',
     '(("Q1 MLB", "#2471a3", P1), ("Q2 MLB arm", "#c0392b", P3), ("Q3 Triple-A", "#8a8a8a", P4))'),
    ("roc title", 'ax.set_title(f"Pitchers: injured list within {H} days")', 'ax.set_title(f"In-outing fade and scatter: injured list within {H} days")'),
    ("roc file", 'plt.savefig("pitcher_roc.png", dpi=120)', 'plt.savefig("inouting_roc.png", dpi=120)'),
    ("floor labels", '["P1 MLB", "P3 MLB arm", "P4 Triple-A"]', '["Q1 MLB", "Q2 MLB arm", "Q3 Triple-A"]'),
    ("floor dots", 'enumerate((("P1", P1), ("P3", P3), ("P4", P4)))', 'enumerate((("Q1", P1), ("Q2", P3), ("Q3", P4)))'),
    ("floor file", 'plt.savefig("pitcher_floors.png", dpi=120)', 'plt.savefig("inouting_floors.png", dpi=120)'),
    ("summary labels",
     '"test": ["P1 MLB U2 daily rank, 2024-2026", "P2 MLB M2 over B0, 2026", "P3 MLB arm, U2 daily rank", "P4 Triple-A U2 daily rank"]',
     '"test": ["Q1 MLB U2 daily rank, 2024-2026", "Q4 MLB M2 over B0, 2026", "Q2 MLB arm, U2 daily rank", "Q3 Triple-A U2 daily rank"]'),
    ("verdict file", 'summary.to_csv("pitcher_verdict.csv", index=False)', 'summary.to_csv("inouting_verdict.csv", index=False)'),
    ("auc file", 'P2["main"].to_csv("pitcher_auc_mlb_2026.csv")', 'P2["main"].to_csv("inouting_auc_mlb_2026.csv")'),
]
for name, old, new in REPL:
    assert src.count(old) == 1, f"replacement '{name}' matches {src.count(old)} times"
    src = src.replace(old, new)
print(f"{len(REPL)} replacements applied")
exec(compile(src, "frozen_pitcher_inouting", "exec"))
