# %% [markdown]
# # Pitch shape x swing shape: diagnostic 0 (label-free)
#
# Before any matchup study is planned in detail, this notebook only counts what the data can support.
# It reads **no plate-appearance result column** (no `events`, `description`, `woba_value`,
# `delta_run_exp`, launch or expected-stat columns). It prints: the column list, how full the bat
# tracking and pitch-flight columns are by season and month, how many plate appearances, batters
# and pitchers there are, how much prior history a batter and a pitcher have at the time of each
# plate appearance, and a sanity check of the vertical approach angle (VAA) computed from the
# pitch-flight columns.

# %%
import glob
import os

import numpy as np
import pandas as pd

INPUT_ROOT = os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input")
FORBIDDEN = {"events", "description", "woba_value", "woba_denom", "delta_run_exp", "delta_home_win_exp",
             "launch_speed", "launch_angle", "estimated_woba_using_speedangle", "estimated_ba_using_speedangle",
             "estimated_slg_using_speedangle", "babip_value", "iso_value", "bb_type", "hit_location", "hc_x", "hc_y",
             "hit_distance_sc", "launch_speed_angle", "type", "post_bat_score", "post_fld_score",
             "post_away_score", "post_home_score", "des"}
TRACK = ["bat_speed", "swing_length", "attack_angle", "attack_direction", "swing_path_tilt",
         "intercept_ball_minus_batter_pos_x_inches", "intercept_ball_minus_batter_pos_y_inches"]
FLIGHT = ["release_speed", "release_pos_x", "release_pos_z", "release_extension", "release_spin_rate",
          "spin_axis", "pfx_x", "pfx_z", "vx0", "vy0", "vz0", "ax", "ay", "az", "arm_angle", "plate_x", "plate_z"]
KEYS = ["game_pk", "game_date", "at_bat_number", "pitch_number", "batter", "pitcher", "stand", "p_throws",
        "pitch_type", "is_swing"]


def find(name):
    p = sorted(glob.glob(os.path.join(INPUT_ROOT, "**", name), recursive=True))
    assert len(p) >= 1, f"missing {name}"
    return p[0]


frames = []
for y in (2024, 2025, 2026):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(find(f"statcast_{y}.parquet"))
    names = f.schema_arrow.names
    print(y, "rows", f.metadata.num_rows, "columns", len(names))
    print("  ", names)
    cols = [c for c in KEYS + TRACK + FLIGHT if c in names]
    print("  absent:", [c for c in KEYS + TRACK + FLIGHT if c not in names])
    assert not (set(cols) & FORBIDDEN)
    d = f.read(columns=cols).to_pandas()
    d["season"] = y
    frames.append(d)
d = pd.concat(frames, ignore_index=True)
del frames
d["game_date"] = pd.to_datetime(d["game_date"])
d["month"] = d["game_date"].dt.month
sw = d["is_swing"].astype("boolean").fillna(False).to_numpy(bool)
print("\nall pitches", len(d), "swings", int(sw.sum()))

# %% [markdown]
# ## How full are the bat-tracking columns (among swings) and the flight columns (all pitches)?

# %%
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
tr = [c for c in TRACK if c in d]
print(d.loc[sw].groupby(["season", "month"])[tr].apply(lambda g: g.notna().mean()).round(3))
fl = [c for c in FLIGHT if c in d]
print(d.groupby("season")[fl].apply(lambda g: g.notna().mean()).round(3).T)

# %% [markdown]
# ## Plate appearances, batters, pitchers, pairs

# %%
pa = d.drop_duplicates(["game_pk", "at_bat_number"])[["season", "game_pk", "game_date", "at_bat_number",
                                                       "batter", "pitcher", "stand", "p_throws"]]
print(pa.groupby("season").agg(pa=("game_pk", "size"), batters=("batter", "nunique"),
                               pitchers=("pitcher", "nunique"), games=("game_pk", "nunique")))
pairs = pa.groupby(["batter", "pitcher"]).size()
print("batter-pitcher pairs", len(pairs), "PA per pair quantiles", pairs.quantile([.5, .9, .99]).to_dict())
print("share of PA whose pair has >= 10 PA in all three seasons", float((pairs[pairs >= 10].sum()) / pairs.sum()))

# %% [markdown]
# ## History available at the time of each plate appearance (strictly earlier dates only)

# %%
day_b = d.loc[sw & d["bat_speed"].notna()].groupby(["batter", "game_date"]).size().rename("n").reset_index()
day_b = day_b.sort_values(["batter", "game_date"])
day_b["cum_before"] = day_b.groupby("batter")["n"].cumsum() - day_b["n"]
day_p = d.groupby(["pitcher", "game_date"]).size().rename("n").reset_index().sort_values(["pitcher", "game_date"])
day_p["cum_before"] = day_p.groupby("pitcher")["n"].cumsum() - day_p["n"]
x = pa.merge(day_b[["batter", "game_date", "cum_before"]].rename(columns={"cum_before": "b_swings_before"}),
             on=["batter", "game_date"], how="left")
x = x.merge(day_p[["pitcher", "game_date", "cum_before"]].rename(columns={"cum_before": "p_pitches_before"}),
            on=["pitcher", "game_date"], how="left")
# a batter with no tracked swing that day has no row; take his last cumulative count before the date
x = x.sort_values("game_date")
for who, tab, col in (("batter", day_b, "b_swings_before"), ("pitcher", day_p, "p_pitches_before")):
    tab = tab.sort_values("game_date").assign(cum_incl=lambda t: t["cum_before"] + t["n"])
    m = pd.merge_asof(x[["game_date", who]].reset_index(), tab[["game_date", who, "cum_incl"]],
                      on="game_date", by=who, allow_exact_matches=False, direction="backward").set_index("index")
    x[col] = m["cum_incl"].reindex(x.index).fillna(0)
for s in (2024, 2025, 2026):
    g = x[x["season"] == s]
    print(s, "PA", len(g),
          "| batter tracked swings before: >=100", round(float((g.b_swings_before >= 100).mean()), 3),
          ">=300", round(float((g.b_swings_before >= 300).mean()), 3),
          ">=1000", round(float((g.b_swings_before >= 1000).mean()), 3),
          "| pitcher pitches before: >=300", round(float((g.p_pitches_before >= 300).mean()), 3),
          ">=1000", round(float((g.p_pitches_before >= 1000).mean()), 3))

# %% [markdown]
# ## Vertical approach angle sanity check (label-free)

# %%
yf = 17.0 / 12.0
vyf = -np.sqrt(d["vy0"] ** 2 - 2 * d["ay"] * (50.0 - yf))
t = (vyf - d["vy0"]) / d["ay"]
vzf = d["vz0"] + d["az"] * t
d["vaa"] = -np.degrees(np.arctan(vzf / vyf))
print(d.groupby("pitch_type")["vaa"].agg(["size", "mean", "std"]).query("size > 5000").round(2))
print("VAA vs plate_z correlation (fastballs):",
      round(float(d.loc[d.pitch_type.isin(["FF"]), ["vaa", "plate_z"]].corr().iloc[0, 1]), 3))
if "attack_angle" in d:
    s2 = sw & d["attack_angle"].notna() & d["vaa"].notna()
    print("attack angle on swings: mean", round(float(d.loc[s2, "attack_angle"].mean()), 2),
          "sd", round(float(d.loc[s2, "attack_angle"].std()), 2),
          "| corr(attack_angle, vaa) on swings", round(float(d.loc[s2, ["attack_angle", "vaa"]].corr().iloc[0, 1]), 3),
          "| corr(attack_angle, plate_z)", round(float(d.loc[s2, ["attack_angle", "plate_z"]].corr().iloc[0, 1]), 3))
    # within-batter spread vs between-batter spread of the swing descriptors (how stable a "swing shape" is)
    for c in [c for c in ("bat_speed", "swing_length", "attack_angle", "swing_path_tilt", "attack_direction") if c in d]:
        g = d.loc[sw & d[c].notna()].groupby(["season", "batter"])[c]
        n = g.size()
        keep = n[n >= 200].index
        m = g.mean().loc[keep]
        w = g.std().loc[keep]
        print(f"  {c:18s} batters(>=200 swings/season) {len(keep)}  between-batter sd of means {m.std():.2f}"
              f"  typical within-batter sd {w.median():.2f}")
print("done")
