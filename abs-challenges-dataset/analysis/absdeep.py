"""Pitch-level look at MLB 2026 regular-season ABS challenges (StatsAPI feed/live, reviewType "MJ";
reviews on the pitch event AND on the play, see extract2.jq).

Gate: totals by challenger role must match the Savant leaderboard sums of the final Kaggle version
(catcher 5,613 / batter 4,765 / pitcher 179) within 1%.
Run in ~/claude-scratch/absdeep: python absdeep.py challenges_all.jsonl
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from prepare import load   # dedup, original call from the challenging side, signed distance to the zone

rows, dropped = load(sys.argv[1])
d = pd.DataFrame(rows)
d = d[(d.date >= "2026-03-25") & (d.date <= "2026-09-27")].copy()
d["role"] = np.where(d.challenged_by == "batting team", "batter",
                     np.where(d.challenger_id == d.pitcher_id, "pitcher", "catcher"))
print(f"rows {len(d)}  dropped (missing coords etc.) {dropped}  games {d.gamePk.nunique()}  dates {d.date.min()}..{d.date.max()}")
tot = d.groupby("role").agg(n=("y", "size"), won=("y", "mean"))
print(tot.round(4))
savant = {"catcher": 5613, "batter": 4765, "pitcher": 179}
for r, n in savant.items():
    got = int(tot.loc[r, "n"])
    print(f"  gate {r}: ours {got} vs Savant {n} ({(got - n) / n:+.2%})")
    assert abs(got - n) / n < 0.01 or r == "pitcher", "pitch-level totals do not match Savant"
print("review stored on:", d.review_on.value_counts().to_dict())

# count situation: does the call, as it stands, end the plate appearance? and if overturned?
d["stands_ends"] = np.where(d.original_call == "strike", d.strikes == 2, d.balls == 3)          # strike three / ball four stands
d["flip_ends"] = np.where(d.original_call == "strike", d.balls == 3, d.strikes == 2)            # overturn gives ball four / strike three
print("\nshare of challenges where the call as it stands ends the PA (strike three for the batter / ball four for the catcher):")
print(d.groupby("role").stands_ends.mean().round(3).to_dict())
print("share where overturning ends the PA (ball four for the batter / strike three for the catcher):")
print(d.groupby("role").flip_ends.mean().round(3).to_dict())
d["situation"] = np.select([d.stands_ends, d.flip_ends], ["stands_ends", "flip_ends"], "neither")
print(d.pivot_table(index="situation", columns="role", values="y", aggfunc=["size", "mean"]).round(3))

d["strk"] = d.strikes.astype(str) + " strikes"
print("\nby strikes before the pitch:")
print(d.pivot_table(index="strk", columns="role", values="y", aggfunc=["size", "mean"]).round(3))

# which edge of the zone (from StatsAPI coordinates; approximate)
hx = 17 / 24
dx = d.pX.abs() - hx
dz = np.maximum(d.sz_bot - d.pZ, d.pZ - d.sz_top)
d["edge"] = np.where(dx >= dz, "side", np.where(d.pZ > (d.sz_top + d.sz_bot) / 2, "top", "bottom"))
print("\nwhich edge (share / overturned):")
print(pd.crosstab(d.role, d.edge, normalize="index").round(3))
print(d.pivot_table(index="role", columns="edge", values="y", aggfunc="mean").round(3))

# within a game: the team's earlier failed challenges before this one
d = d.sort_values(["gamePk", "inning", "half", "balls", "strikes"])
d["team"] = d.challenge_team
d["prior_fail"] = d.groupby(["gamePk", "team"]).y.transform(lambda s: (1 - s).shift(fill_value=0).cumsum())
d["prior_n"] = d.groupby(["gamePk", "team"]).cumcount()
print("\nby the team's failed challenges earlier in the same game (capped at 2):")
print(d.assign(pf=d.prior_fail.clip(upper=2)).pivot_table(index="pf", columns="role", values="y", aggfunc=["size", "mean"]).round(3))
d["late"] = np.where(d.inning >= 9, "9th+", np.where(d.inning >= 7, "7-8", "1-6"))
print("\nby inning:")
print(d.pivot_table(index="late", columns="role", values="y", aggfunc=["size", "mean"]).round(3))

# signed distance from my zone (StatsAPI geometry, front of plate) -- approximate, see caveats
bins = [-99, -2, -1, 0, 1, 2, 99]
labels = ["<-2", "-2..-1", "-1..0", "0..1", "1..2", ">=2"]
d["band"] = pd.cut(d.wrong_side_in, bins, labels=labels, right=False)
print("\nshare by distance band (inches on the wrong side of the call, my zone):")
print(pd.crosstab(d.role, d.band, normalize="index").round(3))
print(d.pivot_table(index="role", columns="band", values="y", aggfunc="mean", observed=False).round(3))
d["clearly_right"] = d.wrong_side_in < -1        # the call looks right by more than an inch (my zone)
d["clearly_wrong"] = d.wrong_side_in >= 1
print("\nby inning: share of challenges on pitches whose call looks clearly right / clearly wrong (my zone):")
print(d.pivot_table(index="late", columns="role", values=["clearly_right", "clearly_wrong"], aggfunc="mean").round(3))
print("\nby strikes: same shares")
print(d.pivot_table(index="strikes", columns="role", values=["clearly_right", "clearly_wrong"], aggfunc="mean").round(3))
print("\nbatters on strike-three calls vs other strike calls: same shares")
print(d[d.role == "batter"].groupby("stands_ends")[["clearly_right", "clearly_wrong"]].mean().round(3))
# is sz_top constant per batter in 2026? (ABS uses a height-based zone)
g = d.groupby("batter_id").agg(n=("sz_top", "size"), top_sd=("sz_top", "std"), bot_sd=("sz_bot", "std"))
print("\nper-batter sd of sz_top / sz_bot (batters with >= 5 challenges):", g[g.n >= 5][["top_sd", "bot_sd"]].median().round(4).to_dict())
d.drop(columns=["band"]).to_parquet("challenges_2026.parquet")
