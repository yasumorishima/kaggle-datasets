# %% [markdown]
# # ABS Challenges: who wins them, and does it carry from Triple-A to MLB?
#
# MLB introduced the **ABS (Automated Ball-Strike) challenge system** in 2026: batters, pitchers and
# catchers can challenge the umpire's ball/strike call, and Hawk-Eye decides. Triple-A has used it
# since 2025, and MLB tested it in 2025 spring training.
#
# This notebook is a starting point for the dataset
# [ABS Challenges: Triple-A 2025 to MLB 2026](https://www.kaggle.com/datasets/yasunorim/mlb-abs-challenges-aaa-2025-to-mlb-2026):
#
# 1. How many challenges each board has, and how often they succeed
# 2. Does plate discipline (chase rate) go with challenge success for batters?
# 3. Catchers: does a framing proxy go with challenge success?
# 4. Does challenge skill carry over from Triple-A 2025 to MLB 2026 for the same player?
# 5. Age and handedness
#
# Every number below is computed from the dataset in this notebook.

# %%
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 12,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.figsize": (9, 5.5)})

def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) or glob.glob(f"**/{name}", recursive=True)
    return hits[0]

players = pd.read_csv(find("abs_challenges_players.csv"), low_memory=False)
bridge = pd.read_csv(find("abs_aaa2025_to_mlb2026.csv"))
print(players.shape, bridge.shape)

# %% [markdown]
# ## 1. The boards
#
# One row per player per board. A board is a level (MLB / AAA), a season, a game type
# (R regular season, S spring training) and who challenged (batter, pitcher, catcher).

# %%
boards = (players.groupby(["level", "year", "game_type", "challenge_type"])
          .agg(players=("player_id", "size"), challenges=("n_challenges", "sum"),
               overturns=("n_overturns", "sum")))
boards["overturn_rate"] = (boards["overturns"] / boards["challenges"]).round(3)
print(boards)

# %%
reg = boards.reset_index()
reg = reg[(reg.game_type == "R") & (reg.challenges >= 100)]
labels = reg.level + " " + reg.year.astype(str) + " " + reg.challenge_type
fig, ax = plt.subplots()
ax.barh(labels, reg.overturn_rate, color="#4C72B0")
ax.set_xlabel("Share of challenges won")
ax.set_title("Challenge success by board (regular season)")
ax.set_xlim(0, 1)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Batters: plate discipline and challenges
#
# `sc_bat_chase_percent` is the share of pitches outside the zone a batter swung at, from Statcast
# for the same level and season as the board. A batter who rarely chases may also judge
# borderline pitches well. Here: MLB 2026 regular season, batters with at least 100 plate
# appearances and at least 5 challenges.

# %%
b = players[(players.level == "MLB") & (players.year == 2026) & (players.game_type == "R")
            & (players.challenge_type == "batter") & (players.sc_bat_pa >= 100)]
bc = b[b.n_challenges >= 5]
for y in ("rate_challenges", "rate_overturns", "overturns_vs_exp"):
    d = b if y == "rate_challenges" else bc
    r = spearmanr(d.sc_bat_chase_percent, d[y], nan_policy="omit")
    print(f"chase % vs {y}: Spearman {r.statistic:+.3f} (p={r.pvalue:.3g}, n={d[y].notna().sum()})")

fig, ax = plt.subplots()
ax.scatter(bc.sc_bat_chase_percent, bc.rate_overturns, s=bc.n_challenges * 3, alpha=0.5, color="#4C72B0")
ax.set_xlabel("Chase rate, % (Statcast, MLB 2026)")
ax.set_ylabel("Share of challenges won")
ax.set_title("Batters: chase rate vs challenge success")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Catchers: a framing proxy and challenges
#
# `sc_c_oz_called_strike_percent`: of the pitches outside the zone that the batter took, the share
# called a strike while this catcher was behind the plate. It exists for every board, including
# Triple-A, where Savant's official framing leaderboard does not. On the MLB 2026 regular season it
# can be compared with the official framing runs (`sc_c_framing_runs`, qualified catchers only).

# %%
c26 = players[(players.level == "MLB") & (players.year == 2026) & (players.game_type == "R")
              & (players.challenge_type == "catcher")]
q = c26.dropna(subset=["sc_c_framing_runs"])
r = spearmanr(q.sc_c_oz_called_strike_percent, q.sc_c_framing_runs)
print(f"proxy vs official framing runs (MLB 2026, n={len(q)}): Spearman {r.statistic:+.3f} (p={r.pvalue:.3g})")

cat = players[(players.challenge_type == "catcher") & (players.game_type == "R")
              & (players.sc_c_oz_takes >= 300) & (players.n_challenges >= 10)]
for (lvl, yr), g in cat.groupby(["level", "year"]):
    r = spearmanr(g.sc_c_oz_called_strike_percent, g.overturns_vs_exp)
    print(f"{lvl} {yr}: proxy vs overturns_vs_exp  Spearman {r.statistic:+.3f} (p={r.pvalue:.3g}, n={len(g)})")

fig, ax = plt.subplots()
for (lvl, yr), g in cat.groupby(["level", "year"]):
    ax.scatter(g.sc_c_oz_called_strike_percent, g.overturns_vs_exp, alpha=0.6, label=f"{lvl} {yr}")
ax.axhline(0, color="grey", lw=1)
ax.set_xlabel("Out-of-zone takes called strike, %")
ax.set_ylabel("Overturns vs expected")
ax.set_title("Catchers: framing proxy vs challenge results")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Does it carry from Triple-A 2025 to MLB 2026?
#
# `abs_aaa2025_to_mlb2026.csv` has players who were on both regular-season boards, with each
# measure side by side. Below: the same player's Triple-A 2025 value against the same player's MLB 2026 value,
# for players with at least 5 challenges at both levels.

# %%
both = bridge[(bridge.n_challenges_aaa2025 >= 5) & (bridge.n_challenges_mlb2026 >= 5)]
rows = []
for m in ("rate_challenges", "rate_overturns", "overturns_vs_exp", "sc_bat_chase_percent"):
    for ct, g in both.groupby("challenge_type"):
        x, y = g[f"{m}_aaa2025"], g[f"{m}_mlb2026"]
        k = x.notna() & y.notna()
        if k.sum() >= 10:
            r = spearmanr(x[k], y[k])
            rows.append({"measure": m, "challenger": ct, "n": int(k.sum()),
                         "spearman": round(r.statistic, 3), "p": round(r.pvalue, 4)})
carry = pd.DataFrame(rows)
print(carry)

# %%
g = both[both.challenge_type == "batter"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, m, name in ((axes[0], "rate_challenges", "Challenge rate"), (axes[1], "rate_overturns", "Share won")):
    ax.scatter(g[f"{m}_aaa2025"], g[f"{m}_mlb2026"], alpha=0.5, color="#4C72B0")
    lim = [0, max(g[f"{m}_aaa2025"].max(), g[f"{m}_mlb2026"].max()) * 1.05]
    ax.plot(lim, lim, color="grey", lw=1)
    ax.set_xlabel("Triple-A 2025")
    ax.set_ylabel("MLB 2026")
    ax.set_title(f"Batters: {name}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Age and handedness
#
# Bio columns (`bats`, `age`, ...) are on every row. Challenge rate and success by batting side and
# age group, MLB 2026 regular-season batters with at least 100 plate appearances.

# %%
b = b.assign(age_group=pd.cut(b.age, [0, 25, 29, 33, 50], labels=["<=25", "26-29", "30-33", "34+"]))
summary = b.groupby("bats").agg(batters=("player_id", "size"), challenges=("n_challenges", "sum"),
                                won=("n_overturns", "sum"), mean_rate=("rate_challenges", "mean"))
summary["share_won"] = (summary.won / summary.challenges).round(3)
print(summary)
age = b.groupby("age_group", observed=True).agg(batters=("player_id", "size"), challenges=("n_challenges", "sum"),
                                                won=("n_overturns", "sum"))
age["share_won"] = (age.won / age.challenges).round(3)
print(age)

# %% [markdown]
# ## Ideas to take further
#
# - Join to Statcast pitch data by `player_id` to see which pitches get challenged.
# - Use the Triple-A 2025 columns (challenge and plate discipline) to predict MLB 2026 challenge success.
# - Compare spring-training habits (`game_type == "S"`) with the regular season.
# - In MLB 2025 spring only about two thirds of pitches have a Statcast zone: check
#   `*_zone_tracked_percent` before using that board's zone columns.
