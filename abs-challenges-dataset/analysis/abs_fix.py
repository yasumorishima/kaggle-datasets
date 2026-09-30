"""Re-derive the claims the audit knocked down, from pitch-level data (no Savant expected values).
Run in ~/claude-scratch/absdeep (challenges_2026.parquet from absdeep.py, kv3/ csv).
"""
import numpy as np
import pandas as pd
from scipy.stats import binom, spearmanr

d = pd.read_parquet("challenges_2026.parquet")
P = pd.read_csv("kv3/abs_challenges_players.csv", low_memory=False)
m = P[(P.level == "MLB") & (P.year == 2026) & (P.game_type == "R")]

# how clear were the challenged pitches, by role ("clear" = more than an inch from the edge in my zone)
d["cl"] = np.select([d.wrong_side_in >= 1, d.wrong_side_in < -1], ["clearly wrong call", "clearly right call"], "within an inch")
t = pd.crosstab(d.role, d.cl, normalize="index").round(3)
print("share of challenged pitches by clarity (my zone):\n", t)
print("\nshare won inside each clarity class:\n", d.pivot_table(index="role", columns="cl", values="y", aggfunc="mean").round(3))
print(pd.crosstab(d.role, d.cl))
# what the gap would be if catchers had the batters' mix of clarity classes (catchers' within-class rates)
wc = d[d.role == "catcher"].groupby("cl").y.mean()
wb = d[d.role == "batter"].groupby("cl").y.mean()
mixb = d[d.role == "batter"].cl.value_counts(normalize=True)
mixc = d[d.role == "catcher"].cl.value_counts(normalize=True)
print(f"\ncatchers' rates with batters' mix: {(wc * mixb).sum():.3f} (actual catchers {d[d.role == 'catcher'].y.mean():.3f}, batters {d[d.role == 'batter'].y.mean():.3f})")
print(f"batters' rates with catchers' mix: {(wb * mixc).sum():.3f}")

# chase tiers (Savant chase) joined to batters' challenged pitches
b = m[(m.challenge_type == "batter") & (m.sc_bat_pa >= 100)].copy()
b["tier"] = pd.qcut(b.sc_bat_chase_percent, 3, labels=["low", "mid", "high"])
x = d[d.role == "batter"].merge(b[["player_id", "tier"]], left_on="batter_id", right_on="player_id", how="inner")
print(f"\nbatter challenges joined to chase tiers: {len(x)} of {int((d.role == 'batter').sum())}")
print(pd.crosstab(x.tier, x.cl, normalize="index").round(3))
print(x.pivot_table(index="tier", columns="cl", values="y", aggfunc="mean", observed=False).round(3))
print(x.groupby("tier", observed=True).y.agg(["size", "mean"]).round(3))

# selectivity: challenges made vs Savant's expected number of challenges (an average challenger's count in
# the player's opportunities); and binomial with the catchers' average share won
c = m[(m.challenge_type == "catcher") & (m.n_challenges >= 60)].copy()
avg = m[m.challenge_type == "catcher"].n_overturns.sum() / m[m.challenge_type == "catcher"].n_challenges.sum()
for name in ("J.T. Realmuto", "Francisco Alvarez", "Salvador Perez"):
    r = c[c.player_name == name].iloc[0]
    up = binom.sf(r.n_overturns - 1, r.n_challenges, avg)
    lo = binom.cdf(r.n_overturns, r.n_challenges, avg)
    print(f"{name}: {r.n_overturns:.0f}/{r.n_challenges:.0f}  exp_chal {r.exp_chal:.1f}  challenges/exp {r.n_challenges / r.exp_chal:.2f}  "
          f"P(>=) {up:.2e} ({1 - (1 - up) ** len(c):.4f} over {len(c)})  P(<=) {lo:.4f} ({1 - (1 - lo) ** len(c):.3f} over {len(c)})")
c["sel"] = c.n_challenges / c.exp_chal
c["share"] = c.n_overturns / c.n_challenges
r = spearmanr(c.sel, c.share)
print(f"catchers >=60: challenges/expected vs share won rho {r.statistic:+.3f} p {r.pvalue:.3g} n {len(c)}")
print(c.sort_values("sel")[["player_name", "n_challenges", "exp_chal", "sel", "share"]].head(5).round(2).to_string(index=False))
# pitcher rows: exp_chal equals the catchers' total -- check
pp = m[m.challenge_type == "pitcher"]
print("pitcher rows exp_chal sum", round(pp.exp_chal.sum(), 1), "catcher", round(m[m.challenge_type == "catcher"].exp_chal.sum(), 1))
# age with overturns_vs_exp
bb = m[(m.challenge_type == "batter") & (m.sc_bat_pa >= 100)].copy()
bb["age_group"] = pd.cut(bb.age, [0, 25, 29, 33, 50], labels=["<=25", "26-29", "30-33", "34+"])
print(bb.groupby("age_group", observed=True).overturns_vs_exp.agg(["size", "sum", "mean"]).round(2))
