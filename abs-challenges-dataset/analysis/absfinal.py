"""Numbers for the rewritten ABS article, from the final Kaggle version (after the 2026 regular season).
Run in ~/claude-scratch/absdeep/kv3.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest

P = pd.read_csv("abs_challenges_players.csv", low_memory=False)
B = pd.read_csv("abs_aaa2025_to_mlb2026.csv")
pd.set_option("display.width", 220)


def wilson(k, n):
    r = binomtest(int(k), int(n)).proportion_ci(method="wilson")
    return r.low, r.high


print("rows", P.shape, B.shape)
print("\n== boards: share won, expected, runs ==")
rows = []
for (lvl, yr, gt), d0 in P.groupby(["level", "year", "game_type"]):
    for ct, d in d0.groupby("challenge_type"):
        n, k = int(d.n_challenges.sum()), int(d.n_overturns.sum())
        if n < 50:
            continue
        lo, hi = wilson(k, n)
        rows.append(dict(board=f"{lvl} {yr} {gt}", who=ct, players=int((d.n_challenges > 0).sum()), n=n, won=k,
                         share=round(k / n, 3), lo=round(lo, 3), hi=round(hi, 3),
                         exp=round(d.exp_chal_gained.sum(), 1), won_minus_exp=round(k - d.exp_chal_gained.sum(), 1),
                         runs=round(d.net_net_runs.sum(), 1) if "net_net_runs" in d else None,
                         runs_gained=round(d.n_chal_runs_gained.sum(), 1), runs_lost=round(d.n_chal_runs_lost.sum(), 1),
                         k=int(d.n_strikeouts.sum()), bb=int(d.n_walks.sum())))
print(pd.DataFrame(rows).to_string(index=False))

m = P[(P.level == "MLB") & (P.year == 2026) & (P.game_type == "R")]
print("\n== MLB 2026 R: columns sample for one catcher ==")
print(m[m.challenge_type == "catcher"].sort_values("n_challenges", ascending=False).head(1).T.head(60).to_string())

for ct in ("catcher", "batter"):
    d = m[(m.challenge_type == ct) & (m.n_challenges >= 10)].copy()
    d["share"] = d.n_overturns / d.n_challenges
    cols = ["player_name", "team_abbr", "n_challenges", "n_overturns", "share", "exp_chal_gained", "overturns_vs_exp" if "overturns_vs_exp" in d else "net_net_chal", "net_net_runs", "n_chal_runs_gained", "rate_challenges"]
    cols = [c for c in cols if c in d]
    print(f"\n== MLB 2026 R {ct}s >= 10 challenges: n={len(d)}; top/bottom by overturns vs expected ==")
    key = "overturns_vs_exp" if "overturns_vs_exp" in d else "net_net_chal"
    print(d.sort_values(key, ascending=False)[cols].head(8).to_string(index=False))
    print(d.sort_values(key)[cols].head(5).to_string(index=False))
    print(d.sort_values("n_challenges", ascending=False)[cols].head(5).to_string(index=False))

print("\n== batters: chase vs success (MLB 2026 R, >=100 PA, >=10 challenges) ==")
b = m[(m.challenge_type == "batter") & (m.sc_bat_pa >= 100)].copy()
bc = b[b.n_challenges >= 10].copy()
bc["share"] = bc.n_overturns / bc.n_challenges
for y, d in (("share", bc), ("overturns_vs_exp", bc), ("rate_challenges", b)):
    r = spearmanr(d.sc_bat_chase_percent, d[y], nan_policy="omit")
    print(f"chase vs {y}: rho {r.statistic:+.3f} p {r.pvalue:.3g} n {d[y].notna().sum()}")
b["tier"] = pd.qcut(b.sc_bat_chase_percent, 3, labels=["low", "mid", "high"])
t = b.groupby("tier", observed=True).agg(batters=("player_id", "size"), chase=("sc_bat_chase_percent", "median"),
                                         n=("n_challenges", "sum"), won=("n_overturns", "sum"), exp=("exp_chal_gained", "sum"))
t["share"] = (t.won / t.n).round(3); t["ci"] = [f"{wilson(k, n)[0]:.3f}-{wilson(k, n)[1]:.3f}" for k, n in zip(t.won, t.n)]
t["exp_share"] = (t.exp / t.n).round(3)
print(t.to_string())
bc[["player_name", "sc_bat_chase_percent", "n_challenges", "share"]].to_csv("scatter_chase.csv", index=False)
print("lowest chase batters with >=10 challenges:\n", bc.nsmallest(5, "sc_bat_chase_percent")[["player_name", "sc_bat_chase_percent", "n_challenges", "share"]].to_string(index=False))
print("highest chase:\n", bc.nlargest(5, "sc_bat_chase_percent")[["player_name", "sc_bat_chase_percent", "n_challenges", "share"]].to_string(index=False))

print("\n== catchers: framing proxy ==")
c = m[m.challenge_type == "catcher"]
q = c.dropna(subset=["sc_c_framing_runs", "sc_c_oz_called_strike_percent"])
r = spearmanr(q.sc_c_oz_called_strike_percent, q.sc_c_framing_runs)
print(f"proxy vs framing runs rho {r.statistic:+.3f} p {r.pvalue:.2g} n {len(q)}")
q[["player_name", "sc_c_oz_called_strike_percent", "sc_c_framing_runs", "sc_c_oz_takes"]].to_csv("scatter_framing.csv", index=False)
cat = P[(P.challenge_type == "catcher") & (P.game_type == "R") & (P.sc_c_oz_takes >= 300) & (P.n_challenges >= 10)]
for (lvl, yr), d in cat.groupby(["level", "year"]):
    r = spearmanr(d.sc_c_oz_called_strike_percent, d.overturns_vs_exp)
    print(f"{lvl} {yr} proxy vs overturns_vs_exp rho {r.statistic:+.3f} p {r.pvalue:.3g} n {len(d)}")
cat[(cat.level == "MLB") & (cat.year == 2026)][["player_name", "sc_c_oz_called_strike_percent", "overturns_vs_exp", "n_challenges"]].to_csv("scatter_framing_vs_chal.csv", index=False)

print("\n== AAA 2025 -> MLB 2026 ==")
both = B[(B.n_challenges_aaa2025 >= 5) & (B.n_challenges_mlb2026 >= 5)]
for mtr in ("rate_challenges", "rate_overturns", "overturns_vs_exp", "sc_bat_chase_percent"):
    for ct, d in both.groupby("challenge_type"):
        x, y = d[f"{mtr}_aaa2025"], d[f"{mtr}_mlb2026"]; k = x.notna() & y.notna()
        if k.sum() >= 10:
            r = spearmanr(x[k], y[k]); print(f"{mtr:22s} {ct:7s} n {k.sum():3d} rho {r.statistic:+.3f} p {r.pvalue:.3g}")
g = both[both.challenge_type == "batter"]
g[["player_name", "rate_challenges_aaa2025", "rate_challenges_mlb2026", "overturns_vs_exp_aaa2025", "overturns_vs_exp_mlb2026"]].to_csv("scatter_carry.csv", index=False)
print("carry examples (batters, highest AAA challenge rate):\n",
      g.nlargest(5, "rate_challenges_aaa2025")[["player_name", "rate_challenges_aaa2025", "rate_challenges_mlb2026", "n_challenges_aaa2025", "n_challenges_mlb2026"]].to_string(index=False))

print("\n== age (MLB 2026 R batters >=100 PA) ==")
b["age_group"] = pd.cut(b.age, [0, 25, 29, 33, 50], labels=["<=25", "26-29", "30-33", "34+"])
t6 = b.groupby("age_group", observed=True).agg(batters=("player_id", "size"), n=("n_challenges", "sum"), won=("n_overturns", "sum"), exp=("exp_chal_gained", "sum"))
t6["share"] = (t6.won / t6.n).round(3); t6["ci"] = [f"{wilson(k, n)[0]:.3f}-{wilson(k, n)[1]:.3f}" for k, n in zip(t6.won, t6.n)]
t6["won_minus_exp"] = (t6.won - t6.exp).round(1)
print(t6.to_string())
