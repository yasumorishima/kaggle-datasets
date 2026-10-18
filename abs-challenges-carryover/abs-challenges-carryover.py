# %% [markdown]
# # Does ABS challenge skill carry from Triple-A to MLB?
#
# MLB used the **ABS challenge system** in games for the first time in 2026. Triple-A has had it since
# 2025, so many players arrived in MLB with a season of challenges already on record. This notebook
# asks one question with the dataset
# [ABS Challenges: Triple-A 2025 to MLB 2026](https://www.kaggle.com/datasets/yasunorim/mlb-abs-challenges-aaa-2025-to-mlb-2026):
#
# **If you know how a player challenged in Triple-A, how much do you know about how he will challenge in MLB?**
#
# Most players make only a handful of challenges, so a plain correlation cannot tell "the skill does
# not carry" from "the counts are too small to show it". The notebook:
#
# 1. **Skill vs luck** on each board: how much players really differ, once coin-flip noise is removed.
# 2. **Carry-over**: the correlation between a player's *true* rates on two boards, estimated with a
#    model that accounts for each player's number of challenges (no minimum-count filter).
# 3. **Prediction**: does the Triple-A record beat "everyone is league average" when predicting each
#    MLB 2026 challenge, against a shuffle floor?
# 4. **Frozen predictions for MLB 2027** from the Triple-A 2026 record, written to an output file now,
#    to be scored when the 2027 boards exist.
#
# Two measures, both checked below as exact identities in the data:
# `rate_challenges` = `n_challenges` / `n_total_sample` (**how often** a player challenges) and
# `rate_overturns` = `n_overturns` / `n_challenges` (**how often he wins**).
# `overturns_vs_exp` is not used: Savant's expected values describe what an average player would do
# with the same opportunities, mixing how often and how well.
#
# ### How this notebook was made (read before the numbers)
#
# - While sizing the data I saw the raw Spearman correlations from Triple-A 2025 to MLB 2026 for players
#   with at least 5 challenges on both boards (the same subset the starter notebook uses).
# - A first version of sections 1-3 was run once, then reviewed. The review changed the carry-over method
#   (a joint model replaced a simulated reference that used only the first board's spread), added a
#   recalibration control and a "how often" test to section 3, and rewrote the noise caveat.
# - Because sections 1-3 were seen, **none of their results is presented as a confirmatory test**; they
#   are descriptive, each with its interval and floor. The one test whose outcome nobody knows yet is
#   section 4.
#
# ### Key takeaways
#
# - **How often a player challenges carries over.** From Triple-A 2025 to MLB 2026 the correlation of
#   true challenge rates is about 0.77 for batters (95% interval 0.61 to 0.90) and for catchers (0.48 to 0.96).
# - **How often a batter wins carries over too** (interval 0.54 to 0.99; with a planted correlation of 0
#   the interval wrongly excluded 0 in 1 of 40 simulations), **but MLB batters differ little** (true spread
#   about 2 percentage points), so one Triple-A season improves MLB predictions only slightly: +2.0 per
#   1,000 challenges in log loss over the league average (0.85 to 3.47), above the shuffle floor.
# - **For catchers, carry-over of success is not shown** (the interval reaches 0, and the prediction
#   gain's interval includes 0).
# - **Plate discipline adds nothing detectable** on top of the challenge record: chase rate for batters,
#   out-of-zone called strikes for catchers.
# - Section 4 freezes predictions for MLB 2027 from the Triple-A 2026 record, to be scored next year.
#
# The numbers above are from this notebook's run; the tables below are the source.

# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, expit, logit, logsumexp
from scipy.stats import chi2, spearmanr

plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 12,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.figsize": (10, 5.5)})
RNG = np.random.default_rng(20261018)
N_BOOT = 1000
N_PERM = 1000
COLORS = {"batter": "#2F5D8A", "catcher": "#C8553D"}
OUT = os.environ.get("ABS_FIG_DIR")  # local runs save figures; on Kaggle they are shown inline
pd.set_option("display.width", 220)


def show(name):
    plt.tight_layout()
    if OUT:
        plt.savefig(os.path.join(OUT, f"{name}.png"), dpi=110)
        plt.close()
    else:
        plt.show()


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) or glob.glob(f"**/{name}", recursive=True)
    return hits[0]


players = pd.read_csv(find("abs_challenges_players.csv"), low_memory=False)
reg = players[players.game_type == "R"].copy()

x = reg[reg.n_challenges > 0]
assert np.allclose(x.rate_overturns, x.n_overturns / x.n_challenges)
assert np.allclose(x.rate_challenges, x.n_challenges / x.n_total_sample)
assert not reg.duplicated(["level", "year", "challenge_type", "player_id"]).any()

BOARDS = [("AAA", 2025), ("AAA", 2026), ("MLB", 2026)]
TYPES = ["batter", "catcher"]  # pitchers made 5 challenges in Triple-A 2025 and 179 in MLB 2026
MEASURES = {"win": ("n_overturns", "n_challenges", "rate_overturns", "How often he wins"),
            "often": ("n_challenges", "n_total_sample", "rate_challenges", "How often he challenges")}


def board(level, year, ctype):
    return reg[(reg.level == level) & (reg.year == year) & (reg.challenge_type == ctype)].set_index("player_id")


print(reg.groupby(["level", "year", "challenge_type"])[["n_challenges", "n_overturns", "n_total_sample"]].sum())

# %% [markdown]
# ## 1. Skill vs luck on each board
#
# Each player has his own true rate, drawn from a beta distribution, and his record is coin flips at
# that rate (a **beta-binomial** model, maximum likelihood). The fit gives the **spread of true rates**
# between players (standard deviation, percentage points) and `k`, the count at which a player's own
# record and the league average get **equal weight** when estimating his true rate (the shrinkage
# weight on his own record is `n / (n + k)`).
#
# **Assumption**: trials are independent given the player. If they are not (one umpire's zone over a
# game, a team's challenge budget, a manager's policy), the real noise is larger than coin flips and
# the true spreads below are **overestimates**. This applies most to *how often*: MLB teams start each
# game with two challenges and keep one when it wins, so challenges are rationed per team and game, and
# catchers often challenge on the staff's behalf. The dataset has season totals only, so this cannot
# be tested here. Picking which pitches to challenge counts as skill in this model.
#
# Intervals are 95% bootstrap intervals over players. `at_bound` is the share of bootstrap fits where
# `k` reached the upper limit (no detectable spread).


# %%
K_MAX = 1e7


def fit_betabinom(k, n):
    k, n = np.asarray(k, float), np.asarray(n, float)
    keep = n > 0
    k, n = k[keep], n[keep]
    m0 = k.sum() / n.sum()

    def nll(t):
        mu, s = expit(t[0]), np.exp(t[1])
        return -np.sum(betaln(k + mu * s, n - k + (1 - mu) * s) - betaln(mu * s, (1 - mu) * s))

    best = None
    for s0 in (3.0, 100.0, 1e4):
        r = minimize(nll, [logit(m0), np.log(s0)], method="L-BFGS-B",
                     bounds=[(-8, 8), (np.log(0.1), np.log(K_MAX))])
        if best is None or r.fun < best.fun:
            best = r
    return expit(best.x[0]), np.exp(best.x[1])


def sd_true(mu, s):
    return np.sqrt(mu * (1 - mu) / (s + 1))


def skill_summary(k, n, boot):
    mu, s = fit_betabinom(k, n)
    idx = np.arange(len(k))
    bs = np.array([fit_betabinom(k[j], n[j]) for j in (RNG.choice(idx, len(idx)) for _ in range(boot))])
    sds = sd_true(bs[:, 0], bs[:, 1])
    return {"mean": mu, "k": s, "sd": sd_true(mu, s),
            "sd_lo": np.quantile(sds, .025), "sd_hi": np.quantile(sds, .975),
            "k_lo": np.quantile(bs[:, 1], .025), "k_hi": np.quantile(bs[:, 1], .975),
            "at_bound": np.mean(bs[:, 1] > 0.99 * K_MAX)}


FITS, rows = {}, []
for level, year in BOARDS:
    for ct in TYPES:
        b = board(level, year, ct)
        b = b[b.n_total_sample > 0]
        for m, (kc, nc, rc, label) in MEASURES.items():
            d = b[b[nc] > 0]
            s = skill_summary(d[kc].values, d[nc].values, boot=N_BOOT if m == "win" else 300)
            FITS[(level, year, ct, m)] = s
            scale = 100 if m == "win" else 1000
            unit = "pts" if m == "win" else "per 1,000 opps"
            rows.append({"board": f"{level} {year}", "challenger": ct, "measure": m, "players": len(d),
                         "median_n": int(d[nc].median()), "mean_rate": round(s["mean"], 4),
                         f"true_sd": f"{scale*s['sd']:.2f} ({scale*s['sd_lo']:.2f}-{scale*s['sd_hi']:.2f}) {unit}",
                         "k": f"{s['k']:.0f} ({s['k_lo']:.0f}-{s['k_hi']:.0f})",
                         "at_bound": round(s["at_bound"], 3)})
skill = pd.DataFrame(rows)
print(skill.to_string(index=False))

# %% [markdown]
# Compare `k` with `median_n`, what a typical player has on that board. When `k` is far above
# `median_n`, one season of that player's record says little beyond the league average.

# %%
fig, ax = plt.subplots(figsize=(10, 5))
labels, w_win, w_often, cols = [], [], [], []
for level, year in BOARDS:
    for ct in TYPES:
        b = board(level, year, ct)
        nw = b.loc[b.n_challenges > 0, "n_challenges"].median()
        no = b.loc[b.n_total_sample > 0, "n_total_sample"].median()
        labels.append(f"{level} {year} {ct}s")
        w_win.append(nw / (nw + FITS[(level, year, ct, "win")]["k"]))
        w_often.append(no / (no + FITS[(level, year, ct, "often")]["k"]))
        cols.append(COLORS[ct])
y = np.arange(len(labels))
ax.barh(y + 0.2, w_often, height=0.38, color="#9AA5B1", label="How often he challenges")
ax.barh(y - 0.2, w_win, height=0.38, color=cols)
ax.barh([], [], color=COLORS["batter"], label="How often he wins (batters)")
ax.barh([], [], color=COLORS["catcher"], label="How often he wins (catchers)")
ax.set_yticks(y, labels)
ax.set_xlim(0, 1)
ax.set_xlabel("Weight on a typical player's own season record (rest: league average)")
ax.set_title("A season tells you how often a player challenges, much less how well")
ax.legend(loc="lower right")
show("fig1_skill_share")

# %% [markdown]
# ## 2. Carry-over: correlation of true rates between two boards
#
# For each pair of boards, every player on both (for *how well*: at least one challenge on each; for
# *how often*: any opportunities on each). Each board gets its own mean and spread of true rates on
# the logit scale, and the two true rates of a player have correlation `rho` (a **bivariate
# logit-normal** model with binomial records, integrated by Gauss-Hermite quadrature). `rho` is the
# carry-over with the coin-flip noise taken out; its 95% interval comes from the profile likelihood.
# `rho = 1` would mean the same ordering of true rates on both boards.
#
# Pairs: **Triple-A 2025 to MLB 2026** (the question), **Triple-A 2025 to Triple-A 2026** (next
# season without the level jump) and **Triple-A 2026 to MLB 2026** (same season, players who split time).
# Players who reached MLB are a selected group; `rho` describes them, not all Triple-A players.

# %%
PAIRS = [(("AAA", 2025), ("MLB", 2026)), (("AAA", 2025), ("AAA", 2026)), (("AAA", 2026), ("MLB", 2026))]
RHO_GRID = np.round(np.r_[np.arange(-0.9, 0.901, 0.1), 0.95, 0.99], 3)
SD_FLOOR = 0.02  # a fitted logit spread below this on either board leaves rho undefined
Q_NODES = {"win": 30, "often": 40}  # checked below against 60 and 80 nodes


def gh_nodes(q):
    z, w = np.polynomial.hermite_e.hermegauss(q)
    return z, w / w.sum()


def joint_fit(k1, n1, k2, n2, q=40, rho_grid=None):
    k1, n1, k2, n2 = (np.asarray(v, float)[:, None] for v in (k1, n1, k2, n2))
    z, w = gh_nodes(q)
    lw = np.log(w)
    za, zb = z[:, None], z[None, :]

    def logbin(k, n, eta):
        return k * -np.logaddexp(0, -eta) + (n - k) * -np.logaddexp(0, eta)

    def nll(theta, rho):
        m1, ls1, m2, ls2 = theta
        l1 = logbin(k1, n1, m1 + np.exp(ls1) * z[None, :])  # players x q
        z2 = rho * za + np.sqrt(1 - rho ** 2) * zb  # q x q
        eta2 = m2 + np.exp(ls2) * z2
        l2 = logbin(k2[:, :, None], n2[:, :, None], eta2[None])  # players x q x q
        tot = l1[:, :, None] + l2 + lw[None, :, None] + lw[None, None, :]
        return -np.sum(logsumexp(tot.reshape(len(k1), -1), axis=1))

    start = [logit(k1.sum() / n1.sum()), np.log(0.3), logit(k2.sum() / n2.sum()), np.log(0.3)]
    bounds = [(-10, 10), (np.log(1e-3), np.log(5)), (-10, 10), (np.log(1e-3), np.log(5))]

    def prof(rho, x0):
        # warm start plus two cold starts: a warm start alone can stay stuck at the spread's lower bound
        best = None
        for s0 in (None, 0.3):
            init = x0 if s0 is None else [start[0], np.log(s0), start[2], np.log(s0)]
            r = minimize(nll, init, args=(rho,), method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        return best.fun, best.x

    if rho_grid is None:
        rho_grid = RHO_GRID
    x0, vals = start, []
    for rho in rho_grid:  # warm start along the grid
        f, x0 = prof(rho, x0)
        vals.append((f, x0))
    f_all = np.array([v[0] for v in vals])
    i = int(np.argmin(f_all))
    lo_b, hi_b = rho_grid[max(i - 1, 0)], rho_grid[min(i + 1, len(rho_grid) - 1)]
    fine = np.linspace(lo_b, hi_b, 7)
    ffine = [prof(r, vals[i][1]) for r in fine]
    # profile on the grid and the fine points together, sorted by rho
    pts = sorted([(r, f, x) for r, (f, x) in zip(rho_grid, vals)] + [(r, f, x) for r, (f, x) in zip(fine, ffine)],
                 key=lambda t: t[0])
    rr = np.array([p[0] for p in pts])
    ff = np.array([p[1] for p in pts])
    j = int(np.argmin(ff))
    rho_hat, f_hat, th = rr[j], ff[j], pts[j][2]
    cut = f_hat + chi2.ppf(0.95, 1) / 2
    ins = np.where(ff <= cut)[0]
    a, b = ins.min(), ins.max()

    def cross(o, n):  # linear interpolation of the cutoff crossing between an outside and an inside point
        return rr[o] + (cut - ff[o]) * (rr[n] - rr[o]) / (ff[n] - ff[o])

    ci_lo = rr[a] if a == 0 else cross(a - 1, a)
    ci_hi = rr[b] if b == len(rr) - 1 else cross(b + 1, b)
    return {"rho": rho_hat, "lo": ci_lo, "hi": ci_hi,
            "lo_at_edge": a == 0, "hi_at_edge": b == len(rr) - 1,
            "m1": th[0], "m2": th[2], "sd1": np.exp(th[1]), "sd2": np.exp(th[3]),
            "undefined": min(np.exp(th[1]), np.exp(th[3])) < SD_FLOOR}


def pair_data(p1, p2, ct, m):
    kc, nc, rc, _ = MEASURES[m]
    a, b = board(*p1, ct), board(*p2, ct)
    d = a.join(b, how="inner", lsuffix="_1", rsuffix="_2")
    return d[(d[f"{nc}_1"] > 0) & (d[f"{nc}_2"] > 0)]


# quadrature check on the first pair: the estimate must not move with more nodes
d = pair_data(*PAIRS[0], "batter", "win")
grid = RHO_GRID
r30 = joint_fit(d.n_overturns_1, d.n_challenges_1, d.n_overturns_2, d.n_challenges_2, q=30, rho_grid=grid)
r60 = joint_fit(d.n_overturns_1, d.n_challenges_1, d.n_overturns_2, d.n_challenges_2, q=60, rho_grid=grid)
print("quadrature check, batters how well (30 vs 60 nodes): rho, lo, hi",
      [round(float(r30[k]), 3) for k in ("rho", "lo", "hi")], [round(float(r60[k]), 3) for k in ("rho", "lo", "hi")])
assert all(abs(r30[k] - r60[k]) < 0.03 for k in ("rho", "lo", "hi"))
# how often has far more trials per player, so the likelihood is sharper: every how-often row is
# refitted below with 80 nodes and must agree with the 40-node fit

rows = []
for p1, p2 in PAIRS:
    for ct in TYPES:
        for m in MEASURES:
            kc, nc, rc, _ = MEASURES[m]
            d = pair_data(p1, p2, ct, m)
            if len(d) < 20:
                continue
            r = joint_fit(d[f"{kc}_1"], d[f"{nc}_1"], d[f"{kc}_2"], d[f"{nc}_2"], q=Q_NODES[m])
            if m == "often":  # node-count check on every how-often row
                r80 = joint_fit(d[f"{kc}_1"], d[f"{nc}_1"], d[f"{kc}_2"], d[f"{nc}_2"], q=80)
                diff = max(abs(r[k] - r80[k]) for k in ("rho", "lo", "hi"))
                print(f"quadrature check {p1}->{p2} {ct} often (40 vs 80 nodes): max difference {diff:.4f}")
                assert diff < 0.03
            rows.append({"pair": f"{p1[0]} {p1[1]} -> {p2[0]} {p2[1]}", "challenger": ct, "measure": m,
                         "players": len(d), "rho": r["rho"], "ci_lo": r["lo"], "ci_hi": r["hi"],
                         "ci_edge": ("lo " if r["lo_at_edge"] else "") + ("hi" if r["hi_at_edge"] else ""),
                         "logit_mean_1": r["m1"], "logit_mean_2": r["m2"],
                         "logit_sd_1": r["sd1"], "logit_sd_2": r["sd2"], "rho_undefined": r["undefined"]})
assert np.all(np.isfinite([x[k] for x in rows for k in ("rho", "ci_lo", "ci_hi")]))
joint = pd.DataFrame(rows)
print(joint.round(3).to_string(index=False))

# %% [markdown]
# **How far to trust these intervals.** With a few challenges per player, the model has little to go
# on. For every *how well* row whose interval excludes 0, the cell below plants `rho = 0`: it simulates
# records with that row's real numbers of challenges and fitted spreads but independent true rates on
# the two boards, refits the model 40 times, and counts how often the interval wrongly excludes 0.
# A nominal 95% interval should do so about 2.5% of the time (one side).
#
# For *how well*, the point estimate of `rho` sits at the top of the searched range on every real row,
# and under a planted `rho = 0` it scatters over most of the range (`planted_0_sd_rho`): with so few
# challenges per player the profile likelihood is flat and its maximum is not informative. **Only the
# interval is read for how well**, and the figure below shows intervals without a point. For *how often*
# the second part of the cell plants `rho = 0` and `rho = 0.6` (catchers, Triple-A 2025 to MLB 2026,
# the smallest *how often* group) and checks that the point estimate comes back near the planted value.

# %%
N_CALIB = 40
calib = []
for _, row in joint[(joint.measure == "win") & (joint.ci_lo > 0) & ~joint.rho_undefined].iterrows():
    p1, p2 = [(s.split()[0], int(s.split()[1])) for s in row.pair.split(" -> ")]
    d = pair_data(p1, p2, row.challenger, "win")
    n1, n2 = d.n_challenges_1.values.astype(int), d.n_challenges_2.values.astype(int)
    wrong, est = 0, []
    for _ in range(N_CALIB):
        k1 = RNG.binomial(n1, expit(row.logit_mean_1 + row.logit_sd_1 * RNG.standard_normal(len(d))))
        k2 = RNG.binomial(n2, expit(row.logit_mean_2 + row.logit_sd_2 * RNG.standard_normal(len(d))))
        r = joint_fit(k1, n1, k2, n2, q=Q_NODES["win"])
        wrong += r["lo"] > 0
        est.append(r["rho"])
    calib.append({"pair": row.pair, "challenger": row.challenger, "real_ci_lo": row.ci_lo,
                  "planted_0_excludes_0": f"{wrong}/{N_CALIB}",
                  "planted_0_mean_rho": np.mean(est), "planted_0_sd_rho": np.std(est)})
calib = pd.DataFrame(calib)
print(calib.round(3).to_string(index=False) if len(calib) else "no how-well row excludes 0")

row = joint[(joint.pair == "AAA 2025 -> MLB 2026") & (joint.challenger == "catcher") & (joint.measure == "often")].iloc[0]
d = pair_data(("AAA", 2025), ("MLB", 2026), "catcher", "often")
n1, n2 = d.n_total_sample_1.values.astype(int), d.n_total_sample_2.values.astype(int)
m1, m2 = row.logit_mean_1, row.logit_mean_2
for planted in (0.0, 0.6):
    est, cover = [], 0
    for _ in range(10):
        z1 = RNG.standard_normal(len(d))
        z2 = planted * z1 + np.sqrt(1 - planted ** 2) * RNG.standard_normal(len(d))
        k1 = RNG.binomial(n1, expit(m1 + row.logit_sd_1 * z1))
        k2 = RNG.binomial(n2, expit(m2 + row.logit_sd_2 * z2))
        r = joint_fit(k1, n1, k2, n2, q=Q_NODES["often"])
        est.append(r["rho"])
        cover += r["lo"] <= planted <= r["hi"]
    print(f"how often, planted rho {planted:.1f}: mean estimate {np.mean(est):+.2f} (sd {np.std(est):.2f}), "
          f"interval contains it {cover}/10")

# %% [markdown]
# Interval ends are where the profile likelihood crosses the 95% cutoff, interpolated between the
# points where it was evaluated (a grid from -0.9 to 0.99). `ci_edge` marks an interval that reaches
# the end of that range, i.e. the data do not bound it on that side. `logit_sd_*` is the fitted spread of true rates on each board (logit
# scale). When either spread is essentially zero (`rho_undefined`), the players on that board are not
# detectably different, and a correlation between their true rates has no meaning; those rows are
# shown as hollow markers in the figure.

# %%
for m in MEASURES:
    d = joint[joint.measure == m].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        # for how well, the point estimate is not shown: it is biased upward (see the planted test below)
        ax.errorbar(r.rho, yi, xerr=[[r.rho - r.ci_lo], [r.ci_hi - r.rho]],
                    fmt="o" if m == "often" else "none", color=COLORS[r.challenger], ms=9, capsize=6, lw=3,
                    mfc="white" if r.rho_undefined else COLORS[r.challenger], alpha=0.5 if r.rho_undefined else 1)
    ax.set_yticks(y, [f"{r.pair}\n{r.challenger}s (n={r.players})" for _, r in d.iterrows()])
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Correlation of true rates, same player on both boards (95% interval)")
    ax.set_title(f"{MEASURES[m][3]}: does it carry over?" + ("" if m == "often" else " (95% intervals)"))
    show(f"fig2_carry_{m}")

# %% [markdown]
# For comparison with the starter notebook, the raw Spearman correlation on players with at least 5
# challenges on both boards (the starter's filter; no other threshold is reported), with a 95%
# bootstrap interval and a **one-sided** permutation p-value (shuffling the second board across players).
# For *how often* this filter selects on the outcome (`n_challenges` is the numerator of the rate), so
# low-rate players are cut; the joint model above has no such filter.

# %%
rows = []
for p1, p2 in PAIRS:
    for ct in TYPES:
        a, b = board(*p1, ct), board(*p2, ct)
        d = a.join(b, how="inner", lsuffix="_1", rsuffix="_2")
        d = d[(d.n_challenges_1 >= 5) & (d.n_challenges_2 >= 5)]
        if len(d) < 15:
            continue
        for m, (kc, nc, rc, _) in MEASURES.items():
            r1, r2 = d[f"{rc}_1"].values, d[f"{rc}_2"].values
            obs = spearmanr(r1, r2).statistic
            assert np.isfinite(obs)
            j = np.arange(len(d))
            bs = [spearmanr(r1[s], r2[s]).statistic for s in (RNG.choice(j, len(j)) for _ in range(N_BOOT))]
            pm = np.array([spearmanr(r1, RNG.permutation(r2)).statistic for _ in range(N_PERM)])
            rows.append({"pair": f"{p1[0]} {p1[1]} -> {p2[0]} {p2[1]}", "challenger": ct, "measure": m,
                         "players": len(d), "spearman": obs, "ci_lo": np.quantile(bs, .025),
                         "ci_hi": np.quantile(bs, .975), "p_one_sided": (1 + np.sum(pm >= obs)) / (N_PERM + 1)})
raw = pd.DataFrame(rows)
print(raw.round(3).to_string(index=False))

# %% [markdown]
# ## 3. Prediction: does the Triple-A 2025 record beat the league average in MLB 2026?
#
# Players on both boards. Each predictor is scored by **log loss** on the MLB 2026 record (lower is
# better), reported as the gain over **League**: per 1,000 challenges for *how well*, per 1,000
# opportunities for *how often*. Every predictor starts from the same MLB league rate for the challenger
# type, computed leaving the player out.
#
# *How well* (players with at least one challenge on each board):
# - **League**: everyone gets the league rate.
# - **Raw Triple-A**: the player's Triple-A rate (clipped to 0.02-0.98), moved to the MLB league level on
#   the logit scale. A deliberately naive reference: with one or two challenges the raw rate is extreme.
# - **Shrunk Triple-A**: the Triple-A rate pulled toward the Triple-A mean with the section 1 beta
#   (empirical Bayes), then moved to the MLB level. No parameter is fitted on MLB data.
# - **Shrunk, recalibrated**: a logistic fit of the MLB outcomes on the Shrunk logit offset, leaving one
#   player out (control for the next line).
# - **Shrunk, recalibrated + discipline**: the same plus a standardized Triple-A plate-discipline column
#   (batters `sc_bat_chase_percent`; catchers `sc_c_oz_called_strike_percent`). What discipline adds
#   (`discipline_increment`) is measured on the players who have the column, against the recalibration
#   refitted on the same players. Players with the column missing keep the recalibrated prediction.
#
# A small leak, left in and stated: in the leave-one-out fits, the other players' league rates include
# the held-out player's MLB record (one player among all MLB players of the type).
#
# *How often* (players with opportunities on both boards): **League** and **Shrunk Triple-A** only.
#
# Intervals are 95% bootstrap intervals over players. Floors: for **Shrunk Triple-A**, shuffle the
# Triple-A records across players; for **discipline**, shuffle the discipline column across players
# and refit. A gain the shuffles also reach is not evidence.


# %%
def ll(k, n, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(k * np.log(p) + (n - k) * np.log(1 - p))


def fit_logit(X, k, n, offset, ridge=1e-3):
    """Binomial logistic regression with a fixed offset: logit(p) = offset + w0 + X @ w[1:]."""
    X1 = np.column_stack([np.ones(len(X)), X])

    def f(w):
        zz = offset + X1 @ w
        return np.sum(n * np.logaddexp(0, zz) - k * zz) + ridge * np.sum(w[1:] ** 2)

    def g(w):
        return X1.T @ (n * expit(offset + X1 @ w) - k) + 2 * ridge * np.r_[0, w[1:]]

    return minimize(f, np.zeros(X1.shape[1]), jac=g, method="BFGS").x


def loo_recal(off, extra, km, nm, league):
    """Leave-one-player-out logistic recalibration; extra=None for offset only."""
    out = np.empty(len(off))
    for i in range(len(off)):
        tr = np.ones(len(off), bool)
        tr[i] = False
        cols = [off[tr]]
        xi = [off[i]]
        if extra is not None:
            mu_, sd_ = extra[tr].mean(), extra[tr].std()
            cols.append((extra[tr] - mu_) / sd_)
            xi.append((extra[i] - mu_) / sd_)
        w = fit_logit(np.column_stack(cols), km[tr], nm[tr], logit(league[tr]))
        out[i] = expit(logit(league[i]) + w[0] + np.dot(w[1:], xi))
    return out


def gain_row(name, ct, measure, base, loss, nm, extra=None):
    gi = base - loss
    j = np.arange(len(gi))
    bs = [gi[s].sum() / nm[s].sum() for s in (RNG.choice(j, len(j)) for _ in range(N_BOOT))]
    row = {"measure": measure, "challenger": ct, "predictor": name, "players": len(gi),
           "units": int(nm.sum()), "gain_per_1000": 1000 * gi.sum() / nm.sum(),
           "ci_lo": 1000 * np.quantile(bs, .025), "ci_hi": 1000 * np.quantile(bs, .975)}
    row.update(extra or {})
    return row


DISC = {"batter": "sc_bat_chase_percent", "catcher": "sc_c_oz_called_strike_percent"}
N_DISC_PERM = 200
pred_rows = []
for ct in TYPES:
    A, M = board("AAA", 2025, ct), board("MLB", 2026, ct)
    for m, (kc, nc, rc, _) in MEASURES.items():
        allm = M[M[nc] > 0]
        d = A.join(M, how="inner", lsuffix="_a", rsuffix="_m")
        d = d[(d[f"{nc}_a"] > 0) & (d[f"{nc}_m"] > 0)].copy()
        ka, na = d[f"{kc}_a"].values.astype(float), d[f"{nc}_a"].values.astype(float)
        km, nm = d[f"{kc}_m"].values.astype(float), d[f"{nc}_m"].values.astype(float)
        league = (allm[kc].sum() - km) / (allm[nc].sum() - nm)
        fa = FITS[("AAA", 2025, ct, m)]
        a_, b_ = fa["mean"] * fa["k"], (1 - fa["mean"]) * fa["k"]

        def shrunk(ka, na):
            return expit(logit(league) + logit((a_ + ka) / (a_ + b_ + na)) - logit(fa["mean"]))

        base = ll(km, nm, league)
        shr = shrunk(ka, na)
        fl = np.array([(base - ll(km, nm, shrunk(ka[s], na[s]))).sum() / nm.sum()
                       for s in (RNG.permutation(len(d)) for _ in range(N_PERM))])
        g = (base - ll(km, nm, shr)).sum() / nm.sum()
        assert np.isfinite(g) and np.all(np.isfinite(fl))
        pred_rows.append(gain_row("Shrunk Triple-A", ct, m, base, ll(km, nm, shr), nm, {
            "floor_mean": 1000 * fl.mean(), "floor_97.5": 1000 * np.quantile(fl, .975),
            "p_vs_floor_one_sided": (1 + np.sum(fl >= g)) / (N_PERM + 1)}))
        if m != "win":
            continue
        raw_p = expit(logit(league) + logit(np.clip(ka / na, 0.02, 0.98)) - logit(fa["mean"]))
        pred_rows.append(gain_row("Raw Triple-A", ct, m, base, ll(km, nm, raw_p), nm))
        off = logit(shr) - logit(league)
        # gate: fitted on all players, the recalibration must not lose to League in-sample
        # (it contains League as the case w = 0); a wrong offset in the fit breaks this
        w_all = fit_logit(off[:, None], km, nm, logit(league))
        assert ll(km, nm, expit(logit(league) + w_all[0] + w_all[1] * off)).sum() <= base.sum() + 1e-6
        rec = loo_recal(off, None, km, nm, league)
        pred_rows.append(gain_row("Shrunk, recalibrated", ct, m, base, ll(km, nm, rec), nm))
        disc = d[f"{DISC[ct]}_a"].values.astype(float)
        ok = ~np.isnan(disc)
        assert ok.sum() >= 20 and np.nanstd(disc) > 0, f"{ct}: discipline column too sparse"
        # the increment is measured on the players who have the column, against the same recalibration
        # refitted on those players only, so it is discipline alone
        k_ok, n_ok, lg_ok, off_ok = km[ok], nm[ok], league[ok], off[ok]
        base_ok = loo_recal(off_ok, None, k_ok, n_ok, lg_ok)
        with_ok = loo_recal(off_ok, disc[ok], k_ok, n_ok, lg_ok)
        inc_i = ll(k_ok, n_ok, base_ok) - ll(k_ok, n_ok, with_ok)
        inc = inc_i.sum() / n_ok.sum()
        jj = np.arange(len(inc_i))
        inc_bs = [inc_i[s].sum() / n_ok[s].sum() for s in (RNG.choice(jj, len(jj)) for _ in range(N_BOOT))]
        dfl = []
        for _ in range(N_DISC_PERM):
            sh = loo_recal(off_ok, RNG.permutation(disc[ok]), k_ok, n_ok, lg_ok)
            dfl.append((ll(k_ok, n_ok, base_ok) - ll(k_ok, n_ok, sh)).sum() / n_ok.sum())
        dfl = np.array(dfl)
        assert np.isfinite(inc) and np.all(np.isfinite(dfl))
        recd = rec.copy()
        recd[ok] = with_ok
        pred_rows.append(gain_row("Shrunk, recalibrated + discipline", ct, m, base, ll(km, nm, recd), nm, {
            "discipline_increment": 1000 * inc,
            "increment_ci_lo": 1000 * np.quantile(inc_bs, .025), "increment_ci_hi": 1000 * np.quantile(inc_bs, .975),
            "floor_mean": 1000 * dfl.mean(), "floor_97.5": 1000 * np.quantile(dfl, .975),
            "p_vs_floor_one_sided": (1 + np.sum(dfl >= inc)) / (N_DISC_PERM + 1),
            "missing_discipline": int((~ok).sum())}))
pred = pd.DataFrame(pred_rows)
assert np.all(np.isfinite(pred[["gain_per_1000", "ci_lo", "ci_hi"]].values))
print(pred.round(3).to_string(index=False))

# %% [markdown]
# For **Shrunk Triple-A** the floor is about the Triple-A records; for the discipline line, the floor
# and the interval `increment_ci_*` are about `discipline_increment`.

# %%
d = pred[pred.predictor != "Raw Triple-A"].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(d))[::-1]
for yi, (_, r) in zip(y, d.iterrows()):
    ax.errorbar(r.gain_per_1000, yi, xerr=[[r.gain_per_1000 - r.ci_lo], [r.ci_hi - r.gain_per_1000]],
                fmt="o", color=COLORS[r.challenger], ms=9, capsize=4)
unit = {"win": "per 1,000 challenges", "often": "per 1,000 opportunities"}
ax.set_yticks(y, [f"{r.predictor} ({r.measure})\n{r.challenger}s, {unit[r.measure]}" for _, r in d.iterrows()])
ax.axvline(0, color="grey", lw=1)
ax.set_xlabel("Log-loss gain over league average (95% interval)")
ax.set_title("Does the Triple-A 2025 record help predict MLB 2026?")
show("fig3_prediction")

# %% [markdown]
# ## 4. Frozen predictions for MLB 2027
#
# Sections 1-3 were seen before this notebook was finalized. The test below is not: it predicts the
# MLB 2027 regular season from the Triple-A 2026 record, using exactly the **Shrunk Triple-A** rule
# from section 3 (Triple-A 2026 beta from section 1), and is written to `abs_frozen_2027.csv` in this
# notebook's output, whose version date fixes when it was made.
#
# The file stores each player's predicted **logit offset** from the MLB league rate, because the 2027
# league rate is not known yet. Scoring rule, fixed now:
#
# - **Scored players**: every player in the file who is on an MLB 2027 regular-season board of the same
#   challenger type with at least one challenge (*how well*) or one opportunity (*how often*).
# - **League rate**: for each scored player, the rate of all MLB 2027 regular-season players of that
#   type with at least one challenge (or opportunity), leaving the player out (as in section 3).
# - **Prediction**: `expit(logit(league) + offset_logit)`, scored by log loss against League alone.
# - **Claim** "the Triple-A record helps" for a measure and challenger type: the 95% bootstrap interval
#   (1,000 resamples of the scored players) of the gain excludes 0 **and** the gain exceeds the 97.5%
#   point of 1,000 shuffles of the offsets across the scored players.
#
# The beta used for each group (`beta_mean`, `beta_k`) is stored in the file so the offsets can be
# recomputed from the Triple-A 2026 records.

# %%
frozen = []
for ct in TYPES:
    A = board("AAA", 2026, ct)
    for m, (kc, nc, rc, _) in MEASURES.items():
        f = FITS[("AAA", 2026, ct, m)]
        a_, b_ = f["mean"] * f["k"], (1 - f["mean"]) * f["k"]
        d = A[A[nc] > 0]
        post = (a_ + d[kc]) / (a_ + b_ + d[nc])
        frozen.append(pd.DataFrame({"player_id": d.index, "player_name": d.player_name.values,
                                    "challenger": ct, "measure": m, "aaa2026_k": d[kc].values,
                                    "aaa2026_n": d[nc].values, "beta_mean": f["mean"], "beta_k": f["k"],
                                    "offset_logit": (logit(post) - logit(f["mean"])).values}))
frozen = pd.concat(frozen, ignore_index=True)
assert np.all(np.isfinite(frozen.offset_logit)) and not frozen.duplicated(["player_id", "challenger", "measure"]).any()
frozen.to_csv("abs_frozen_2027.csv", index=False)
print(frozen.groupby(["challenger", "measure"]).offset_logit.describe().round(3))

# %% [markdown]
# ## Summary
#
# Main numbers from the tables above, printed in one place.

# %%
for ct in TYPES:
    s = FITS[("MLB", 2026, ct, "win")]
    print(f"[{ct}s] MLB 2026 true spread of success rate {100*s['sd']:.1f} pts "
          f"({100*s['sd_lo']:.1f}-{100*s['sd_hi']:.1f}); equal weight after {s['k']:.0f} challenges")
    for m in MEASURES:
        r = joint[(joint.pair == "AAA 2025 -> MLB 2026") & (joint.challenger == ct) & (joint.measure == m)].iloc[0]
        note = " [undefined: no detectable spread on one board]" if r.rho_undefined else ""
        point = f"rho {r.rho:+.2f} " if m == "often" else "rho "
        print(f"  AAA25->MLB26 {m:5s}: {point}(95% {r.ci_lo:+.2f} to {r.ci_hi:+.2f}), {r.players} players{note}")
        p = pred[(pred.challenger == ct) & (pred.measure == m) & (pred.predictor == "Shrunk Triple-A")].iloc[0]
        print(f"     Shrunk AAA vs league: {p.gain_per_1000:+.2f} per 1,000 (95% {p.ci_lo:+.2f} to {p.ci_hi:+.2f}; "
              f"shuffle 97.5% {p['floor_97.5']:+.2f}, one-sided p {p.p_vs_floor_one_sided:.3f})")
