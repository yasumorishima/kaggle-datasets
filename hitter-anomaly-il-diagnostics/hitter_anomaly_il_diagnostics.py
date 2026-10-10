# %% [markdown]
# # Why Did the Shuffled Anomaly Block Beat the Real One? (Diagnostics)
#
# The frozen run of [Do Hitters Drift From Normal Before the IL?](https://www.kaggle.com/code/yasunorim/do-hitters-drift-from-normal-before-the-il) found no support for either test. One reported number did not fit: the floor for P2, which rotates each hitter's deviations in time, added **+0.040 AUC on average** over the rival model, while the real, correctly timed deviations added only **+0.007**.
#
# This notebook runs the frozen pipeline **unchanged** (fetched from commit `20cc037`, checked by md5) up to the models and asks why. It is **exploratory**: nothing here changes the frozen verdicts.
#
# Candidate explanations, each with a test:
#
# 1. **Missing values move with the rotation.** Early-season rows have no deviation yet; rotating carries those gaps to other dates. Test: rotate only the defined values and keep every gap where it was.
# 2. **The rotated block keeps who a hitter is, but drops when things happened.** A model trained on correctly timed deviations may learn 2024-25 timing patterns that do not carry over to 2026, while a rotated block can only teach hitter-level traits. Test: replace the block with each hitter-season's average deviation (non-causal, explanation only).
# 3. **Something else.** Permutation importance of a rotated model, and where the gaps sit relative to the injured list.

# %%
import hashlib

import requests

SHA = "20cc03733378e186aa1ddeac0ce5f49a33917b7e"
URL = f"https://raw.githubusercontent.com/yasumorishima/kaggle-datasets/{SHA}/hitter-anomaly-il/hitter_anomaly_il.py"
src = requests.get(URL, timeout=60).text
assert hashlib.md5(src.encode()).hexdigest() == "2535beee8db43c6f483c97bd5cb47c57", "not the frozen notebook"
cut = src.index("# %% [markdown]\n# ### Time-shift floor")
exec(compile(src[:cut], "frozen_hitter_anomaly_il", "exec"))
real_gap = main.loc["M2 - B0", "AUC"]
print(f"frozen pipeline reproduced: AUC(M2) - AUC(B0) = {real_gap:+.6f} (frozen run +0.006792)")
if not FAKE:  # FAKE is the frozen notebook's code-test switch (random placements)
    assert abs(real_gap - 0.006791949463195834) < 1e-4, "inputs changed since the frozen run"

# %% [markdown]
# ## The rotations, side by side
#
# Ten seeds each (100-109, the first ten of the frozen floor). B0 is the real rival model throughout; labels are never moved.

# %%
auc_b0 = auc(y14, test.p_B0.to_numpy())
SEEDS = range(100, 110)


def k_for(n, rng):
    return int(rng.integers(30, n - 30 + 1)) if n >= 61 else n // 2


def rot_nan_moves(df, rng):
    """The frozen floor: the whole block, gaps included, rolls together."""
    out = df.copy()
    blk = out[BLOCK].to_numpy().copy()
    for _, ix in out.groupby(["batter", "season"]).indices.items():
        if len(ix) >= 2:
            blk[ix] = np.roll(blk[ix], k_for(len(ix), rng), axis=0)
    out[BLOCK] = blk
    return out


def rot_gaps_fixed(df, rng):
    """Only defined values roll, column by column; every gap stays on its date."""
    out = df.copy()
    blk = out[BLOCK].to_numpy().copy()
    for _, ix in out.groupby(["batter", "season"]).indices.items():
        k_seed = rng.integers(0, 2**31)
        for c in range(blk.shape[1]):
            f = ix[np.isfinite(blk[ix, c])]
            if len(f) >= 2:
                blk[f, c] = np.roll(blk[f, c], k_for(len(f), np.random.default_rng(k_seed + c)))
    out[BLOCK] = blk
    return out


def gap_with(tr, te, cols=None):
    cols = cols or SETS["M2"]
    m = fit(cols, tr, train.il14.to_numpy())
    return auc(y14, m.predict_proba(te[cols].to_numpy(float))[:, 1]) - auc_b0, m


res = []
for sd in SEEDS:
    rng = np.random.default_rng(sd)
    res.append(("gaps move (frozen floor)", sd, gap_with(rot_nan_moves(train, rng), rot_nan_moves(test, rng))[0]))
    rng = np.random.default_rng(sd)
    res.append(("gaps fixed", sd, gap_with(rot_gaps_fixed(train, rng), rot_gaps_fixed(test, rng))[0]))
res = pd.DataFrame(res, columns=["rotation", "seed", "gap"])
print(res.groupby("rotation").gap.agg(["mean", "min", "max"]).round(4).to_string())
print(f"real (correctly timed) block: {real_gap:+.4f}")

# %% [markdown]
# ## Level only: each hitter-season's average deviation in place of the daily one
#
# This uses the whole season's rows for every day, so it is **not** a forecast; it only asks whether a hitter-level trait carried by the deviations explains the rotated gain.

# %%


def season_mean(df):
    out = df.copy()
    out[BLOCK] = out.groupby(["batter", "season"])[BLOCK].transform("mean")
    return out


lvl_gap, _ = gap_with(season_mean(train), season_mean(test))
print(f"block replaced by its hitter-season mean: {lvl_gap:+.4f}")
miss_cols = [c + " missing" for c in BLOCK]


def gaps_only(df):
    out = df.copy()
    for c in BLOCK:
        out[c + " missing"] = out[c].isna().astype(float)
    return out


gap_miss, _ = gap_with(gaps_only(train), gaps_only(test), CTX + miss_cols)
print(f"only the gap pattern of the block (missing or not) in place of its values: {gap_miss:+.4f}")

# %% [markdown]
# ## Where the gaps sit

# %%
for name, df in (("train", train), ("test", test)):
    m = df.U2.isna()
    print(f"{name}: U2 missing on {m[df.il14].mean():.1%} of pre-IL rows and {m[~df.il14].mean():.1%} of other rows")
r100 = rot_nan_moves(test, np.random.default_rng(100))
m = r100.U2.isna()
print(f"test after one frozen-floor rotation: missing on {m[test.il14].mean():.1%} of pre-IL rows and {m[~test.il14].mean():.1%} of other rows")

# %% [markdown]
# ## What a rotated model leans on (seed 100)

# %%
rng = np.random.default_rng(100)
tr_r, te_r = rot_nan_moves(train, rng), rot_nan_moves(test, rng)
g100, m100 = gap_with(tr_r, te_r)
pi_r = permutation_importance(m100, te_r[SETS["M2"]].to_numpy(float), y14, scoring="roc_auc", n_repeats=5, random_state=3)
print(f"seed 100 gap {g100:+.4f}")
print(pd.Series(pi_r.importances_mean, index=SETS["M2"]).sort_values(ascending=False).head(12).round(4).to_string())

# %%
fig, ax = plt.subplots(figsize=(9, 5))
order = ["gaps move (frozen floor)", "gaps fixed"]
for i, r in enumerate(order):
    v = res[res.rotation == r].gap
    ax.scatter(np.full(len(v), i), v, color="#8a8a8a", s=50)
ax.scatter([2], [lvl_gap], color="#2471a3", s=80)
ax.scatter([3], [gap_miss], color="#2471a3", s=80)
ax.axhline(real_gap, color="#c0392b", lw=2, label=f"real, correctly timed {real_gap:+.3f}")
ax.axhline(0, color="#cccccc", lw=1)
ax.set_xticks(range(4), ["rotated,\ngaps move", "rotated,\ngaps fixed", "season mean\n(not causal)", "gap pattern\nonly"])
ax.set_ylabel("AUC gain over the rival model")
ax.set_title("Where the shuffled block's gain comes from")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("diagnostics.png", dpi=120)
plt.show()
res.to_csv("rotations.csv", index=False)
