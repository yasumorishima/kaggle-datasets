# Pitch-level charts for the ABS article. Reads challenges_2026.parquet written by absdeep.py.
# Also one chart from the Savant boards (expected vs actual share won). Run in the folder holding the parquet;
# pass the Kaggle csv path as argv[1].
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORANGE, GRAY, INK, SUB, LIGHT = "#e4560f", "#b9b8b3", "#0b0b0b", "#52514e", "#dedcd6"
plt.rcParams["font.family"] = "Noto Sans CJK JP"
d = pd.read_parquet("challenges_2026.parquet")
P = pd.read_csv(sys.argv[1], low_memory=False)
import os
os.makedirs("fig", exist_ok=True)


def base(ax, left=False):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(left and s == "left")
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(colors=SUB, length=0)


def title(fig, text, note=None):
    fig.text(0.04, 0.95, text, fontsize=19, color=INK, ha="left", va="top", weight="bold")
    if note:
        fig.text(0.04, 0.875, note, fontsize=13, color=SUB, ha="left", va="top")


# B. share won by strikes before the pitch
t = d.pivot_table(index="strikes", columns="role", values="y", aggfunc="mean")
n = d.pivot_table(index="strikes", columns="role", values="y", aggfunc="size")
print(t.round(3)); print(n)
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
fig.subplots_adjust(top=0.74, left=0.08, right=0.82, bottom=0.14)
for role, jp, c in (("catcher", "捕手", ORANGE), ("batter", "打者", SUB)):
    v = t[role].values
    ax.plot(range(3), v, color=c, lw=3.5, marker="o", ms=9)
    for k in range(3):
        ax.text(k, v[k] + (0.025 if c == ORANGE else -0.05), f"{v[k] * 100:.0f}%", ha="center", fontsize=13, color=c, weight="bold" if c == ORANGE else "normal")
    ax.text(2.12, v[-1], jp, color=c, fontsize=15, va="center", weight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(["0 ストライク", "1 ストライク", "2 ストライク"], fontsize=14)
ax.set_ylim(0.3, 0.72); ax.set_yticks([]); base(ax)
title(fig, "2 ストライクになると、打者のチャレンジは通りにくくなる", "MLB 2026 公式戦・チャレンジした球の前のストライク数別に、判定が覆った割合")
fig.savefig("fig/abs_ja_7_strikes.png"); plt.close(fig)

# C. batters: strike-three calls vs other calls (share of challenges, share won)
b = d[d.role == "batter"]
k3 = b[b.stands_ends]; other = b[~b.stands_ends]
print("batter strike-three challenges", len(k3), round(len(k3) / len(b), 3), round(k3.y.mean(), 3), "other", round(other.y.mean(), 3))
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
fig.subplots_adjust(top=0.70, left=0.40, right=0.95, bottom=0.08)
items = [("見逃し三振の判定への異議\n（全体の %d%%）" % round(len(k3) / len(b) * 100), k3.y.mean(), ORANGE),
         ("それ以外のストライク判定への異議\n（全体の %d%%）" % round(len(other) / len(b) * 100), other.y.mean(), GRAY)]
for i, (lab, v, c) in enumerate(items):
    ax.barh(1 - i, v, height=0.5, color=c)
    ax.text(v + 0.01, 1 - i, f"{v * 100:.0f}%", va="center", fontsize=16, color=ORANGE if c == ORANGE else SUB, weight="bold")
ax.set_yticks([1, 0]); ax.set_yticklabels([x[0] for x in items], fontsize=13.5)
ax.set_xlim(0, 0.75); ax.set_xticks([]); base(ax); ax.spines["bottom"].set_visible(False)
title(fig, "打者のチャレンジの 3 分の 1 は、見逃し三振への異議", "MLB 2026 公式戦・打者のチャレンジのうち判定が覆った割合")
fig.savefig("fig/abs_ja_8_strike3.png"); plt.close(fig)

# D. by inning
t2 = d.pivot_table(index="late", columns="role", values="y", aggfunc="mean").loc[["1-6", "7-8", "9th+"]]
print(t2.round(3)); print(d.pivot_table(index="late", columns="role", values="y", aggfunc="size"))
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
fig.subplots_adjust(top=0.74, left=0.08, right=0.82, bottom=0.14)
for role, jp, c in (("catcher", "捕手", ORANGE), ("batter", "打者", SUB)):
    v = t2[role].values
    ax.plot(range(3), v, color=c, lw=3.5, marker="o", ms=9)
    for k in range(3):
        ax.text(k, v[k] + (0.025 if c == ORANGE else -0.05), f"{v[k] * 100:.0f}%", ha="center", fontsize=13, color=c, weight="bold" if c == ORANGE else "normal")
    ax.text(2.12, v[-1], jp, color=c, fontsize=15, va="center", weight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(["1〜6 回", "7〜8 回", "9 回以降"], fontsize=14)
ax.set_ylim(0.3, 0.72); ax.set_yticks([]); base(ax)
title(fig, "試合の終盤ほど、チャレンジは通りにくい", "MLB 2026 公式戦・イニング別に、判定が覆った割合")
fig.savefig("fig/abs_ja_9_inning.png"); plt.close(fig)
print("ok")
