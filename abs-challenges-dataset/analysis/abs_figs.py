# Charts for the rewritten ABS article (Japanese-manufacturer-deck style: the title is the message,
# one accent color, direct labels, no gridlines). Savant-board numbers come from the final Kaggle
# version (abs_challenges_players.csv); correlations are copied from absfinal_output.txt.
# Run in ~/claude-scratch/absdeep/kv3 ; writes fig/.
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORANGE, GRAY, INK, SUB, LIGHT = "#e4560f", "#b9b8b3", "#0b0b0b", "#52514e", "#dedcd6"
plt.rcParams["font.family"] = "Noto Sans CJK JP"
os.makedirs("fig", exist_ok=True)
P = pd.read_csv("abs_challenges_players.csv", low_memory=False)


def base(ax, left=False):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(left and s == "left")
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(colors=SUB, length=0)


def title(fig, text, note=None):
    fig.text(0.04, 0.95, text, fontsize=19, color=INK, ha="left", va="top", weight="bold")
    if note:
        fig.text(0.04, 0.875, note, fontsize=13, color=SUB, ha="left", va="top")


def board(level, year, gt="R"):
    return P[(P.level == level) & (P.year == year) & (P.game_type == gt)]


# 1. share won by challenger, three boards
labs = [("MLB", 2026, "MLB 2026"), ("AAA", 2026, "3A 2026"), ("AAA", 2025, "3A 2025")]
who = [("catcher", "捕手", ORANGE), ("batter", "打者", SUB), ("pitcher", "投手", GRAY)]
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
fig.subplots_adjust(top=0.74, left=0.14, right=0.95, bottom=0.10)
for j, (lvl, yr, name) in enumerate(labs):
    d0 = board(lvl, yr)
    for i, (ct, jp, c) in enumerate(who):
        d = d0[d0.challenge_type == ct]
        n, k = d.n_challenges.sum(), d.n_overturns.sum()
        if n < 100:
            continue
        y = (2 - j) * 1.0 + (1 - i) * 0.26
        ax.barh(y, k / n, height=0.22, color=c)
        ax.text(k / n + 0.008, y, f"{jp} {k / n * 100:.0f}%（{n:,} 回）", va="center", fontsize=12.5,
                color=ORANGE if c == ORANGE else SUB, weight="bold" if c == ORANGE else "normal")
ax.set_yticks([2, 1, 0]); ax.set_yticklabels([l[2] for l in labs], fontsize=14)
ax.axvline(0.5, color=LIGHT, lw=1)
ax.set_xlim(0, 0.9); ax.set_xticks([]); base(ax); ax.spines["bottom"].set_visible(False)
title(fig, "どのレベルでも、捕手のチャレンジが一番通る", "公式戦のチャレンジのうち判定が覆った割合（3A 2025 の投手は 5 回だけなので省略）")
fig.savefig("fig/abs_ja_1_share.png"); plt.close(fig)

# 2. catchers with >= 60 challenges, MLB 2026: share won
c = board("MLB", 2026)
c = c[(c.challenge_type == "catcher") & (c.n_challenges >= 60)].copy()
c["share"] = c.n_overturns / c.n_challenges
c = c.sort_values("share")
fig, ax = plt.subplots(figsize=(10, 7.4), dpi=150)
fig.subplots_adjust(top=0.80, left=0.26, right=0.95, bottom=0.06)
avg = board("MLB", 2026).query("challenge_type == 'catcher'")
avg = avg.n_overturns.sum() / avg.n_challenges.sum()
hi = set(c.nlargest(2, "share").player_name) | set(c.nsmallest(1, "share").player_name)
for y, (_, r) in enumerate(c.iterrows()):
    col = ORANGE if r.player_name in hi else GRAY
    ax.plot([avg, r.share], [y, y], color=LIGHT, lw=1.5, zorder=1)
    ax.scatter(r.share, y, s=46, color=col, zorder=2)
    if r.player_name in hi:
        dy = -0.9 if r.player_name == "Salvador Perez" else 0
        ax.text(r.share + (0.012 if r.share > avg else -0.012), y + dy, f"{r.player_name}  {r.n_overturns:.0f}/{r.n_challenges:.0f}",
                va="center", ha="left" if r.share > avg else "right", fontsize=12.5, color=ORANGE, weight="bold")
ax.axvline(avg, color=SUB, lw=1, ls=(0, (4, 3)))
ax.text(avg, len(c) + 0.2, f"捕手全体 {avg * 100:.1f}%", fontsize=12, color=SUB, ha="center")
ax.set_yticks([]); ax.set_xlim(0.30, 1.02)
ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]); ax.set_xticklabels([f"{v:.0%}" for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)], fontsize=12)
base(ax)
title(fig, "同じ捕手でも、通す割合は 47% から 81% まで開く", f"MLB 2026 公式戦・60 回以上チャレンジした捕手 {len(c)} 人（1 点が 1 人）")
fig.savefig("fig/abs_ja_2_catchers.png"); plt.close(fig)

# 3. chase vs share won (batters >= 10 challenges)
s = pd.read_csv("scatter_chase.csv")
fig, ax = plt.subplots(figsize=(10, 6.2), dpi=150)
fig.subplots_adjust(top=0.78, left=0.1, right=0.95, bottom=0.14)
ax.scatter(s.sc_bat_chase_percent, s.share * 100, s=np.clip(s.n_challenges, 10, 60) * 1.6, color=GRAY, alpha=0.6, lw=0)
for lo, hi_, med, sh in ((0, 27.7, 24.8, 51.3), (27.7, 33.4, 30.4, 48.0), (33.4, 60, 36.9, 45.7)):
    pass
tiers = [(24.8, 51.3), (30.4, 48.0), (36.9, 45.7)]
ax.plot([t[0] for t in tiers], [t[1] for t in tiers], color=ORANGE, lw=3.5, marker="o", ms=10, zorder=3)
for x, y in tiers:
    ax.text(x, y + 3.2, f"{y:.0f}%", color=ORANGE, fontsize=14, ha="center", weight="bold")
for name in ("Ceddanne Rafaela", "Taylor Ward"):
    r = s[s.player_name == name].iloc[0]
    ax.annotate(name, (r.sc_bat_chase_percent, r.share * 100), xytext=(8, -14) if name == "Taylor Ward" else (-60, -24), textcoords="offset points", fontsize=11.5, color=SUB)
ax.set_xlabel("ボール球スイング率（ゾーン外の球を振った割合、%）", fontsize=13, color=SUB)
ax.set_ylabel("チャレンジが通った割合（%）", fontsize=13, color=SUB)
ax.set_ylim(0, 100); ax.tick_params(labelsize=12)
base(ax, left=True); ax.spines["left"].set_color(GRAY)
title(fig, "ボール球を振らない打者ほど、チャレンジが通る", "MLB 2026・10 回以上チャレンジした打者 182 人（点の大きさ＝回数）／橙＝スイング率で 3 等分した 465 人の合計")
fig.savefig("fig/abs_ja_3_chase.png"); plt.close(fig)

# 4. framing proxy: vs official framing runs, and vs challenge skill
f1 = pd.read_csv("scatter_framing.csv")
f2 = pd.read_csv("scatter_framing_vs_chal.csv")
fig, axs = plt.subplots(1, 2, figsize=(11, 6), dpi=150)
fig.subplots_adjust(top=0.76, left=0.08, right=0.97, bottom=0.14, wspace=0.28)
axs[0].scatter(f1.sc_c_oz_called_strike_percent, f1.sc_c_framing_runs, s=34, color=ORANGE, alpha=0.75, lw=0)
axs[0].set_xlabel("ゾーン外の見逃しがストライクになった割合（%）", fontsize=12, color=SUB)
axs[0].set_ylabel("公式のフレーミング（点）", fontsize=12, color=SUB)
axs[0].text(0.02, 0.97, "公式のフレーミングとは\n順位相関 0.83（56 人）", transform=axs[0].transAxes, fontsize=13, color=ORANGE, va="top", weight="bold")
axs[1].scatter(f2.sc_c_oz_called_strike_percent, f2.overturns_vs_exp, s=34, color=GRAY, alpha=0.8, lw=0)
axs[1].axhline(0, color=LIGHT, lw=1)
axs[1].set_xlabel("ゾーン外の見逃しがストライクになった割合（%）", fontsize=12, color=SUB)
axs[1].set_ylabel("チャレンジの上手さ（期待との差、回）", fontsize=12, color=SUB)
axs[1].text(0.02, 0.97, "チャレンジの上手さとは\n順位相関 0.03（91 人）", transform=axs[1].transAxes, fontsize=13, color=SUB, va="top", weight="bold")
for ax in axs:
    ax.tick_params(labelsize=11); base(ax, left=True); ax.spines["left"].set_color(GRAY)
title(fig, "ストライクに見せる技術と、判定を覆す判断は別もの", "MLB 2026 公式戦の捕手・横軸は Statcast から自分で出した値（3A でも計算できる）")
fig.savefig("fig/abs_ja_4_framing.png"); plt.close(fig)

# 5. carry-over AAA 2025 -> MLB 2026 (batters >= 5 challenges both)
g = pd.read_csv("scatter_carry.csv")
fig, axs = plt.subplots(1, 2, figsize=(11, 6), dpi=150)
fig.subplots_adjust(top=0.76, left=0.08, right=0.97, bottom=0.14, wspace=0.25)
axs[0].scatter(g.rate_challenges_aaa2025 * 100, g.rate_challenges_mlb2026 * 100, s=36, color=ORANGE, alpha=0.75, lw=0)
axs[0].set_xlabel("3A 2025 のチャレンジ率（%）", fontsize=12, color=SUB); axs[0].set_ylabel("MLB 2026 のチャレンジ率（%）", fontsize=12, color=SUB)
axs[0].text(0.02, 0.97, "どれだけ挑むか\n順位相関 0.37", transform=axs[0].transAxes, fontsize=13, color=ORANGE, va="top", weight="bold")
axs[1].scatter(g.overturns_vs_exp_aaa2025, g.overturns_vs_exp_mlb2026, s=36, color=GRAY, alpha=0.8, lw=0)
axs[1].axhline(0, color=LIGHT, lw=1); axs[1].axvline(0, color=LIGHT, lw=1)
axs[1].set_xlabel("3A 2025 の上手さ（期待との差、回）", fontsize=12, color=SUB); axs[1].set_ylabel("MLB 2026 の上手さ（期待との差、回）", fontsize=12, color=SUB)
axs[1].text(0.02, 0.97, "期待との差\n順位相関 0.05", transform=axs[1].transAxes, fontsize=13, color=SUB, va="top", weight="bold")
for ax in axs:
    ax.tick_params(labelsize=11); base(ax, left=True); ax.spines["left"].set_color(GRAY)
title(fig, "3A から持ち込まれるのは、挑む回数の癖", "3A 2025 と MLB 2026 の両方で 5 回以上チャレンジした打者 63 人")
fig.savefig("fig/abs_ja_5_carry.png"); plt.close(fig)
print("ok")
