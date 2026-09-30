# Figure 6 (replaces the Savant expected-vs-actual chart, which misread Savant's expected values):
# what kind of pitch each role challenges, from pitch-level data. Run in ~/claude-scratch/absdeep.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORANGE, GRAY, INK, SUB, LIGHT = "#e4560f", "#b9b8b3", "#0b0b0b", "#52514e", "#dedcd6"
plt.rcParams["font.family"] = "Noto Sans CJK JP"
d = pd.read_parquet("challenges_2026.parquet")
d["cl"] = np.select([d.wrong_side_in >= 1, d.wrong_side_in < -1], ["wrong", "right"], "near")
fig, ax = plt.subplots(figsize=(10, 5.4), dpi=150)
fig.subplots_adjust(top=0.66, left=0.10, right=0.96, bottom=0.08)
cols = [("wrong", "判定が明らかに違って見える球", ORANGE), ("near", "境目から 1 インチ以内", LIGHT), ("right", "判定どおりに見える球", GRAY)]
for i, (role, jp) in enumerate((("catcher", "捕手"), ("batter", "打者"))):
    y = 1 - i
    s = d[d.role == role].cl.value_counts(normalize=True)
    left = 0
    for k, lab, c in cols:
        v = s.get(k, 0)
        ax.barh(y, v, left=left, height=0.55, color=c)
        ax.text(left + v / 2, y, f"{v * 100:.0f}%", ha="center", va="center", fontsize=15,
                color="white" if c == ORANGE else INK, weight="bold" if c == ORANGE else "normal")
        left += v
    print(role, s.round(3).to_dict())
ax.set_yticks([1, 0]); ax.set_yticklabels(["捕手", "打者"], fontsize=15)
ax.set_xlim(0, 1); ax.set_xticks([]); ax.tick_params(length=0, colors=SUB)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.text(0.04, 0.80, "■ 判定が明らかに違って見える球", fontsize=12.5, color=ORANGE, ha="left", va="top")
fig.text(0.40, 0.80, "■", fontsize=12.5, color="#c9c7c1", ha="left", va="top")
fig.text(0.42, 0.80, "境目から 1 インチ以内", fontsize=12.5, color=SUB, ha="left", va="top")
fig.text(0.66, 0.80, "■", fontsize=12.5, color=GRAY, ha="left", va="top")
fig.text(0.68, 0.80, "判定どおりに見える球", fontsize=12.5, color=SUB, ha="left", va="top")
fig.text(0.04, 0.95, "捕手は、判定が明らかに違って見える球を選んでいる", fontsize=19, color=INK, ha="left", va="top", weight="bold")
fig.text(0.04, 0.875, "MLB 2026 公式戦・チャレンジした球の内訳（ゾーンの境目から 1 インチ以上離れているかで分けた）", fontsize=13, color=SUB, ha="left", va="top")
fig.savefig("fig/abs_ja_6_clarity.png")
print("ok")
