# Dataset 3: MLB Pitcher Arsenal Evolution (2020-2025)

投手の球種構成（Arsenal）の年次変化を追跡するデータセット

## 📊 Dataset Overview

- **対象期間**: 2020-2025シーズン（6シーズン）
- **対象投手**: 各シーズンで100球以上投球した投手
- **データ形式**: Wide format（1行 = 投手×シーズン、球種を横展開）
- **推定行数**: 5,000-10,000行
- **推定カラム数**: 70-100列

## 🎯 主要球種

FF (Fastball), SI (Sinker), FC (Cutter), SL (Slider), CU (Curveball), CH (Changeup), FS (Splitter), KC (Knuckle Curve), FO (Forkball), EP (Eephus), KN (Knuckleball)

## 📈 メトリクス（各球種ごと）

- **usage_pct**: 使用率 (%)
- **avg_speed**: 平均球速 (mph)
- **avg_spin**: 平均回転数 (rpm)
- **whiff_rate**: 空振り率
- **avg_pfx_x**: 平均横変化量 (inch)
- **avg_pfx_z**: 平均縦変化量 (inch)

## 🚀 Usage

### Google Colab（推奨）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yasumorishima/kaggle-datasets/blob/main/dataset3_pitcher_arsenal/pitcher_arsenal_evolution_2020_2025.ipynb)

1. 上のバッジをクリックしてColabで開く
2. 全セルを順番に実行
3. `pitcher_arsenal_evolution_2020_2025.csv` がダウンロードされる

**注意**: データ取得に30分〜1時間かかります

### ローカル実行

```bash
# 必要なパッケージインストール
pip install pybaseball duckdb pandas numpy

# Jupyter Notebook起動
jupyter notebook pitcher_arsenal_evolution_2020_2025.ipynb
```

## 📁 Files

- `pitcher_arsenal_evolution_2020_2025.ipynb` - データ取得ノートブック
- `pitcher_arsenal_evolution_2020_2025.csv` - 出力データ（実行後に生成）
- `README.md` - このファイル

## 🎓 Use Cases

- 投手の球種トレンド分析（例: 菊池雄星のスライダー増加）
- シーズン間の変化検出（例: 怪我前後の球速低下）
- 球団別の戦略分析（例: アストロズのスライダー重視）
- 機械学習の特徴量（投手パフォーマンス予測等）

## 🔗 Related Datasets

- [Japanese MLB Players Statcast (2015-2025)](https://www.kaggle.com/datasets/yasunorim/japan-mlb-pitchers-batters-statcast) - 日本人MLB選手の詳細データ

## 📝 License

データソース: MLB Advanced Media (Statcast)
