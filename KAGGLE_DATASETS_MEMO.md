# Kaggle Datasets 作成メモ

## 基本情報
- **リポジトリ**: https://github.com/yasumorishima/kaggle-datasets
- **ローカル**: `C:\Users\fw_ya\Desktop\Claude_code\kaggle-datasets`
- **Kaggleユーザー**: yasunorim
- **目標**: Bronze 3個 → Datasets Contributor昇格

## kaggle CLI
- **パス**: `C:\Users\fw_ya\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\kaggle.exe`
- **データセット作成**: `kaggle datasets create -p <folder>`
- **データセット更新**: `kaggle datasets version -p <folder> -m "message"`
- **注意**: `PYTHONUTF8=1` が必要な場合あり

## ワークフロー
1. GitHubでgenerate.ipynb管理
2. Google Colabでnotebook実行 → CSV生成
3. CSVをローカルにダウンロード → dataset-metadata.jsonと同じフォルダに配置
4. `kaggle datasets create -p <folder>` で初回アップロード
5. 更新時は `kaggle datasets version -p <folder> -m "update message"`
6. CSVは.gitignoreで除外（大きいため）

## データセット一覧

### 1. Japanese MLB Players Statcast Data (2015-2025)【最優先】
- **フォルダ**: `japanese-mlb-players-statcast/`
- **ステータス**: **Kaggleアップロード済み**（2/8）。https://www.kaggle.com/datasets/yasunorim/japanese-mlb-players-statcast
- **データ量**: 投球119,106件(25投手, 65MB) + 打撃52,261件(9打者, 29MB) + players.csv(33選手)
- **期待upvotes**: 10-30（Bronze-Silver）
- **内容**: 日本人MLB選手全員のStatcast pitch-by-pitchデータ
- **ファイル構成**:
  - `japanese_mlb_pitching.csv` - 投球データ（全投手）
  - `japanese_mlb_batting.csv` - 打撃データ（全打者）
  - `players.csv` - 選手メタデータ
- **game_typeカラム**: S=Spring Training, R=Regular Season, F/D/L/W=Postseason

#### 対象選手（30ユニーク選手、投手25エントリ + 打者9エントリ = 34エントリ）

**投手（24名）**:
| # | 選手 | MLBAM ID | チーム | シーズン | 備考 |
|---|------|----------|--------|----------|------|
| 1 | 大谷翔平 | 660271 | LAA/LAD | 2018-2023 | 打者としても登録。2019TJ、2024-2025DH専念 |
| 2 | ダルビッシュ有 | 506433 | TEX/LAD/CHC/SD | 2017-2025 | 2012-2014はStatcast前 |
| 3 | 山本由伸 | 808967 | LAD | 2024-2025 | |
| 4 | 今永昇太 | 705838 | CHC | 2024-2025 | |
| 5 | 千賀滉大 | 694973 | NYM | 2023-2025 | 2024怪我で登板少 |
| 6 | 菊池雄星 | 579328 | SEA/TOR/HOU/LAA | 2019-2025 | |
| 7 | 松井裕樹 | 680686 | SD | 2024-2025 | リリーバー |
| 8 | 前田健太 | 628317 | LAD/MIN/DET | 2016-2025 | |
| 9 | 菅野智之 | 807185 | BAL | 2025 | MLB1年目 |
| 10 | 佐々木朗希 | 811521 | LAD | 2025 | MLB1年目 |
| 11 | 小笠原慎之介 | 700247⚠️ | WSH | 2025 | MLB1年目、リリーバー |
| 12 | 上沢直之 | 683822 | BOS | 2024 | 1試合のみ |
| 13 | 澤村拓一 | 669022⚠️ | BOS | 2021-2022 | リリーバー。2023NPB復帰 |
| 14 | 山口俊 | 685493⚠️ | TOR | 2020 | 1シーズンのみ |
| 15 | 田中将大 | 547888 | NYY | 2015-2020 | 2021NPB復帰 |
| 16 | 平野佳寿 | 673633⚠️ | ARI | 2018-2019 | リリーバー |
| 17 | 牧田和久 | 628318⚠️ | SD | 2018 | サブマリン投手 |
| 18 | 岩隈久志 | 461325 | SEA | 2015-2017 | 引退 |
| 19 | 上原浩治 | 493157 | BOS/CHC | 2015-2017 | クローザー。2018NPB復帰 |
| 20 | 田澤純一 | 547749 | BOS/MIA/LAA | 2015-2019 | アマチュアから直接MLB |
| 21 | 藤浪晋太郎 | 692006 | OAK | 2023 | MLB1年のみ。日本人最速102.1mph |
| 22 | 有原航平 | 685503 | TEX | 2021 | MLB1年のみ |
| 23 | 藤川球児 | 493117⚠️ | CHC | 2015 | TJ復帰後2試合のみ。最終MLB年 |
| 24 | 和田毅 | 647098⚠️ | CHC | 2015 | 最終MLB年 |
| 25 | 村田透 | 603402⚠️ | CLE | 2015 | 少量登板 |

**打者（9名、うち大谷は投手と重複）**:
| # | 選手 | MLBAM ID | チーム | シーズン | 備考 |
|---|------|----------|--------|----------|------|
| 1 | 大谷翔平 | 660271 | LAA/LAD | 2018-2025 | 二刀流 |
| 2 | 鈴木誠也 | 673548 | CHC | 2022-2025 | |
| 3 | 吉田正尚 | 807799 | BOS | 2023-2025 | |
| 4 | 筒香嘉智 | 660294⚠️ | TB/LAD/PIT | 2020-2022 | |
| 5 | 秋山翔吾 | 673451⚠️ | CIN | 2020-2021 | |
| 6 | 青木宣親 | 493114 | SF/SEA/HOU/TOR/NYM | 2015-2017 | 2018NPB復帰 |
| 7 | 川崎宗則 | 493128⚠️ | TOR/CHC | 2015 | 最終MLB年 |
| 8 | 加藤豪将 | 641741⚠️ | TOR | 2022 | 日系アメリカ人 |
| 9 | ラース・ヌートバー | 663457 | STL | 2021-2025 | 日系アメリカ人、WBC日本代表 |

~~⚠️ = Colab初回実行時にplayerid_lookupで検証必要（13名）~~ → **全件検証済み（2/8）、9件修正済み**

#### 次のステップ
1. ~~Colabでgenerate.ipynb実行~~ → ローカルpybaseballで実行完了（2/8）
2. ~~CSVダウンロード → ローカルフォルダに配置~~ → 完了
3. ~~`kaggle datasets create`~~ → アップロード済み（2/8）
4. **Kaggle上でDescription（英語）を充実させる** ← 次ここ
5. **既存ノートブック（ダルビッシュ・今永・千賀・菊池）からデータセットにリンク**

### 2. MLB Bat Tracking Leaderboard (2024-2025)【低作業量】
- **フォルダ**: `mlb-bat-tracking/`（未作成）
- **ステータス**: 企画段階
- **期待upvotes**: 5-15（Bronze）
- **内容**: 2024年新機能のバットトラッキングデータ（bat_speed, swing_length等）
- **pybaseball関数**: `statcast_batter_bat_tracking()`
- **特徴**: Kaggleにゼロのデータ。最新機能で話題性◎

### 3. MLB Pitcher Arsenal Evolution (2020-2025)【中作業量】
- **フォルダ**: `mlb-pitcher-arsenal-evolution/`（未作成）
- **ステータス**: 企画段階
- **期待upvotes**: 10-20（Bronze-Silver）
- **内容**: 投手の球種別データ経年変化（球速・回転数・変化量・使用率）
- **pybaseball関数**: `statcast_pitcher_pitch_arsenal()` + `statcast_pitcher_arsenal_stats()`
- **特徴**: ML/クラスタリング向き。既存ノートブックと相乗効果大

### 4. MLB Statcast 2024-2025 Full Season【高作業量】
- **フォルダ**: `mlb-statcast-2024-2025/`（未作成）
- **ステータス**: 企画段階
- **期待upvotes**: 20-50+（Silver+）
- **内容**: 2024-2025レギュラーシーズン完全pitch-by-pitchデータ
- **pybaseball関数**: `statcast()`（日付範囲指定で週ごとに取得）
- **特徴**: 既存データセット（2022年止まり、29 upvotes）の後継。最大ポテンシャル
- **注意**: データ量が大きい（~700K行/シーズン）。取得に時間かかる

## Kaggle既存データセット調査結果（2/7時点）

| データセット | Upvotes | Downloads | 最終更新 |
|---|---|---|---|
| MLB Pitch Data 2015-2018 | 261 | 13,139 | 2020年5月 |
| Baseball Databank (1871-2015) | 187 | 22,844 | 2019年 |
| Hitters Baseball Data (1986-87) | 80 | 4,848 | 2020年 |
| MLB Hitting/Pitching All Time | 48 | 2,899 | 2023年4月 |
| 2022 MLB Player Stats | 41 | 3,670 | 2023年7月 |
| MLB Statcast Data | 29 | 1,907 | **2022年1月（4年更新なし）** |
| MLB Postseason 2024 Pitch-by-Pitch | 7 | 331 | 2024年11月 |
| Shohei Ohtani Career Stats | 3 | 116 | 2025年10月 |

**主なギャップ**:
- Statcastデータが2022年止まり（最大のチャンス）
- 日本人選手特化データセットがほぼなし（大谷1件のみ、3 upvotes）
- Bat Tracking（2024年新機能）のデータセットがゼロ
- 投手Arsenal経年変化データセットなし

## ブログ記事（各データセット公開後）
- Zenn/Qiita記事として各データセットの解説記事を書く
- 内容: データセット紹介、使い方、簡単な分析例
- 既存MLB分析ノートブックとの相互リンク
