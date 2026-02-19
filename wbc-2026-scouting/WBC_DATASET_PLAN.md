# WBC 2026 Kaggle Dataset 計画メモ

## 全体の流れ
1. **Kaggle Dataset** → 2. **Kaggle Notebook** → 3. **WBCアプリリンク貼付** → 4. **ブログ記事**

## Dataset構成（6ファイル）

| CSV | 内容 | ソース |
|---|---|---|
| `rosters.csv` | 全306人クリーンロースター（name, country, pos, team, role等） | `wbc2026_rosters.csv` パース |
| `batter_summary.csv` | 打者集計（PA/AVG/OBP/SLG/OPS/HR/K%/BB%/xwOBA/EV/LA） | 各国statcastから計算 |
| `pitcher_summary.csv` | 投手集計（投球数/被打率/被長打率/K%/BB%/xwOBA/球速/回転数） | 各国pitchers_statcastから計算 |
| `statcast_batters.csv` | 全打者pitch-level生データ（country列追加） | 18国分結合 |
| `statcast_pitchers.csv` | 全投手pitch-level生データ（country列追加） | 14国分結合 |
| `stadiums.csv` | 球場座標データ | そのまま |

## 生成スクリプト
- `generate.py` 作成済み（`C:\...\kaggle-datasets\wbc-2026-scouting\generate.py`）
- wbc-scoutingの`data/`ディレクトリを読んでCSVを出力
- 集計ロジックはapp_dr.pyの`batting_stats()`/`pitching_stats()`と同一

## 残タスク
- [ ] `generate.py` を実行してCSV生成
- [ ] `dataset-metadata.json` 作成（Kaggle API用）
- [ ] 生成されたCSVの確認・サニティチェック
- [ ] GitHub push（kaggle-datasetsリポ）
- [ ] Kaggle upload（`kaggle datasets create -p wbc-2026-scouting`）
- [ ] Kaggle Notebook作成（チーム比較分析 + Streamlitアプリリンク）
- [ ] ブログ記事（note or Zenn）

## 注意事項
- テキスト・説明文は**断定を避け、控えめ表現**で書く（「〜と考えられます」「〜の可能性があります」）
- 選手への言及も控えめに（重要方針11）
- データは生データを大量に含めて応用が効くようにする
- CSVはGitHubにも保存（なるべくGitHub管理）

## combine_statcast の注意点
- `*_statcast.csv`グロブは`*_pitchers_statcast.csv`も拾ってしまうので、打者用では`_pitchers_`を含まないもののみ対象にする必要あり（要修正）
- Venezuela: `venezuela_statcast.csv`/`venezuela_pitchers_statcast.csv` あり（players_*.pyファイルなし、30アプリ対象外だがCSVはある）
