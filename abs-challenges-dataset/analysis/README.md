# Analysis behind the ABS challenges article

Code and outputs for the article that uses this dataset together with pitch-level StatsAPI data:
[Qiita (Japanese)](https://qiita.com/ussu_ussu_ussu/items/57c5760b3f9917b3883f) /
[DEV.to (English)](https://dev.to/yasumorishima/abs-challenges-from-triple-a-to-mlb-catchers-win-most-and-batters-who-dont-chase-win-more-499a).

## Pitch-level challenges (MLB StatsAPI, keyless)

| File | What it does |
|---|---|
| `collect.sh` | Walks the schedule day by day and extracts every ABS pitch challenge from each final game's `feed/live` with `extract2.jq` (about 2,430 games for the 2026 regular season) |
| `extract2.jq` | One row per challenge. **A review is stored in two places**: on the pitch event when the plate appearance goes on, and on the **play** (`allPlays[k].reviewDetails`) when the final call ends it (strikeout / walk). Reading only the pitch misses 25% of challenges |
| `prepare.py` | `load()`: de-duplicates, derives the original call from which side challenged (the pitch's `call` is the call *after* review), and the signed distance from the ball's surface to the zone |
| `absdeep.py` | Breakdown by challenger, count, inning and distance. **Gate:** totals by challenger must match the Savant board sums (10,557 in 2026: catcher 5,612 vs 5,613, batter 4,766 vs 4,765, pitcher 179 vs 179). Output: `absdeep_output.txt` |
| `abs_fix.py` | Which pitches each role chooses (call looks clearly wrong / within an inch / looks right), the swapped-mix calculation, chase-rate thirds, and catchers' challenges vs Savant's expected count. Output: `abs_fix_output.txt` |

## Board-level numbers (this dataset)

| File | What it does |
|---|---|
| `absfinal.py` | Share won by board and challenger, catcher and batter lists, chase rate, framing proxy, Triple-A 2025 → MLB 2026 carry-over, age. Output: `absfinal_output.txt` |

**Savant's expected values** (`exp_chal`, `exp_chal_gained`) are how many times an average player would challenge, and win, given the same opportunities as this player. They are not computed for the pitches the player chose, so `exp_chal_gained / n_challenges` is not an expected win rate for the player's own challenges.

## Charts

`abs_figs.py` (charts 1–5, from the dataset), `abs_pitch_figs.py` (7–9) and `abs_fig6.py` (6) draw the Japanese charts; `make_en.py` writes English copies of those scripts by replacing the strings.

## Running

```bash
bash collect.sh 2026-03-25 2026-09-28        # writes challenges_all.jsonl (resumable)
python absdeep.py challenges_all.jsonl       # writes challenges_2026.parquet
mkdir -p kv3 && cp ../path/to/abs_challenges_players.csv ../path/to/abs_aaa2025_to_mlb2026.csv kv3/
python abs_fix.py
(cd kv3 && python ../absfinal.py && python ../abs_figs.py)
```

The distance to the zone is computed from StatsAPI coordinates (front of the plate) and the zone top and bottom in StatsAPI; ABS judges over the middle of the plate, so near the edge the two differ. Pitches an inch or more from the edge were called as they look 97.9% to 99.7% of the time.
