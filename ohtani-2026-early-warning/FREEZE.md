# Frozen before looking at any 2026 (or 2023) series below month level — 2026-10-09

Question: could Ohtani's 2026 decline (and the 2023 injury) have been seen in his own
process data before the injured-list placement, using only data available on each day,
and how often does the same rule raise an alarm in seasons where nothing happened?

Already seen before freezing (disclosed): 2026 monthly table (Aug wOBA .355, EV 91.9,
K% .273; Sep 36 PA); IL 2026-09-08 retro (right biceps inflammation); 2023 UCL tear
(left start 2023-08-23). No daily/rolling series of any season has been looked at.

Signals (each on a trailing window of the player's own events strictly before day t):
  B1 EV90     90th percentile launch_speed of last 40 batted balls (type X)
  B2 HardHit  share of last 40 batted balls with launch_speed >= 95
  B3 BatSpeed mean bat_speed of last 100 swings with bat_speed >= 50 (2024+ only)
  B4 Whiff    misses / swings over last 150 swings
  R  wOBA     sum woba_value / sum woba_denom over last 100 PA   (the RESULT, for comparison)
  P1 FFvelo   mean release_speed of the four-seamers in his last start (pitching file)

Baseline: the same windowed metric computed on every day of the player's PREVIOUS
regular season (>= 300 PA; for P1 the previous season with >= 10 starts).
Alarm: metric on day t is worse than the previous season's 5th percentile
(B4: above the 95th) on 3 consecutive game days. Nothing is computed until the
window is full. No parameter here is changed after the results are seen.

Reported for each event season (Ohtani 2023, 2026): first alarm date per signal vs
the IL/injury date and vs the first R alarm (the results slump everyone sees).
Reported for reference seasons (every Japanese hitter-season in the dataset with a
>= 300 PA previous season and no injured-list placement that year): alarms per 100
team games = false-alarm rate. IL placements from MLB StatsAPI transactions.
