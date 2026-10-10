# Can a hitter's own anomalies warn of the injured list? Frozen 2026-10-10, before any feature below is compared with the injured list

FREEZE_population.md tested one hand-picked signal (a 2.0 mph bat-speed drop) and five level flags with fixed
lines, and none of them preceded the injured list better than chance. That design never asked the broader
question: **does anything about a hitter move away from his own normal before he goes on the injured list?**
This file fixes that test: every swing, batted-ball, plate-discipline and playing-time parameter in the data,
each measured against the same hitter's own past, scored without labels (anomaly detection) and with labels
(machine learning), trained on 2024-2025 and scored once on 2026.

What has already been seen, stated before the run: the injured-list labels of 2024-2026 were read through
the H1/H2 signals only (bat speed, EV90, hard-hit, whiff, wOBA level flags; all null). None of the other
parameters below has been compared with the injured list. The fill rates of the swing columns per season
(all near 1.0 in 2024, 2025 and 2026 on processed sample days) were checked before this file; no outcome
was read for it. Nothing below is changed after the run; any change goes in AMENDMENTS with its date and
reason, before the run.

## Data
- Pitch-by-pitch Statcast, regular season 2024-2026, from the Kaggle dataset "MLB Statcast + Bat Tracking
  (2024-2026)" (yasunorim/mlb-statcast-bat-tracking-2024-2025), the same version as the population notebook.
- Injured-list placements and activations: MLB StatsAPI transactions, parsed exactly as in the population
  notebook (effective date, stints less than 10 days apart merged, body region from the injury sentence).
- Primary position and birth date: StatsAPI /people. Batters whose primary position is P are excluded.
- Shohei Ohtani 2026 is excluded from training and scoring (he motivated the project); his 2026 scores are
  shown as a case only.

## Rows and outcome
- One row per batter and game day t (a day on which he saw at least one pitch). Every feature on day t uses
  only pitches from earlier days (strictly before t).
- IL14(t) = an IL placement with effective date in [t, t + 14 days] (primary). IL30 is reported alongside.
- Rows whose 14-day window passes the last regular-season date of their season are dropped (censoring).
- Training: 2024 and 2025. Test: 2026. Test rows are never used for fitting, scaling, thresholds or choices.

## Parameters (P = 20), each a rolling value over the batter's most recent events before t, same season
Swing, last 100 competitive swings (bat_speed >= 50): mean bat_speed, swing_length, attack_angle,
attack_direction, swing_path_tilt, intercept_ball_minus_batter_pos_y_inches (contact depth),
intercept_ball_minus_batter_pos_x_inches.
Batted ball, last 40 balls in play with launch_speed: 90th percentile of launch_speed, mean launch_angle,
sweet-spot share (launch_angle 8 to 32), pull share (spray angle from hc_x, hc_y toward the batter's pull
side, angle beyond 15 degrees), mean estimated_woba_using_speedangle.
Plate discipline: swing rate over the last 150 pitches seen; chase rate = swings / pitches over the last 100
pitches in zones 11-14; whiff rate over the last 150 swings.
Plate appearances, last 100 PA: K share, BB share, wOBA.
Playing time, last 10 game days of the batter: mean days between consecutive game days (including the gap
to t), mean plate appearances per game day; plus the two playing-time items below.
Each of the 18 swing, batted-ball, discipline and PA parameters is computed twice: with the long windows
above (H2's frozen sizes) and with short windows (30 competitive swings, 15 balls in play, 50 pitches seen,
30 chase-zone pitches, 50 swings for whiff, 30 PA), because a 14-day horizon needs the last two weeks to
dominate the value. A value is missing until its window is full, and a window value needs at least half of its
events with the column filled. Playing time adds: days since the previous game day (raw gap to t), and
plate appearances in the previous game minus his median per game this season before it (at least 3 games).

## Own-normal deviation (z)
For parameter p on day t: centre = median of the batter's values of p on his game days of the current
season up to t - 30 days (calendar) when there are at least 20 such days with a value, otherwise the median
over the whole previous season (at least 20 days). Scale = 1.4826 x MAD over his whole previous season when
it has at least 60 days with a value; otherwise the league-typical scale for that parameter and window (the
median over hitters with at least 60 values of the whole-season MAD in 2025, the last training season;
label-free). The data start in
2024, so every 2024 row and every rookie or call-up uses the league scale. z = (x_t - centre) / scale;
missing if neither baseline exists or the scale is 0; clipped to [-10, 10]. (A pooled two-season centre would
make an off-season swing change look anomalous all year, so the centre is in-season. The scale comes from a
full season because a short in-season baseline of overlapping windows has a too-small MAD, which inflated z
early in the season in a code test.)

## Unsupervised anomaly scores (no labels used)
- U1 = max |z| over the 18 short-window swing, batted-ball, discipline and PA parameters with a value (at
  least 8 required, else missing). Playing time is left out of U1-U4 (it is in the rival model B0).
- U2 = mean of z^2 over the same parameters (primary unsupervised score).
- U3 = Isolation Forest (scikit-learn, n_estimators 300, random_state 0, other settings default) fitted on
  the 2024-2025 short-window z vectors (missing z set to 0, meaning "at his normal"), scored on 2026 (higher
  = more anomalous). U3 and U4 are fitted and scored only on rows where U1/U2 are defined.
- U4 (reported) = robust Mahalanobis distance of the same vectors (MinCovDet, random_state 4, fitted on a
  50,000-row sample of 2024-2025), since 7 of the 18 parameters are one correlated swing cluster.
- Every U score is scored as its rank among all hitters with a value on the same day, (midrank - 0.5) /
  count, so the mean is 0.5 on every day (every value is known that morning). The raw level drifts with the point in the season (the median U2 fell from about 1.4-1.8 in
  spring to 1.25-1.35 in September in a label-free code test), and IL placements are not spread evenly over
  the season, so a pooled AUC of the raw level would partly measure the calendar.

## Supervised models (same learner for all)
HistGradientBoostingClassifier(learning_rate 0.05, max_iter 300, max_leaf_nodes 31, min_samples_leaf 200,
l2_regularization 1.0, early_stopping False, random_state 0). No tuning; these values are fixed here.
- B0 (rival: what is known without swing or ball data): age on t, primary position group (C / IF / OF / DH;
  today's primary position from StatsAPI, applied to every season), number of his IL placements since
  2022-01-01 before t, days since his latest activation on or before t (capped at 365; 365 if none; IL moves
  of 2022-2023 are fetched for this history only), his MLB games played in the previous season (StatsAPI),
  game days so far this season, day of season, and the playing-time parameters (raw, and z for the two
  10-game ones).
- M2 (anomaly model, primary): B0 + the z of the 18 parameters in both windows (36) + U1, U2. No raw
  levels, so the gain cannot come from who the batter is.
- M1 (secondary): M2 + the raw values of the 18 parameters in both windows.

## Primary tests (2026 test rows only; 95% intervals by bootstrap over batters, 2,000 resamples, seed 2;
two primaries, each read at 97.5% (Bonferroni))
- P1 (unsupervised): AUC of U2's daily rank for IL14, on the test rows where U2 is defined (the share is
  reported). Supported only if the 97.5% interval lower bound is above 0.5 AND the real AUC is above all of
  200 rotations (seeds 1000-1199) in which each 2026 hitter-season's defined daily-rank values are rotated
  in time by a random whole number of positions between 30 and (count - 30) (half the count when fewer than
  61); only defined values move, so every rotation is scored on the same rows and labels as the real AUC. The
  rotation keeps every hitter's level and spread, so a hitter who is always near the top (short baselines,
  platoon windows) cannot pass P1 by level alone.
- P2 (anomaly ML adds to known risk): AUC(M2) - AUC(B0) for IL14, paired over the same resampled batters.
  Supported if the 97.5% interval lower bound is above 0.

## Reported whatever the result
- AUCs of U1, U3, U4, B0, M1, M2 for IL14 and IL30 (IL30 scored with the same IL14-trained models), with
  intervals; AUC(M1) - AUC(B0).
- Upper-body and lower-body placements separately (same region rule as FREEZE_population.md): AUC of U2, B0
  and M2 using only that region's placements as positives (rows whose next placement within 14 days is of
  another region are dropped).
- Sensitivity: rows within 30 days after an activation removed from the 2026 scoring, models unchanged
  (returning from the IL can itself look anomalous).
- Sensitivity: M2 without the playing-time features in either model (does the gain come from swings and
  balls, or from a hurt player being rested before the IL?).
- An alarm rule fixed now: alarm on day t for the top 1% of hitters with a U2 that day by U2, rounded down,
  at least one.
  On 2026: alarms per batter-season, share of alarms followed by IL14, share of IL placements with an alarm
  in the 14 days before.
- Permutation importance of M2 on 2026 (AUC drop, 5 repeats, seed 3), top 10.
- The alarm days are written to alarms_2026.csv with each hitter's three largest deviations; the realised
  2026 alarm rate is reported.
- AUC(M2) and AUC(B0) on the rows where U2 is defined (P1's population); the raw (unranked) U2 AUC next to
  the ranked one.
- Label-free checks: Spearman correlation of a hitter's mean daily rank with his row count, with having a
  previous season and with the share of rows on his own scale; the share of rows per baseline kind; the sd of
  z by season; hitters per day.
- Event level: per 2026 hitter, each placement's pre-IL rows form one unit and his other rows are cut into
  14-day blocks; unit score = max and mean of U2 and of the M2 probability (the max favours larger units,
  so unit sizes are reported); AUC over units. The number of 2026 placements
  with no row in the 14 days before although he had played earlier that season (hurt while not playing) is
  reported, and placements before his first 2026 game separately. Alarm days per hitter-season are reported.
- A time-shift floor for P2 (part of the P2 rule): 20 times (seeds 100-119), keep the true labels and the
  real B0, and rotate only the anomaly block (the 36 z and U1, U2) within each batter-season, in training and
  test alike, by a random whole number of game days between 30 and (days in that batter-season - 30)
  (half the days when there are fewer than 61); refit M2 and take AUC(M2) - AUC(B0) on 2026. If the real gap is
  not above all 20, P2 is NOT SUPPORTED even if its interval clears 0 (the gain would then come from the
  block's structure, not from when the deviations happen). The rotation moves the block's missing values
  with it; the rows and labels scored stay the same.

## Sizing (label-free, before the run)
About 190 placements of non-pitchers are expected in 2026 (564 over three seasons in the population study).
Hanley-McNeil with ~190 independent positives and many negatives gives SE(AUC) ~ 0.021 at AUC 0.55, so at
97.5% two-sided P1 can detect AUC ~ 0.55 and above; for P2 a paired gap of roughly 0.03-0.04 (day-level rows
are clustered by hitter, which widens this; the bootstrap over hitters accounts for it). A NOT SUPPORTED
below these sizes means "smaller than this study can see", not "zero".

## Known limits, stated before the result
- The injured list is administrative and many injuries are acute; nothing in swing data can precede a
  pulled hamstring on one play. Upper-body results are the closest test of a slowly worsening condition.
- A batter playing hurt may be benched before the IL; the playing-time features see that, so it is kept in
  B0 and the no-playing-time sensitivity is reported.
- The 14-day horizon (instead of 30) was chosen after the 30-day results of FREEZE_population.md were seen
  (null); IL30 is reported for that reason.
- Switch hitters: pull share is side-relative by construction; attack direction and contact-lateral may mix
  both sides of the plate in one baseline (unsure how Savant signs them).
- A role change (regular to platoon) makes the previous-season scale too small for noisier windows.
- Days with few games have coarse daily ranks (midranks for ties; hitters per day are reported).
- 2026 is the only test season (one season, about 200 placements of non-pitchers); the next untouched data
  is the 2027 season, on which the 2024-2025 fitted M2 and the alarm rule can be scored unchanged (with the
  2026 seasons as previous-season scales and the 2027 cross-section for daily ranks).

## Changes made before freezing (2026-10-10; no feature had been compared with the injured list)
- From the plan audit: centre from the current season; short windows added; early exit, raw rest gap,
  pre-2024 IL history and previous-season games added to playing time and B0; the floor now rotates only the
  anomaly block with true labels; U4, the U2-subset AUCs, event-level AUC, the realised alarm rate and the
  sizing were added.
- From code tests on a random sample of 61 hitters (sample seed 7: 60 hitters with more than 2,500 pitches
  in 2025-2026 plus Ohtani, all three seasons, real windows). The real IL transactions were fetched and held
  in memory, but before any score was computed they were replaced by random placements (per hitter-season a
  Poisson(0.4) number of placements on his game days, uniformly, seed 99); the only real-outcome quantities
  printed were the placement and region counts over all players 2022-2026. Design variants looked at: (1)
  in-season baseline with in-season MAD, raw U scores (U2/U3 AUC 0.58-0.62 on random placements, IL30 lower
  bound above 0.5); (2) previous-season scale and daily ranks (all AUCs 0.46-0.58); (3) the version frozen
  here, after a second plan audit (P1 0.531, 97.5% interval 0.429-0.644, below the max of its 200 rotations
  0.577). With 61 hitters this test could not detect an artefact of AUC 0.55; P1's rotation floor is the guard
  in the real run.
- From the second plan audit: the league-typical scale replaces the in-season MAD; P1's rotation floor and
  the label-free checks were added; limits on switch hitters, role changes, sparse days and 2027 added.
- From the code audit: P1's rotation moves only defined values; the daily rank is (midrank - 0.5) / count;
  the alarm is the daily top 1% rounded down, at least one; U3/U4 only where U2 is defined; the parse check
  on Ohtani's 2026-09-08 placement restored; a planted-signal check (random-placement code test only) asserts
  that P1's floor can be beaten. The changes after the second plan audit were not re-read by that auditor.

## AMENDMENTS
