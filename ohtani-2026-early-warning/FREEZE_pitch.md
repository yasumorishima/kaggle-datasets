# Does a pitcher drift from his own normal before the injured list? Frozen before any pitcher feature is compared with an injured-list move

The hitter studies found nothing: on MLB 2026 (FREEZE_anomaly.md) and on Triple-A 2024-2026 (FREEZE_aaa.md) the
daily rank of U2, the mean squared deviation of a hitter's parameters from his own normal, ranked the days before an
injured-list placement no better than chance, and an anomaly model added nothing over a rival model of known risk.
Hitters' parameters are outcomes of a swing against a pitch. A pitcher's delivery is measured directly on every
pitch (velocity, spin, extension, release point, arm angle, movement), and most pitcher placements are for the arm.
This file applies the same machinery to **pitchers, MLB 2024-2026 and Triple-A 2024-2026**.

What has already been seen, stated before the run:
- Everything in FREEZE_anomaly.md, FREEZE_aaa.md and their outputs (hitters only). No pitcher feature has ever been
  compared with any injured-list move.
- Label-free: in MLB 2024-2026 (a 61-hitter sample of the Kaggle dataset, all pitches those hitters saw) release_speed,
  release_spin_rate, release_extension, release_pos_x, release_pos_z, arm_angle, pfx_x, pfx_z and pitch_type are
  99.0-99.5% filled; on one Triple-A day each of 2025 and 2026 93.8-99.7%.
- Counts only (StatsAPI transactions, placements whose description names RHP or LHP): MLB clubs (sportId=1) 490
  (2024), 458 (2025), 485 (2026), of which an injury sentence naming the arm (elbow, shoulder, forearm, UCL, biceps,
  triceps, rotator cuff, flexor, lat, teres, labrum, Tommy John) 279, 260, 266; Triple-A clubs (sportId=11) 380, 422,
  419, of which 99% carry no injury sentence.
- A separate notebook (Kaggle `yasunorim/triple-a-statcast-pitching-2023-2026-fetch`, code `0381512`) fetched
  Triple-A pitches 2023-2026 with the pitcher columns and the hitter fetch's gates: every finished game present in every
  season (2,224 / 2,232 / 2,232 / 2,223), no game outside the schedule. Parquet md5: 2023
  508eb3a096050b50b6563e844440f06a, 2024 af5f2b7fc1707155613e1a2087cbbaca, 2025 849279d320d0f578e3b241eec5b2cb9a,
  2026 05a32b74a8ecf54793cb236bd78b8b5f. MLB file md5 (the dataset version of the hitter study): 2024 1e59aa536f7741d0e6981eb9e80b88b3, 2025 9aaf0ae673cc5b3f7ab29598da418abe, 2026 49cbf1be3bc454a7769ca447f2ad8e03.
- Label-free, from a random-label code test on 100 pitchers per level: Triple-A arm_angle is 96.0% filled in 2023 and
  97.9% in 2024 (other delivery columns 98.3% or more).
Nothing below is changed after the run; any change goes in AMENDMENTS with its date and reason, before the run.

## Run order
Data gates and every label-free check run and must pass before any injured-list placement is joined to the rows. A
failed gate gives no verdict; it allows a dated amendment and a re-run. The notebook reads a fixed column list; a gate
asserts that no column whose name contains "until" or "next" (e.g. pitcher_days_until_next_game) is read.

## Data
- MLB: pitch-by-pitch Statcast 2024-2026 from the Kaggle dataset "MLB Statcast + Bat Tracking (2024-2026)", the
  version used by the hitter study. Triple-A: the output of the pitching fetch notebook, 2023-2026 (2023 only as the
  previous season of 2024).
- Injured-list moves, rehab windows, MLB/Triple-A/Double-A team lists, people: exactly as FREEZE_aaa.md (with its two
  amendments), including the label rule that only MLB or Triple-A clubs' placements are labels.
- Game logs (StatsAPI `/people` hydrate `stats(group=[pitching],type=[gameLog],season=Y,sportId=S)`, 2023-2026) give
  each pitcher's appearance days at the other level.
- Population: pitchers whose StatsAPI primary position is P or TWP; Shohei Ohtani 2026 excluded (he motivated the
  project). Position players who pitch are excluded.
- Triple-A pitches inside rehab windows are removed (FREEZE_aaa.md rule and gate). MLB rows need no rehab rule.
- Fill gates (label-free): per season and column of every population and season read (Triple-A 2023 included); a
  parameter whose source column is under 90% filled in any season of a population is dropped from that population
  (both windows) before labels are joined, and the drop is printed. File md5, season end (15 September to 6 October)
  and duplicate pitches are gated for both levels.

## Rows and outcome
- One row per pitcher and appearance day t (at least one pitch thrown that day) at the population's level, 2024-2026.
  Every feature on t uses only pitches from earlier days (strictly before t), same season and level.
- IL14(t) = a placement (MLB or Triple-A club) with effective date in [t, t + 14 days]. **Every row with an appearance
  at the other level in (t, t + 14]** (Triple-A appearances inside a rehab window are not counted: a rehab after a
  placement is not pitching at the other level before it) is dropped, positive or negative (an injury may have happened there, and dropping
  only positives would keep optioned pitchers as negatives); the count is printed. IL30 uses (t, t + 30] in the same
  way. Rows whose window passes the season's last date are dropped.
- Role before t: starter if more than half of his appearances this season before t were starts (he pitched in the
  first inning of the game), else reliever; before his first appearance of the season, his previous-season role at
  this level; reliever when he has neither.

## Parameters (P = 14), rolling over the pitcher's most recent pitches before t (same season, same level)
Primary fastball type on t = the most frequent of FF / SI / FC among his pitches this season before t (ties: FF, SI,
FC order); if he has thrown no fastball this season yet, the most frequent in his previous season at this level; if
neither exists, fastball parameters are missing (the count is printed). **Fastball parameters are kept as separate
series per type** (each over his last 150 / 50 pitches of that type, with its own centre and scale from that type's
own series); the z used on t is the primary type's z, missing when that type has no baseline. The number of
primary-type changes per pitcher-season is printed.
Fastball parameters (per type):
1 velocity (mean release_speed), 2 top velocity (90th percentile of release_speed), 3 spin rate, 4 extension,
5 release height (release_pos_z), 6 release side (release_pos_x), 7 arm angle, 8 vertical movement (pfx_z),
9 horizontal movement (pfx_x).
Other parameters: 10 off-speed velocity (mean release_speed of his non-fastball pitches other than FA, PO, IN, EP,
last 100 / 30);
11 zone rate (zones 1-9, last 300 / 100 pitches); 12 whiff rate (misses per swing, last 150 / 50 swings);
13 K - BB per batter faced (last 100 / 30 batters; a batter faced is a pitch with an event other than a baserunning or
truncated event: caught stealing, pickoff, stolen base, wild pitch, passed ball, other advance, truncated plate
appearance, game advisory, runner double play); 14 hard-hit rate allowed (launch_speed >= 95 on balls in play,
last 40 / 15).
A window value needs its window full and at least half of its events with the column filled (the hitter rule).

## Own-normal deviation (z)
As FREEZE_anomaly.md, with thresholds counted in appearance days because a starter appears about 30 times a season:
centre = median of this season's values up to t - 30 days when there are at least 10, else the previous season's
median at this level (at least 10); scale = 1.4826 x MAD of his previous season at this level when it has at least 20
values, is above 0, **and his role before t matches his previous-season role**, else the league-typical scale **of his current
role** (median over pitchers of that role with at least 20 values of the whole-season MAD, role by more than half
starts in that season; MLB 2025 for the MLB population, Triple-A 2023 for the Triple-A population; label-free). A
starter's short window covers about one start and a reliever's several outings, so one scale for both would give the
roles different z spreads. z clipped to [-10, 10].

## Unsupervised score
U2 = mean of z^2 over the 14 short-window parameters with a value, at least 7. U1 = max |z|. U2_del = the same over
the delivery parameters 1-10 (at least 5). **Daily rank within role** (role before t) among the pitchers of that
population with a value that day, (midrank - 0.5) / count; the rank pooled over roles is reported.

## Models (learner and settings exactly FREEZE_anomaly.md; trained on 2024-2025, scored on 2026, per population)
- B0 (rival: known risk and workload): age, throwing hand, IL placements at any of the three levels since
  2022-01-01 before t, days since his latest return on or before t (capped 365), previous-season pitches at MLB plus
  Triple-A (StatsAPI season pitching stats, numberOfPitches summed), appearances so far this season, day of season,
  days since his previous appearance, pitches in his previous appearance, pitches in the last 7, 14 and 28 days at this
  level, share of his appearances this season before t that were starts (he pitched in the first inning; 0.5 before
  any), whether his previous appearance was a start, days since his last appearance at the other level (capped 365)
  and his appearances at the other level this season before t; his previous-season start share and a role-change flag
  (role before t differs from previous-season role); pitches in the last 7, 14 and 28 days at both levels combined
  (game-log numberOfPitches); earlier arm placements counted separately; his previous-season mean velocity of today's
  primary fastball type at this level (raw velocity is a known risk factor); appearances on consecutive days in the
  last 7 days; the most pitches he threw in one inning in his previous appearance. **Every B0 item uses appearances
  strictly before t.**
- M2: B0 + the 28 z + U1, U2. M3 (reported): M2 + D, the season-to-date mean of each short z and of U2 (strictly
  before t, at least 5 values; FREEZE_aaa.md's D with the threshold in appearance days).

## Primary tests (four; Bonferroni, each interval read at the 0.625% and 99.375% points; bootstrap over pitchers
across seasons, 2,000 resamples, seed 2; floor gates are extra guards)
- P1 (MLB, unsupervised): AUC of U2's daily rank for IL14 on MLB rows 2024-2026 pooled where U2 is defined.
  Supported if the lower bound is above 0.5 AND above all 200 rotations (seeds 1000-1199) of each pitcher-season's
  defined daily-rank values. **Rotation rule for pitchers**: a pitcher-season with n >= 7 defined values is rotated by k
  drawn uniformly from [m, n - m], m = max(2, ceil(n / 6)) (about 30 calendar days for a starter); series with n < 7 are
  not rotated and their share is printed. (The hitter rule rotated every series shorter than 61 by exactly n // 2, which
  for pitchers would make the 200 rotations near-copies.) Gate: at least 90% of defined values belong to rotated series;
  the number of distinct floor values is printed.
- P2 (MLB, anomaly model over the rival, 2026): AUC(M2) - AUC(B0), paired. Supported if the lower bound is above 0
  AND above all 20 gaps-fixed rotation floors (seeds 100-119): one k per pitcher-season of n rows drawn as in P1
  (n >= 7); each z column with n_c >= 2 defined values rolls over its defined values by round(k * n_c / n); missing
  values stay on their dates; U1, U2 recomputed; labels and B0 untouched; refitted. Gate as in P1.
- P3 (MLB, arm): as P1, with positives restricted to rows before an arm placement; rows before other placements
  dropped. Rotation floor as P1 (seeds 2000-2199). **Arm** = the lowercased injury sentence matches
  \b(elbow|shoulder|forearm|arm|ucl|ulnar|biceps?|triceps?|rotator cuff|flexor|lat|latissimus|teres|labrum|tommy john|thoracic outlet)\b
  and contains none of hip, groin, oblique, back, knee, ankle, foot, hamstring, quad, calf. Asserted strings: "left hip
  flexor strain" is not arm, "right lateral meniscus" is not arm, "right arm fatigue" is arm, "right UCL sprain" is arm.
  (The sizing counts above used an earlier list.)
- P4 (Triple-A, unsupervised): as P1 on Triple-A rows 2024-2026 (seeds 3000-3199).

## Interpretation rules, fixed now
- Return or level change: P1-P4 (and M2 - B0) are also computed on rows more than 30 days after a return from the
  injured list, and on rows more than 30 days after his last appearance at the other level. A primary that passes but
  whose point estimate in either subset is at or below the mean of the same rotations scored on that subset is
  reported as **timing of a return or level change**.
- Results versus delivery: U2_del's AUC with its rotations is reported for P1, P3 and P4; a primary that passes but
  whose U2_del AUC is at or below its floor mean is reported as **results, not delivery** (for P4: a roster move by
  results). Permutation importance of M2 on 2026 is summed over delivery parameters (1-10), results parameters
  (11-14) and the summary scores (U1, U2); if P2 passes and the delivery sum is not positive, it is reported as
  **results, not delivery**.
- Triple-A placements rarely say why; a passing P4 is reported as "before a Triple-A placement", not "before an
  injury".

## Reported whatever the result
P1/P3/P4 by season and their floors by season, by baseline kind and by role, with the pooled-role rank, and for IL30;
event level (each placement's pre-IL rows against 14-day blocks); placements with no pre-IL row (before his first
appearance, or no appearance in the 14 days before); share of rows per baseline kind by role; U1; P2 for Triple-A (trained on Triple-A 2024-2025) with its floor;
B0, M2, M3 for IL14 and IL30 in both populations; arm-only M2 - B0 (MLB); starters and relievers separately (role by
his appearances that season before t: more than half starts); alarm rule (top 1% of pitchers that day by U2): alarm
days, share followed by IL14, share of placements with an alarm in the 14 days before; counts (rows, pitchers,
placements by level/list/arm, rows dropped by the other-level rule, rehab removals, P1 effective sample = placements x
share of pre-IL rows with U2); label-free checks (U2 drift by month, baseline kinds, z sd, pitchers per day, rank vs
rows / previous season / own-scale share).

## Code test (random placements, before the run)
As FREEZE_aaa.md: random placements (Poisson 0.4 per pitcher-season, seed 99); a timing signal planted in one short z
beats all P2 floors; the P2 floors keep at least 80% of a planted pitcher-season trait's gain on average; P1's floor is
beatable by a planted signal; random placements at twice the rate on starter rows (a role artefact) do not pass P1's
interval; floors are not degenerate (the gate above); the arm strings; look-ahead (pitches on or after t set to
extreme values leave the primary type, the parameters and every B0 item unchanged up to t).

## Sizing (label-free counts above)
MLB: about 480 pitcher placements a season (about 270 arm). Placements per pitcher-season are higher than for hitters
and appearance days fewer, so the effective sample is printed per season before the verdict; 900 is an upper estimate
(2024 MLB has no previous season, so starters have no centre until mid-season). With 900 effective placements over three seasons
SE(AUC) is about 0.010 (before clustering): at the 99.375% level P1 has 50% power at about AUC 0.527; P3 (about 500
effective) at about 0.536; P4 (Triple-A, about 600 effective with turnover) at about 0.533; P2 (one season) a gap of
about 0.03-0.04.

## Known limits, stated before the result
- Many pitcher injuries happen in one pitch; a row's features end the day before, so a delivery change inside the
  game of day t is not seen until the next appearance (the next row is usually after the placement).
- Starters appear about every five days, so a starter has about two or three rows in a 14-day window.
- A changed pitch mix or a new primary fastball moves fastball parameters for reasons other than injury.
- Spin and movement measurement changed little across 2024-2026 at MLB; Triple-A spin is less complete in 2025 on the
  sampled day (93.8%).

## Changes made before freezing
- From the code audit (opus, 2026-10-11): md5 gates fail closed for both levels; the fill table is printed and only the
  90% drop rule applies (a 97% hard assert would have stopped the run on Triple-A 2023 arm angle, 96.0%); the fill gate
  covers description and launch_speed on balls in play; pitch types with missing values no longer break comparisons;
  the other-level rule ignores rehab windows; batters faced and off-speed pitches defined; own MAD of 0 falls back to
  the league scale; P1's planted check and the P2 gate added; the rotation gate applies to P1, P3 and P4 only; P2's
  subsets use the same floor models; IL30, U1, P3 pooled-role and P2 by role added to the reported items.
- From the plan audit (opus, 2026-10-11): pitcher rotation rule (the hitter rule degenerates for short series); league
  scale and daily rank by role, own scale only with an unchanged role; the other-level rule drops every row, not only
  positives; B0 items strictly before t and a gate on future columns; per-type fastball series; the arm word list with
  exclusions and asserted strings; U2_del and the results-versus-delivery rule; more B0 items; fill gates; subset
  floors from the same rotations; more reported items and planted checks.

## AMENDMENTS
