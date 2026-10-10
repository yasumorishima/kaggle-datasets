# Does a Triple-A hitter drift from his own normal before the injured list? Frozen before any feature below is compared with a Triple-A injured-list move

FREEZE_anomaly.md (MLB, frozen in `20cc037`) found no support on 2026: the daily rank of U2 (mean squared
deviation from the hitter's own normal) had AUC 0.49 for an injured-list placement within 14 days, and the
anomaly model added +0.007 AUC over the rival model (interval across 0). One season of about 200 placements
can only see effects of roughly AUC 0.55 or a gain of 0.03-0.04. This file repeats the test on **Triple-A
2024-2026**, a population whose injured-list moves have never been compared with any feature, and adds one
new causal feature suggested by the MLB diagnostics.

What has already been seen, stated before the run:
- MLB 2024-2026: everything in FREEZE_anomaly.md and its diagnostics notebook (`hitter-anomaly-il-diagnostics/`,
  `57ac035`). The diagnostics found that the frozen P2 floor rotated the block's missing values with it; with
  gaps kept on their dates the floor's mean gain was +0.0086 (real +0.0068). A **non-causal** season mean of
  each hitter-season's deviations added +0.019 over the rival model on MLB 2026 (it uses later rows of the
  same season, including rows after a return from the injured list).
- Triple-A, label-free only: on one day in each of 2023-2026 every game (15) is in Savant's minors search;
  among balls in play launch_speed, launch_angle and hc_x are 99% or more filled and
  estimated_woba_using_speedangle 97-99.6%; bat-tracking columns are empty in every Triple-A season. A code
  test of the fetch notebook on the first two game days of 2024 and 2026 found every finished game present.
- The fetch notebook (Kaggle `yasunorim/triple-a-statcast-2023-2026-fetch`, code `4acf8b1`) ran once: every finished
  game present in every season (2,224 / 2,232 / 2,232 / 2,223 games; 691,656 / 675,342 / 671,257 / 671,556 pitches for
  2023-2026), no game outside the schedule. Parquet md5: 2023 a81f27d0a67211923e337214bbdaeb5c, 2024
  73d3e7081e04b34fc1bdbc6cd18e872e, 2025 093dc465d9b7eae8ef29dce8f66685b4, 2026 fc32bbaf9bc9476d14416c5463760a91.
  The study reads that output; the study prints the md5 of the files it reads.
- Triple-A transaction counts by type (MLB StatsAPI, sportId=11, 2023-2026), counted without any feature:
  injured-list placements whose description names a non-pitcher are 303 (2023), 266 (2024), 270 (2025),
  259 (2026), mostly the 7-day list; many descriptions carry no injury. MLB clubs' rehab assignments appear
  as "<club> sent <player> on a rehab assignment to <club>" (sportId=1).
Nothing below is changed after the run; any change goes in AMENDMENTS with its date and reason, before the run.

## Run order
The pitch completeness gates (fetch notebook) and every label-free check of the study notebook run and must
pass before any injured-list placement is joined to the rows. A failed gate gives no verdict; it allows a dated
amendment and a re-run.

## Data
- Pitch-by-pitch Triple-A Statcast, regular season 2023-2026, fetched by a separate notebook
  (`aaa-statcast-fetch/`) from Baseball Savant's minors search (`savant-extras` >= 0.6.0,
  `statcast_minors(level="AAA")`, one request per day with a finished game) and attached to the study as its
  output. 2023 is used only as the previous season of 2024 (centres, scales) and for the league-typical scale;
  no 2023 row is scored.
- Fetch gates (fail the fetch): every day with a finished Triple-A regular-season game (StatsAPI schedule,
  sportId=11, gameType R, state Final / Completed Early / Game Over) returns rows after up to five tries; at
  least 97% of finished games are present in every season; no major-league game; regular season only.
  Study gates (fail the study): per season, zone, description, stand at least 99% filled; woba_denom present
  on 24-27% of pitches; among balls in play launch_speed, launch_angle, hc_x, hc_y at least 97%.
- Injured-list moves: StatsAPI transactions 2022-2026 with sportId 1 (MLB clubs), 11 (Triple-A) and 12
  (Double-A). A placement says "placed ... on the ... injured list"; a return says "activated", "reinstated" or
  "returned" ... "from the ... injured list". Effective date = effectiveDate (or date). The club is the
  transaction's toTeam (or fromTeam), classified by that season's team lists (/teams, sportId 1, 11, 12).
  Placements of the same player less than 10 days apart are merged (as in the MLB notebook).
  - **Labels** use only placements by MLB or Triple-A clubs. Double-A placements count only in the IL history.
  - Body region from the injury sentence with the MLB notebook's rule; a region analysis runs only with at
    least 60 positive rows of that region.
- Rehab windows: a window starts at an MLB club's placement of the player, or at an MLB club's "sent ... on a
  rehab assignment" (whichever is earlier when both exist), and ends at the first of: a return at either level,
  a release, outright, option or designation for assignment, or the end of that season. His Triple-A pitches inside a
  window are removed before anything is computed (he is already injured, and his normal is not Triple-A's).
  Gate: removed pitches are printed per hitter-season; the run fails if a hitter-season loses more than 50% of
  its pitches without any rehab-assignment transaction for him that season.
- MLB game days of each batter, 2023-2026 (call-up items and the label rule below): StatsAPI game logs
  (`/people` with `hydrate=stats(group=[hitting],type=[gameLog],season=Y,sportId=1)`), any appearance.
- Primary position and birth date: StatsAPI /people (today's primary position, applied to every season). Primary
  position P excluded.

## Rows and outcome
- One row per batter and Triple-A game day t (at least one pitch seen that day, outside rehab windows),
  2024-2026. Every feature on day t uses only Triple-A pitches from earlier days (strictly before t), same
  season.
- IL14(t) = a placement (by an MLB or Triple-A club) with effective date in [t, t + 14 days], primary. A
  placement labels row t only if he has **no MLB game day in (t, effective date]**; rows with an MLB game day in
  that interval followed by a placement within 14 days are **dropped** (neither positive nor negative), because
  the injury may have happened in MLB play; the number dropped is reported. IL30 is reported alongside with the
  same rule.
- Rows whose 14-day window passes the last Triple-A regular-season date of their season are dropped.

## Parameters (P = 11), each a rolling value over the batter's most recent Triple-A events before t
The 11 non-swing parameters of FREEZE_anomaly.md, with the same definitions, windows and fill rule:
EV90, launch angle, sweet-spot share, pull share, xwOBAcon (balls in play: long 40, short 15); swing rate
(pitches seen: 150 / 50); chase (zones 11-14: 100 / 30); whiff (swings: 150 / 50); K, BB, wOBA (PA: 100 / 30).
Playing time as in FREEZE_anomaly.md (days between Triple-A games and PA per game over the last 10 game days,
rest gap, early exit), plus two call-up items for the rival model: days since his last MLB game day before t
(capped at 365; 365 if none since 2023) and his MLB game days this season before t.

## Own-normal deviation (z)
Exactly FREEZE_anomaly.md: centre = median of this season's values up to t - 30 days when there are at least
20, else the previous Triple-A season's median (at least 20); scale = 1.4826 x MAD of his previous
Triple-A season when it has at least 60 values, else the league-typical scale = median over hitters with at
least 60 values of the whole-season MAD **in Triple-A 2023** (label-free, and not a scored season).
z clipped to [-10, 10].

## Unsupervised scores (no labels)
- U2 = mean of z^2 over the 11 short-window parameters with a value, **at least 5** (FREEZE_anomaly.md asked
  8 of 18; 5 of 11 keeps the share). U1 = max |z| over the same.
- U3 (Isolation Forest) and U4 (MinCovDet) as in FREEZE_anomaly.md, fitted on Triple-A 2024-2025 rows where
  U2 is defined; reported on 2026.
- Every U score is scored as its daily rank among Triple-A hitters with a value that day, (midrank - 0.5) /
  count.

## New causal feature (from the MLB diagnostics)
- D(t) = for each of the 11 short-window z and for U2, the mean of his values on his earlier game days of the
  same season (strictly before t), defined when at least 10 values exist. Each z already uses only pitches
  before its own day, so D on t uses nothing from t or later. It is the causal form of the non-causal season
  mean that added +0.019 on MLB 2026.

## Models (learner and settings exactly FREEZE_anomaly.md; trained on Triple-A 2024-2025, scored on 2026)
- B0 (rival): age, position group, IL placements at any of the three levels since 2022-01-01 before t, days
  since his latest return at any level on or before t (capped 365), previous-season games at MLB plus Triple-A
  (StatsAPI season stats of both levels summed; team splits summed when there is no total), game days so far
  this season, day of season, playing time (raw, and z for the two 10-game items), and the two call-up items.
- M2: B0 + the 22 z (11 parameters x 2 windows) + U1, U2.
- M3: M2 + D (12 values).
- Not done: applying the MLB-trained models to Triple-A. B0's inputs do not mean the same thing at the two
  levels, so a transferred model would mix a level shift with the anomaly question. U2 needs no training and
  is the transferred score.

## Primary tests (three; Bonferroni, each interval read at the 0.833% and 99.167% points; bootstrap over
batters, the cluster being the batter id across seasons, 2,000 resamples, seed 2. The floor gates (resolution
1/21 or 1/201) are extra guards; error control is in the intervals.)
- P1 (the frozen unsupervised score on new data): AUC of U2's daily rank for IL14 on the Triple-A rows of
  **2024, 2025 and 2026 pooled** where U2 is defined (no training is involved). Supported if the lower bound is
  above 0.5 AND the real AUC is above all of 200 rotations (seeds 1000-1199) of each hitter-season's defined
  daily-rank values by a random whole number of positions between 30 and (count - 30) (half the count when
  fewer than 61), as in FREEZE_anomaly.md.
- P2 (anomaly model over the rival, 2026): AUC(M2) - AUC(B0), paired. Supported if the lower bound is above 0
  AND the real gap is above all of 20 rotation floors (seeds 100-119): in each hitter-season of training and
  test, one shift k between 30 and (game days - 30) (half the days when fewer than 61) is drawn; each z column
  with n_c defined values is rolled over its defined values by k when k <= n_c - 30, else by n_c // 2 (columns
  with fewer than 2 defined values stay); every missing value stays on its date; U1 and U2 are recomputed from
  the rotated z (so where U2 is defined does not move); labels and B0 untouched; M2 refitted. The share of
  defined values moved by fewer than 15 positions is printed.
- P3 (season-to-date drift adds to the anomaly model, 2026): AUC(M3) - AUC(M2), paired. Supported if the lower
  bound is above 0 AND the real gap is above all of 20 swap floors (seeds 200-219), each floor being
  AUC(M3 with swapped D) - AUC(M2): within each season (training and test alike), hitter-seasons are matched by
  a random derangement among the hitter-seasons with any D value; each recipient keeps his own D mask, and his
  defined positions receive the donor's defined D values in order (cycling when the donor has fewer; for a column
  in which the donor has none, the next hitter-season along the permutation cycle that has values gives them);
  z block, B0 and labels untouched; M3 refitted.
  Passing means a hitter's own season-to-date deviation adds risk information beyond the daily deviations; it
  does not by itself mean the drift comes before the injury in time.

## Interpretation rules, fixed now
- Injury or roster move: P1, M2 - B0 and M3 - M2 are also computed with positives restricted to placements
  whose description carries injury text (other placements' rows dropped). If a primary passes with all
  placements but its point estimate with injury-text placements only is at or below its floor mean, it is
  reported as **not attributable to injury**.
- Environment change: P1, M2 - B0 and M3 - M2 are also computed on rows more than 30 days after a return from
  the injured list, and on rows more than 30 days after his last MLB game day. If a primary passes but its
  point estimate in either subset is at or below its floor mean, it is reported as **timing of a return or
  demotion**, not drift before an injury.
- Results versus process: permutation importance of M2 and M3 on 2026 is summed separately over results
  parameters (K, BB, wOBA, xwOBAcon, EV90 z and their D) and process parameters (launch angle, sweet spot,
  pull, swing rate, chase, whiff).

## Reported whatever the result
- P1 by season (with its rotation floor by season) and by baseline kind; U1, U3, U4 on 2026; B0, M2, M3 for
  IL14 and IL30 with intervals; AUC(M3) - AUC(B0); M3 - M2 against a rotation of z with D recomputed from the
  rotated z (reported, not a verdict condition).
- Counts: rows, hitters, placements by level and list, placements with a parsed return before his next
  Triple-A game, rehab windows and pitches removed, rows dropped by the MLB-game rule, injury-text share, the
  share of 2026 placements with no Triple-A row in the 14 days before (and before his first 2026 Triple-A game),
  and placements x the share of their pre-IL rows with U2 defined (the effective sample of P1).
- Alarm rule as FREEZE_anomaly.md (top 1% of hitters that day by U2, rounded down, at least one): alarm
  days, share followed by IL14, share of placements with an alarm in the 14 days before.
- Event level (as FREEZE_anomaly.md), permutation importance of M3 (top 10), label-free checks (rank vs rows,
  previous season, own-scale share; baseline kinds; z sd by season; hitters per day).

## Code test (random placements, before the run)
As FREEZE_anomaly.md: the real transactions are replaced by random placements (Poisson 0.4 per hitter-season,
seed 99) before any score is computed. Planted checks that must hold: a timing signal planted in one short z
(aligned with the random labels) beats all P2 floors; for a constant planted per hitter-season (a trait) the P2
floors keep at least 80% of its gain on average; a D trait planted per hitter-season beats all P3 swap floors; P1's floor is beatable by a
planted signal.

## What replaces MLB-specific checks
- Injury-text share and region shares are printed, not asserted (Triple-A descriptions often carry no injury).
- The Ohtani parse check is replaced by a parse check on a fixed Triple-A placement found while sizing
  (written into the code before the run, chosen without looking at any feature).
- Season end: the last Triple-A regular-season date must fall between 15 September and 6 October.
- Previous-season games read sport ids 1 and 11.

## Sizing (label-free counts above)
About 250-270 non-pitcher placements a season. Triple-A rosters turn over, so fewer placements have pre-IL rows
with U2 defined than in MLB; the run prints placements x that share. With 600 effective placements over three
seasons SE(AUC) is about 0.012-0.013 (Hanley-McNeil, before clustering): at the 99.17% level P1 has 50% power
at AUC about 0.535 and 80% power at about 0.545. P2 and P3 have one test season (about 200 placements): a gap of
roughly 0.035-0.045 at 50% power.

## Known limits, stated before the result
- The minor-league 7-day list is also used for roster moves; a struggling hitter is a natural candidate, so
  results-type deviations can predict a roster move (hence the injury-text rule).
- A Triple-A hitter's season is broken by call-ups; the series uses Triple-A pitches only.
- Triple-A rosters turn over, so more rows use the league scale and fewer have a previous Triple-A season.
- Zone-calling rules (the automated ball-strike system) changed across Triple-A 2023-2026, which can shift K,
  BB, chase and swing rate against a previous-season centre early in a season (P1 by baseline kind is reported).
- Pitch data are fetched once by the fetch notebook; Savant may revise data later, so per-season pitch and game
  counts and file md5 are printed for the record.

## Changes made before freezing
- From the plan audit (opus, 2026-10-11): P3 compares M3 with M2 (the first draft compared it with B0, which M2's
  gain alone could pass); labels exclude MLB-play injuries and Double-A clubs; rehab windows end at returns,
  releases, outrights, designations or season end, with a gate; injury-text and environment-change
  interpretation rules; the P2 floor rolls each column by k only when it has room; planted checks for the P2 and
  P3 floors; MLB game days from StatsAPI game logs from 2023; run order; the P3 swap keeps the recipient's mask;
  fill-rate gates; returns include "reinstated"/"returned"; Double-A IL history; P1 by baseline kind; sizing at
  80% power.
- From the code audit (opus, 2026-10-11): rehab windows start only at an MLB club's rehab assignment (Triple-A
  clubs also send their own players to lower levels); the swap draws among hitter-seasons with D and fills a
  donor's empty column from the next along the cycle; coverage and duplicate-pitch gates in the study; placements
  by list and the share with a parsed return are printed; a PA pitch without woba_value is left out of K, BB and
  wOBA (its count is printed).
- From the full-season code test on a sample of 120 hitters (random placements): the rehab gate stopped the run on a
  hitter placed on an MLB injured list and later optioned to Triple-A with no activation recorded (an injured
  player cannot be optioned), so an option now ends a window.
- From the same code test: the planted-trait check first asked that the trait not beat the P2 floors' maximum by
  more than 0.005; the trait gain was +0.328 against floors up to +0.319 (the floors keep the trait, but models
  differ by about 0.02 at random), so the check now asks the floors' mean to keep 80% of the trait's gain.

## AMENDMENTS
