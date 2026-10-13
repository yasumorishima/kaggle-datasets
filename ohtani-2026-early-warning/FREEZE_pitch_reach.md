# Pre-registration: does a pitcher's two-strike reach-back fade before the injured list? (MLB and Triple-A, 2024-2026)

Status: FROZEN when merged to main (notebook `pitcher-reach-il/pitcher_reach_il.py`, md5
`21980219105b5b21e714c35c2acd5938`); the merge commit is recorded in the README line of this study.

## Why this study

Four frozen pitcher studies found no support for any of their primaries:

- `FREEZE_pitch.md` (commit `2c6b01d`): window **levels** of 14 parameters against each pitcher's own normal (MLB AUC
  0.518, arm placements 0.535, Triple-A 0.543, each inside its rotation floor; model over the rival -0.022).
- `FREEZE_pitch_inouting.md` (commit `76537f0`): fastball **fade and scatter inside an outing** (0.513, 0.520, 0.462;
  -0.002).
- `FREEZE_pitch_mix.md` (commit `f7e2a53`): **pitch mix** and **secondary-pitch quality** (0.501, 0.501, 0.496; -0.023).
- `FREEZE_pitch_command.md` (commit `d0253d1`, amendment `4f758e2`): **command**, location scatter and zone outcomes
  (0.513, 0.519, 0.504; -0.015).

All four measured each pitch, or each window of pitches, without regard to the game situation it was thrown in. Not
measured: what a pitcher **keeps in reserve**. Most pitchers throw their fastball harder once they have two strikes
(code-test sample: +0.63 mph in MLB, +0.51 mph in Triple-A against the outing's own mean). A pitcher who is hurting, or
tiring over weeks, may hold his ordinary velocity and lose the extra he finds when he needs it, or may find it only by
changing his delivery. This study measures that reach-back with the same rows, labels, rival, floors and tests.

## Disclosures (before any new parameter met a label)

- **Read before this plan:** the complete printed output of the four frozen pitcher runs (tests with floors, subsets,
  reported items, model tables, permutation importance by group and its top 10, the event level and the alarm rule). In
  all four runs only the rival B0 (n_prev_il and other workload and history items) put AUC above 0.5. In the level run
  the short fastball velocity z was in the MLB top 10 of permutation importance on the MLB 2026 test labels; in the
  command run the group importances on those labels were B0 +0.053, scatter -0.007, zone outcomes -0.006.
- **Families on these placements:** this is the fifth pre-registered family on the same pitcher placements (MLB and
  Triple-A clubs, 2024-2026), and V4 reuses the MLB 2026 test labels on which the earlier importances were computed. The
  Bonferroni correction below covers this family only. Each verdict is therefore also printed at a **20-way reading
  across the five families** (0.125% and 99.875%), and a 4-way SUPPORTED is written as "supported within this family,
  not across the five pitcher families" unless the 20-way reading also holds. The 20-way bounds are the 0.125th and
  99.875th percentiles of the finite draws among 2,000 bootstrap draws (numpy's linear interpolation, at index
  0.00125 x (n - 1), about 2.5 of 0-1,999), noisier than the 4-way
  bounds.
- **How the family was chosen (label-free).** A first plan measured the fastball's joint dispersion over 8 delivery and
  flight columns and the Gaussian mutual information between the delivery block and the flight block over the last 50
  fastballs. A plan audit (a separate model run, label-free, on the code-test sample; its four probes re-run by me with the
  same output, and its reach-back numbers reproduced by a separate implementation) found that, after removing each pitcher's mean, the
  dispersion follows the within-outing scatter already measured by `FREEZE_pitch_inouting.md` (Spearman 0.71 MLB /
  0.74 Triple-A; with a level shift inside the window R^2 0.554 / 0.605), and that the coupling's change from one
  50-fastball block to the next is not distinguishable from shuffling whole outings (lag-1 excess +0.011 / -0.024 over
  that null). A second candidate, the response of outing-start velocity to rest days and previous load, had no stable
  per-pitcher value (season split-half correlation of the slope -0.12 to +0.29). Both were dropped. Reach-back was
  chosen on these label-free numbers (code-test sample, 50-fastball blocks of one type, deviations from the outing mean,
  lag-1 correlation of consecutive blocks after removing the series mean, against 20 shuffles of whole outings inside
  the pitcher-season-type series):

  | measure (MLB / Triple-A) | mean | split-half (odd / even outings) | lag-1 excess over the outing shuffle |
  |---|---|---|---|
  | two-strike minus fewer, velocity | +0.625 / +0.509 mph | 0.80 / 0.53 | +0.067 (3.6 sd) / +0.069 (3.0 sd) |
  | 0-2 or 1-2 minus 0-0, velocity | +0.717 / +0.573 mph | 0.78 / 0.51 | +0.125 (6.3 sd) / +0.107 (3.6 sd) |
  | two-strike minus fewer, extension | | | +0.029 (1.3 sd) / +0.047 (2.3 sd) (audit probe) |
  | two-strike minus fewer, arm angle | | | +0.068 (2.3 sd) / +0.047 (1.4 sd) (audit probe) |
  | (dropped) two-strike minus fewer, spin | | 0.34 / 0.31 (audit probe) | +0.023 (1.2 sd) / +0.001 (0.0 sd) (audit probe) |
  | for scale: window mean velocity | | | +0.338 / +0.373 (audit probe) |

  The velocity reach-back drifts about a fifth as much as the mean velocity and about half as much as the
  within-outing velocity scatter. Within a pitcher-season-type series its Spearman correlation with the block's mean
  velocity is -0.08 / -0.01, with p90 minus mean velocity +0.23 / +0.17, with the log SD of within-outing residuals
  +0.36 / +0.25 (partly mechanical: a larger reach-back widens the spread), with pitches per outing +0.07 / +0.05 and
  with rest -0.02 / -0.03 (audit probe). The extension and arm-angle reach-backs are small (median of the short value on
  the code-test rows: extension 0.005 ft; arm angle 0.18 degrees for relievers, 0.01 for starters) and are kept as the
  delivery-effort group because a pitcher who reaches back by changing his delivery is the case the plan is about; their
  weak drift is stated here so that a null in that group is not read as new.
- **New input:** the count before each pitch. MLB: the `balls` and `strikes` columns of the same md5-checked MLB files
  (not read by the frozen notebook). Triple-A: the same columns of the location fetch of `FREEZE_pitch_command.md`
  (md5 there, unchanged), joined by pitch key with that study's three gates (join share, same pitcher / batter side /
  date, zone agreement), and blanked where the join gates blank the location. **Count gate (added, label-free):** in
  every season and level, among regular-season pitches sorted by (game_pk, at_bat_number, pitch_number), the run stops
  unless the count is filled for at least 98% of pitches and the source count agrees with a count rebuilt from the
  descriptions of the earlier pitches of the same plate appearance (strikes: called / swinging / foul / foul tip /
  bunt foul / missed bunt / automatic strike / pitchout swings, capped at 2; balls: ball / blocked ball / pitchout / hit by
  pitch / intentional ball / automatic ball, capped at 3) on at least 99.5% of pitches for strikes, two-strike or not,
  and balls, and the first pitch of at least 99.5% of plate appearances is 0-0. Measured before freezing: full MLB
  files 2024-2026 (md5 equal to the frozen ones) 100.00% on every check; Triple-A (code-test sample joined to the full
  location files) at least 99.97% on every check; code-test run of the notebook at least 99.97%.
- **Considered and dropped before freezing:** spin reach-back (no drift beyond the shuffle, above); a reach-back of
  vertical movement (lag-1 excess -0.8 / +1.0 sd, audit probe); a line through the outing in place of the outing mean (the
  fade removed; same numbers to within 0.04 mph and 0.8 sd in the probe, so the simpler mean is used); counts from the
  descriptions in place of the source columns (identical on the measured files; the source is used and the rebuild is
  the gate); any directional item (no direction is fixed by a prior result).

## Design: the frozen code with only the parameters replaced

The notebook downloads the frozen pitcher notebook from commit `2c6b01d`
(`pitcher-anomaly-il/pitcher_anomaly_il.py`, md5 `100c853ccb8e42f386f547ead97f0b95`), checks the md5, applies a fixed
list of exact text replacements (each must match exactly once, else the run stops) and runs it. Everything not replaced
is the frozen study unchanged: the inputs and their md5 gates, the 90% fill rule, players and game logs, labels
(placements by MLB or Triple-A clubs within 14 days, rows with an appearance at the other level in the window dropped,
rehab windows), rows (one per pitcher and appearance day, 2024-2026; Ohtani 2026 left out), the primary fastball choice,
own-normal centre and scale (role-matched own previous-season MAD, else the league-typical scale by role), daily ranks
within role, the rival B0, the models, the bootstrap over pitchers, the rotation floors and their gates, the planted
checks under fake labels (including the look-ahead check), subsets, reported items and figures.

The replacements, all in the notebook:
1. The parameter lists replaced by the four parameters below (all fastball series; no non-fastball series).
2. Reading: the MLB files with the frozen columns plus batter side, balls and strikes; the Triple-A files with the
   frozen columns plus batter side, then the location join of `FREEZE_pitch_command.md` (its md5 check and gates), now
   also carrying strikes; then the count gate above, per file.
3. The fill gate maps each parameter to its source columns (a parameter is dropped at a level if any of them is under
   90% filled in any season); the fill table prints balls and strikes.
4. The per-type fastball window loop and the other-parameter loop in `params_for` replaced by the computation below.
5. U1 / U2 need 3 of the 4 short z (frozen: 7 of 14); the velocity-reserve U2 (the frozen "delivery only" U2) needs 2
   of its 2 short z (frozen: 5 of 10). Smoke value of MIN_Z: 2. Outings need at least 4 pitches of the type
   (`IN_MIN = 4`). The velocity, extension and arm-angle reach-backs use the same pitches and count sets, so they are
   defined together unless a column is missing (code test: 0.608 / 0.608 / 0.608 MLB relievers); the put-away item is
   defined less often (0.541).
6. The centre kind recorded per row (`base_kind`) is taken from "FB velocity reach-back (short)".
7. The planted V4 checks use "z_FB velocity reach-back (short)" instead of "z_FB velocity (short)".
8. Group labels: "delivery" becomes "velocity reserve", "results" becomes "delivery effort"; the interpretation note
   "results, not delivery" becomes "delivery effort, not velocity reserve", its Triple-A qualifier "(a roster move by
   delivery effort)".
9. Title, text, test labels V1-V4 and output file names (`reach_*`); library versions and the median short values by
   role printed.
10. Added: the equality gates (after the labels and the frozen "effective sample" line, before any test).
11. The frozen look-ahead plant (every pitch from the middle day on becomes a cutter with value 999) now sets
    release_speed, release_extension and arm_angle to 999 + pitch_number (so deviations from the outing mean are not
    zero), strikes to 2 on odd pitch numbers (so the first pitch of the day is not 0-0) and 0 on even ones, and balls
    to 1; the frozen check that nothing up to the day changes then covers the new series. The frozen requirement that
    a non-fastball series changes after the day (there is none here) is replaced by: no non-fastball series exists, and
    at least one long and one short fastball series changes after the day (this confirms the plant reached the new
    code; it is not a test of it). Added after it, at each level: the four parameters are recomputed by a plain loop
    over days (`reach_plain`, pandas group means and explicit windows) for the plant's pitcher-season, the first five
    (in sorted order) with two outings on one date, for each of the 24 series the first (in sorted order) with a defined
    value (the plant's if none has one, so that the coverage check below stops the run), and 60 drawn with seed 0, all
    among this level's pitcher-seasons of 2024-2026 and each counted once, and must equal the vectorised series (same
    parameter names, same NaN pattern, difference at most 1e-9); every one of the 24 series (3 types x 4 parameters x 2
    windows) must have at least one defined value compared at each level. The plant only sees leaks across days; this
    gate also sees a wrong use of the count, the outing or the window rules within a day.
12. Added: the subset "same role as the previous appearance" in the subset tables of V1-V4, the interpretation note
    "role change" (below), and a label-free print of the share of rows with each short z defined, by role.
13. The 20-way bounds and summary column.

**Equality gates (real run only):** the run stops unless these equal the frozen run's printed values: rows 61,966
(MLB, 1,180 pitchers) and 55,686 (Triple-A, 1,853 pitchers); usable rows 53,384 with 3,553 positive (MLB) and 45,453
with 1,884 positive (Triple-A); MLB 2026 AUC(B0) 0.5644912856858865 (tolerance 1e-6). The in-outing, mix and command
runs passed the same gates on Kaggle.

## The four parameters (long and short windows, only pitches strictly before the day, same season and level)

Per fastball type (FF, SI, FC), on the pitcher's pitches of that type in the season and level: a pitch's **deviation**
is its value minus the mean of the pitches of that type with a value in the same outing (game_pk and date); an outing
with fewer than 4 such pitches gives no deviation. The **count** is the source count before the pitch. A **window** is
the last 150 / 50 pitches of the type strictly before the day (the frozen fastball windows); its value is the mean
deviation of the pitches in set A minus the mean deviation of the pitches in set B, defined when the window is full, at
least half of its pitches have a deviation, and each set has at least 15 / 5 deviations (two-strike items) or 9 / 3
(put-away). The z used on a day is that of the day's primary fastball (frozen choice).

Velocity reserve:
1. **FB velocity reach-back**: release_speed; A = two strikes, B = fewer than two strikes.
2. **FB put-away velocity**: release_speed; A = 0-2 or 1-2, B = 0-0.

Delivery effort:
3. **Extension reach-back**: release_extension; A = two strikes, B = fewer.
4. **Arm-angle reach-back**: arm_angle; A = two strikes, B = fewer.

Eight z per row. Groups: velocity reserve (1-2), delivery effort (3-4).

## Primaries (Bonferroni over four: 0.625% and 99.375%)

The frozen tests, with the new parameters:

- **V1** MLB 2024-2026, all placements: AUC of the daily within-role rank of U2. SUPPORTED if the lower bound > 0.5 and
  the AUC is above all 200 rotations (k uniform in [m, n - m], m = max(2, ceil(n / 6)), n >= 7).
- **V2** MLB 2024-2026, arm placements only (negatives as V1): as V1.
- **V3** Triple-A 2024-2026, all placements: as V1.
- **V4** MLB 2026, trained on 2024-2025: AUC(M2) - AUC(B0), M2 = B0 + the 8 z + U1 + U2. SUPPORTED if the lower bound
  > 0 and the gap is above all 20 gaps-fixed rotation floors (each z column rolled over its defined values by
  round(k * n_c / n), U1 / U2 recomputed).

A supported V4 means the reach-back z add to the rival; M2 does not contain the earlier studies' z.

## Reported, not tested

All items the frozen notebook reports, with the new parameters: subsets, velocity-reserve-only U2 (the frozen "delivery
only"), pooled-role rank, IL30, U1, Triple-A 2026 M2 - B0, M3 (with season-to-date means of the new z), arm placements in
2026, permutation importance by group (velocity reserve / delivery effort / rival / summary scores) and its top 10, the
event level and the alarm rule.

## Interpretation rules (fixed now)

As the frozen study, with "results, not delivery" replaced by **"delivery effort, not velocity reserve"** (written when
the velocity-reserve-only AUC is not above its floor mean, or, for V4, when the velocity-reserve group's permutation
importance is not positive). A supported V3 is "before a Triple-A placement, not necessarily an injury" (frozen
wording).

Added: **"role change"**, written for V1-V3 when the AUC on the subset "same role as the previous appearance" is not
above that subset's floor mean, and for V4 when M2 - B0 there is not above its floor mean. Starters reach back more than
relievers (code-test median short velocity reach-back 0.68 vs 0.34 mph in MLB) and a move between roles changes the
counts a pitcher works in, while the daily role and the role-matched scale lag such a move.

Wording across families: the notebook prints the 20-way reading as its own column of the summary; in every write-up of
the result, a 4-way SUPPORTED whose 20-way column is NOT SUPPORTED is written "supported within this family, not across
the five pitcher families".

## Code test

Before the real run, the notebook runs on the code-test sample with random placements (`SMOKE_FAKE_LABELS=1`), in a
virtual environment with Kaggle's library versions (numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1; Python 3.11 there,
3.13 on Kaggle), and must finish with every gate and planted check passing (planted timing beats the floors; the floors
keep a planted trait; a role artefact does not pass V1's interval; the look-ahead check with item 11; the location join
gates; the count gate). Before freezing, planted leaks in the new code were each run through the code test and had to
stop it at the intended gate (checked from each log's assertion, not only its exit code), each built from the final
notebook (md5 above) on the code-test sample; the checker also requires each log to be newer than its mutant and each
mutant newer than the notebook, each mutant's text to equal the notebook with its one replacement, and each log to
start with its mutant's md5 (an earlier chain stopped while writing the mutants and the checker read the logs of the
previous version, found by the second code audit; all ten were then re-run from the final notebook):

| leak | stopped by |
|---|---|
| r1 windows include the day's own pitches | look-ahead check (`('SI', 'FB velocity reach-back')`) |
| r2 each pitch measured against the season mean of its type, not its outing's | look-ahead check (`('FF', 'FB velocity reach-back (short)')`) |
| r3 MLB strikes read from the next pitch of the plate appearance | count gate (MLB 2024: strikes agree 53.98%, first pitch 0-0 37.38%) |
| r4 Triple-A strikes rotated by one pitch inside each plate appearance (keys, pitcher, side, date, zone still match) | count gate (Triple-A 2023: strikes agree 47.49%) |
| r5 MLB balls counted after the pitch | count gate (MLB 2024: balls agree 66.34%) |
| r6 the count read from the pitcher's next pitch inside the reach-back code (crosses into the next day at the end of each outing) | plain-loop gate (`('reach-back differs from the plain loop', (670970, 2026), ('FF', 'FB velocity reach-back'))`) |
| r7 strikes counted after the pitch inside the reach-back code (no look-ahead across days, a wrong definition only) | plain-loop gate (same message) |
| r8 outings grouped by date only (two outings on one date share a mean) | plain-loop gate (`(640448, 2024)`, `('FF', 'FB velocity reach-back')`; the only MLB pitcher-season in the sample with two outings on one date; exercised at MLB only, as the Triple-A sample has none) |
| r9 a full window needs more than w pitches | plain-loop gate (`(670970, 2026)`, `('FF', 'FB velocity reach-back')`) |
| r10 more than half of the window with a deviation, instead of at least half | plain-loop gate (`(670970, 2026)`, `('FF', 'FB velocity reach-back (short)')`) |

The code test of the final notebook finished with every gate passing (`exit 0`, 39 replacements applied): count gate at
least 99.97% on every check and level; Triple-A zone agreement 95.21% / 95.59% / 95.53% / 95.44% (2023-2026); planted
timing beats its floors (MLB +0.460 against a floor maximum of +0.064, Triple-A +0.389 against +0.060); the floors keep a
planted trait (floor mean +0.3050 against +0.2948, +0.2015 against +0.1939); the role artefact does not pass V1's interval
(lower bound 0.421); the plain loop equals the vectorised parameters on 62 MLB pitcher-seasons (one with two outings on one date;
17,792 defined values, at least 303 for each of the 24 series) and 60 Triple-A pitcher-seasons (none with two outings on
one date in the sample; 12,108 defined values, at least 202 for each series).

## Known limits

Release speed, extension and arm angle are Statcast's tracked values; Triple-A tracking differs more by park, which the
outing-mean deviation removes only for shifts constant within an outing. Two-strike pitches are not a random subset: a
pitcher throws more of them when he is ahead and to weaker hitters, and the fastball he throws with two strikes may be
aimed differently (often up), so the reach-back mixes effort with intent. A pitcher who stops throwing his fastball with
two strikes has no value (undefined, not zero), which ties this family to the pitch-mix family. The extension and
arm-angle reach-backs are small and mostly noise at the short window (probe above).

## Not done

No comparison of a new parameter with any placement before this file is on main. No change after the run except a dated
amendment that says whether it came before or after any result was printed. The earlier pitcher studies' verdicts are
not revisited.

## AMENDMENTS
