# Pre-registration: do pitchers' in-outing fade and scatter drift before the injured list? (MLB and Triple-A, 2024-2026)

Status: FROZEN when merged to main (notebook `pitcher-inouting-il/pitcher_inouting_il.py`, md5 e4e6a8d89ace682b91eef8309980acab); the merge commit is recorded in the README line of this study.

## Why this study

The frozen pitcher study (`FREEZE_pitch.md`, commit `2c6b01d`, run once on 2026-10-11) found no support for any of its
four primaries: the drift of 14 window **levels** (window means and a 90th percentile of velocity, spin, extension,
release point, arm angle, movement and results) from each pitcher's own normal did not come before an injured-list
placement (MLB AUC 0.518, arm placements 0.535, Triple-A 0.543, each inside its rotation floor; the model over the
rival -0.022).

A level can stay normal while the delivery inside an outing changes: velocity can fall faster as the outing goes on
(fatigue), and the release point, arm angle and spin axis can scatter more from pitch to pitch (loss of repeatability).
None of these was measured. This study measures them with the same rows, labels, rival, floors and tests.

## Disclosures (before any new parameter met a label)

- **Read before this plan:** the complete printed output of the frozen pitcher run: P1-P4 with floors, their subset
  tables (season, role, centre kind, time since a return / the other level), delivery-only, pooled-role, IL30 and U1
  AUCs, P2 / Triple-A model tables and subsets, permutation importance by group and its top 10 (`n_prev_il`,
  `z_FB velocity (short)`, `throws`, ...), the event level and the alarm rule. It motivated the choice of in-outing
  dimensions.
- **Families on these placements:** this is the second pre-registered family on the same pitcher placements (MLB and
  Triple-A clubs, 2024-2026) after `FREEZE_pitch.md`. The Bonferroni correction below covers this family only.
- **Label-free probe** on the code-test sample (100 pitchers per level, pitch columns only): speed, release point, arm
  angle and spin axis all vary within an appearance (median within-appearance sd of a fastball type: speed 0.82 / 0.84
  mph, release side 0.10 / 0.11 ft, release height 0.06 / 0.07 ft, arm angle 2.1 / 2.3 degrees, spin axis 4.3 degrees in
  Triple-A; MLB / Triple-A); speed falls within an outing (pooled within-appearance slope -0.13 / -0.20 mph per 25
  pitches); fastball spin axis lies between 114 and 260 degrees in the Triple-A sample and 102-107 to 253-254 degrees in
  full MLB seasons (1st-99th percentile; under 0.11% of MLB fastballs below 30 or above 330 degrees, which the circular
  difference handles); release_extension is recorded in 0.1 ft steps, as large as its within-appearance sd, so it is not
  used. MLB spin_axis fill (full files, md5 as above, regular season): 99.16% / 99.39% / 99.47% of all pitches and 99.52%
  / 99.76% / 99.82% of fastballs in 2024 / 2025 / 2026; Triple-A spin_axis fill is printed by the run's fill gate.
- **Considered and dropped before freezing:** the spin axis mean (a level, overlapping the movement parameters already
  tested) and extension scatter (resolution).

## Design: the frozen code with only the parameters replaced

The notebook downloads the frozen pitcher notebook from commit `2c6b01d`
(`pitcher-anomaly-il/pitcher_anomaly_il.py`, md5 `100c853ccb8e42f386f547ead97f0b95`), checks the md5, applies a fixed
list of exact text replacements (each must match exactly once, else the run stops) and runs it. Everything not replaced
is the frozen study unchanged: the inputs and their md5 gates, the 90% fill rule, players and game logs, labels
(placements by MLB or Triple-A clubs within 14 days, rows with an appearance at the other level in the window dropped,
rehab windows), rows (one per pitcher and appearance day, 2024-2026; Ohtani 2026 left out), primary fastball choice,
own-normal centre and scale (role-matched own previous-season MAD, else the league-typical scale by role), daily ranks
within role, the rival B0, the models, the bootstrap over pitchers, the rotation floors and their gates, the planted
checks under fake labels, subsets, reported items and figures.

The replacements, all in the notebook:
1. `spin_axis` added to the columns read and to the printed fill table.
2. The parameter lists (FBP, FBCOL, OTHER, DELIVERY, BASE14) replaced by the five parameters below; OTHER is empty.
3. The fill gate maps each parameter to all of its source columns (release point: both release_pos_x and release_pos_z).
4. The per-type window loop in `params_for` replaced by a function computing the five parameters (below); `oth` is
   empty.
5. U1 / U2 need 3 of the 5 short z (the frozen study needed 7 of 14 short z; U1 / U2 use short z only, unchanged); the
   mechanics U2 needs 2 of its 3 short z (frozen delivery U2: 5 of 10). Smoke values scaled the same way.
6. The centre kind recorded per row (`base_kind`, used in subsets) is taken from "FB velocity scatter (short)" instead
   of "Zone rate (short)"; it now describes a fastball series.
7. The planted look-ahead check also sets spin_axis to an extreme value after t0 and asserts on the fastball series
   (it asserted on the other-parameter series, now empty).
8. The planted P2 checks use "z_FB velocity scatter (short)" instead of "z_FB velocity (short)".
9. Group labels: "delivery" becomes "mechanics", "results" becomes "velocity"; the interpretation note "results, not
   delivery" becomes "velocity, not mechanics", and its Triple-A qualifier "(a roster move by results)" becomes "(a
   roster move by velocity)".
10. Title, text, test labels Q1-Q4 and output file names (`inouting_*`); the numpy, pandas and scikit-learn versions
    are printed first.
11. Added: the equality gates (after the labels and the frozen "effective sample" line, before any test) and the
    directional fade item (at the very end, after the verdict files are written).

**Equality gates (real run only):** with B0 independent of the parameters and fixed seeds, the run stops unless these
equal the frozen run's printed values (a changed Kaggle image, library or StatsAPI correction stops the run here): rows 61,966 (MLB, 1,180 pitchers) and 55,686 (Triple-A, 1,853 pitchers); usable
rows 53,384 with 3,553 positive (MLB) and 45,453 with 1,884 positive (Triple-A); MLB 2026 AUC(B0) 0.5644912856858865.

## The five parameters (per fastball type FF / SI / FC; the z on a day is the one of that day's primary fastball)

An appearance is (pitcher, game_pk, game_date) (a suspended game resumed on a later date is two appearances; a
doubleheader is two appearances on one day, both before any later day). A pitch's order is its position among all the
pitcher's pitches in that appearance, sorted by at_bat_number and pitch_number, counted before filtering by type.
For each fastball type, an appearance contributes only if it has at least 4 pitches of that type with the column
defined; its other pitches carry a missing value. Deviations are from that appearance's own mean (or own line), so
they use only that appearance's pitches; a window that cuts through an appearance still uses deviations from the whole
appearance, which lies entirely before the day.

1. **FB velocity fade**: pooled within-appearance slope of release_speed on order: sum(dx * dy) / sum(dx^2) over the
   window's pitches with both defined (same mask in numerator and denominator), dx and dy deviations from the
   appearance means of order and speed over its pitches of the type with speed defined; in mph per 25 pitches (the
   denominator is always positive: only the oldest appearance in a window can be cut, and every appearance has at
   least 4 distinct orders).
2. **FB velocity scatter**: sqrt of the window mean of r^2 * n / (n - 2), r the residual of speed from the
   appearance's own least-squares line on order (the fade removed).
3. **Release point scatter**: sqrt of the window mean of (dx_side^2 + dz_height^2) * n / (n - 1), from release_pos_x
   and release_pos_z (a pitch needs both), in feet.
4. **Arm angle scatter**: sqrt of the window mean of d^2 * n / (n - 1), arm_angle.
5. **Spin axis scatter**: as 4 with d the signed angular difference of spin_axis from the appearance's circular mean
   (mean sine and cosine), wrapped to [-180, 180).

Windows: the last 150 (long) and 50 (short) pitches of that fastball type strictly before the day (the frozen
fastball windows); a value needs its window full and at least half of its pitches with the deviation defined (the
frozen rule). Ten z per row. Groups: velocity (1-2), mechanics (3-5).

## Primaries (Bonferroni over four: 0.625% and 99.375%)

The frozen tests, with the new parameters:

- **Q1** MLB 2024-2026, all placements: AUC of the daily within-role rank of U2. SUPPORTED if the lower bound > 0.5 and
  the AUC is above all 200 rotations (k uniform in [m, n - m], m = max(2, ceil(n / 6)), n >= 7).
- **Q2** MLB 2024-2026, arm placements only (negatives as Q1): as Q1.
- **Q3** Triple-A 2024-2026, all placements: as Q1.
- **Q4** MLB 2026, trained on 2024-2025: AUC(M2) - AUC(B0), M2 = B0 + the 10 z + U1 + U2. SUPPORTED if the lower bound
  > 0 and the gap is above all 20 gaps-fixed rotation floors (each z column rolled over its defined values by
  round(k * n_c / n), U1 / U2 recomputed).

A supported Q4 means the in-outing z add to the rival; M2 does not contain the frozen level z, so it does not mean
"beyond the levels".

## Reported, not tested

All items the frozen notebook reports, with the new parameters: subsets, mechanics-only U2, pooled-role rank, IL30, U1,
Triple-A 2026 M2 - B0, M3 (with season-to-date means of the new z), arm placements in 2026, permutation importance by
group (velocity / mechanics / rival / summary scores), the event level and the alarm rule. Added: **directional fade**:
the AUC of the daily within-role rank of minus the short fade z (a steeper fade than normal) on Q1's, Q2's and Q3's rows
where it is defined, with 200 rotations (seeds as Q1-Q3: 1000, 2000, 3000), overall and by role. Expected precision,
computed before the run: a reliever's 50-fastball window spans about 6 outings, so its fade has a standard error of
about 0.65 mph per 25 pitches against a typical fade of 0.13-0.20; a starter's about 0.11. Fade is informative mainly
for starters.

## Interpretation rules (fixed now)

As the frozen study, with "results, not delivery" replaced by **"velocity, not mechanics"** (written when the
mechanics-only AUC is not above its floor mean, or, for Q4, when the mechanics group's permutation importance is not
positive). A supported Q3 is "before a Triple-A placement, not necessarily an injury" (frozen wording).

## Code test

Before the real run, the notebook runs on the code-test sample with random placements (`SMOKE_FAKE_LABELS=1`) and must
finish with every gate and planted check passing (planted timing beats the floors; the floors keep a planted trait; a
role artefact does not pass Q1's interval; the look-ahead check). A separate audit recomputed the five parameters from
this text with independent code on 25 pitchers per level (largest difference 2.3e-13, no difference in which values
are missing) and checked causality by perturbing and deleting pitches on and after a day (values up to the day
unchanged in 129 of 129 cases; perturbing earlier pitches does change them).

## Known limits

Release point and arm angle come from park-specific camera systems; their calibration differs by park (more so in
Triple-A), which adds scatter between appearances but not within one. Pitch mix changes move which fastball type is
primary. The five z share one window, so "3 of 5" is close to all-or-nothing. The effective sample (placements times
share with U2) is printed as in the frozen run.

## Not done

No comparison of a new parameter with any placement before this file is on main. No change after the run except a dated
amendment that says whether it came before or after any result was printed. The frozen pitcher study's verdicts are not
revisited.
