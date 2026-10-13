# Pre-registration: does a pitcher's spin-axis gap drift before the injured list? (MLB and Triple-A, 2024-2026)

Status: FROZEN when merged to main (notebook `pitcher-ssw-il/pitcher_ssw_il.py`, md5
`5fa9c4a0e362f6618bd1f02e1a067afc`); the merge commit is recorded in the README line of this study.

## Why this study

Five frozen pitcher studies found no support for any of their primaries:

- `FREEZE_pitch.md` (commit `2c6b01d`): window **levels** of 14 parameters against each pitcher's own normal (MLB AUC
  0.518, arm placements 0.535, Triple-A 0.543, each inside its rotation floor; model over the rival -0.022).
- `FREEZE_pitch_inouting.md` (commit `76537f0`): fastball **fade and scatter inside an outing** (0.513, 0.520, 0.462;
  -0.002).
- `FREEZE_pitch_mix.md` (commit `f7e2a53`): **pitch mix** and **secondary-pitch quality** (0.501, 0.501, 0.496; -0.023).
- `FREEZE_pitch_command.md` (commit `d0253d1`, amendment `4f758e2`): **command** (0.513, 0.519, 0.504; -0.015).
- `FREEZE_pitch_reach.md` (commit `84b78af`): **two-strike reach-back** (0.493, 0.497, 0.492; +0.006).

All five measured quantities the pitcher controls through effort, release or location, read one at a time. Not
measured: the part of a pitch's movement that its spin does not explain. Hawk-Eye reports the spin axis it observes;
the direction of the movement implies the axis that spin alone (Magnus) would give. Their difference, the **spin-axis
gap** (seam-shifted wake), comes from how the seams sit at release, which depends on grip, wrist and finger action. A
pitcher whose forearm or elbow is hurting may keep his velocity, spin rate and release point while the way the ball
leaves his fingers changes. This study measures that gap with the same rows, labels, rival, floors and tests.

## Disclosures (before any new parameter met a label)

- **Read before this plan:** the complete printed output of the five frozen pitcher runs. In all five only the rival B0
  (n_prev_il and other workload and history items) put AUC above 0.5. In the level run the short fastball velocity z was
  in the MLB top 10 of permutation importance on the MLB 2026 test labels; spin_axis itself was not a parameter of any
  family (its within-outing scatter was a parameter of `FREEZE_pitch_inouting.md`; pfx_x and pfx_z levels of the
  fastball were parameters of `FREEZE_pitch.md`, of the primary breaking ball and changeup of `FREEZE_pitch_mix.md`).
- **Families on these placements:** this is the sixth pre-registered family on the same pitcher placements, and S4 reuses
  the MLB 2026 test labels on which the earlier importances were computed. The Bonferroni correction below covers this
  family only. Each verdict is therefore also printed at a **24-way reading across the six families** (0.104% and
  99.896%; the 0.104th and 99.896th percentiles of the finite draws among 2,000 bootstrap draws, numpy's linear
  interpolation, about index 2.1 of 0-1,999, noisier than the 4-way bounds), and a 4-way SUPPORTED is written as
  "supported within this family, not across the six pitcher families" unless the 24-way reading also holds.
- **How the family was chosen (label-free).** A plan audit (a separate model run, label-free) compared five directions on
  the code-test sample and on counts of placements only (no placement joined to a feature): a target change (arm
  placements with long stays: 84 in MLB 2026 with an MLB appearance in the 14 days before, 6 UCL, an AUC half-width of
  about 0.06, and the same feature type), a time-shape change (trend, CUSUM and short-lag change of the frozen levels:
  at most +0.0017 R^2 over the level z for the next outing's velocity, within-series correlation 0.23-0.60 with it),
  batters' contact depth from bat tracking (MLB only; +2.5 sd drift after count and location adjustment), dose items
  (cross-validated R^2 0.91 from B0's workload items) and the spin-axis gap. Spin efficiency (movement per 1,000 rpm)
  was dropped for its correlation with vertical movement (0.57). The audit's five probes were re-run by me with the same
  printed output; the drift numbers were reproduced by a separate implementation (outing means instead of pitch blocks,
  park x month removed leaving the pitcher out, 50 permutations of outing order).

  | measure (MLB / Triple-A) | defined, short window | season split-half | lag-1 excess over the outing shuffle (park x month removed) | partialled on family-1 items (pooled R^2) |
  |---|---|---|---|---|
  | fastball gap (50-pitch blocks) | 0.79 / 0.72 | 0.99 / 0.99 | +5.2 / +3.0 sd | +4.4 / +2.6 sd (0.12 / 0.10) |
  | changeup / splitter gap (30) | 0.53 / 0.56 | 0.98 / 0.97 | +8.7 / +8.2 sd | +5.8 / +8.5 sd (0.39 / 0.08) |
  | breaking-ball gap (30) | 0.74 / 0.70 | 0.98 / 0.97 | +6.1 / +5.3 sd | +6.5 / +5.2 sd (0.03 / 0.05) |
  | for scale: fastball mean velocity (50) | | | +13.0 sd (MLB) | |

  Separate implementation, consecutive outings: lag-1 z MLB +5.7 / +10.2 / +9.0, Triple-A +5.5 / +8.3 / +9.1 (fastball /
  changeup / breaking); at lag 4 the excess falls to +0.5 to +3.6, so the slow part spans two to three outings. Within a
  pitcher-season-type series the fastball gap's Spearman correlation with block mean velocity, within-outing velocity
  SD, pitches per outing and rest is at most 0.05 in absolute value, with pfx_x 0.42 (partly by construction) and with
  spin_axis 0.20. The season split-half (0.97-0.99) is mostly between-pitcher differences; the drift is the relevant
  number.
- **Park component (measured, not adjusted).** Consecutive outings in the same park are much more alike than outings in
  different parks the same number of days apart (MLB fastball, 3-5 days apart: lag-1 +0.37 same park, +0.09 different
  parks; the permutation null is near -0.05). Removing, from each outing, the mean gap residual of all other pitchers in
  the same game (full MLB files 2024-2026, md5 equal to the frozen ones; median 84 other pitches per game) barely changes
  it (same-park lag-1 fastball +0.250 -> +0.203, changeup +0.192 -> +0.201, breaking +0.172 -> +0.162), and a park x
  month adjustment does not either. So the park part is not a park-wide tracking calibration but a pitcher x park effect
  (its cause, for example the mound or the air, is not identified). It adds noise unrelated to the timing of placements and none of the causal adjustments tried removes it,
  so no park adjustment is applied. The pitcher part remains across different parks six or more days apart (MLB raw
  lag-1 fastball +0.03, changeup +0.16, breaking +0.09 against a null near -0.05; Triple-A +0.08 to +0.14 above a
  permutation null).
- **New input:** `spin_axis` from the same md5-checked MLB and Triple-A files (not read by the frozen notebook).
  Code-test sample, lowest season fill: 0.9913 (MLB) and 0.9829 (Triple-A); the run's 90% fill gate applies to the
  real files.
- **Convention gate (added, label-free):** with the movement-implied axis written as 180 - atan2(pfx_x, pfx_z) degrees,
  the median over pitcher-season-types (at least 30 pitches) of |median gap| must be at most 25 degrees and that of the
  mirrored convention (180 + atan2) at least 40; every gap must lie in [-180, 180). Code-test run of the notebook:
  15.7 and 86.6 degrees (MLB, 1,064 pitcher-season-types), 14.9 and 91.9 (Triple-A, 1,231). The run stops otherwise.

## Design: the frozen code with only the parameters replaced

The notebook downloads the frozen pitcher notebook from commit `2c6b01d`
(`pitcher-anomaly-il/pitcher_anomaly_il.py`, md5 `100c853ccb8e42f386f547ead97f0b95`), checks the md5, applies a fixed
list of 47 exact text replacements (each must match exactly once, else the run stops) and runs it. Everything not
replaced is the frozen study unchanged: the inputs and their md5 gates, the 90% fill rule, players and game logs, labels
(placements by MLB or Triple-A clubs within 14 days, rows with an appearance at the other level in the window dropped,
rehab windows), rows (one per pitcher and appearance day, 2024-2026; Ohtani 2026 left out), the primary fastball choice
(still used by the rival's previous-season fastball velocity and the "no primary fastball" count), own-normal centre and
scale (role-matched own previous-season MAD, else the league-typical scale by role), daily ranks within role, the rival
B0, the models, the bootstrap over pitchers, the rotation floors and their gates, the planted checks under fake labels,
the look-ahead check, subsets, reported items and figures. The look-ahead check and every gate added here (convention,
wrap, plain loop, selection, league scale) run in the real run as well and stop it if they fail.

The replacements, all in the notebook:
1. The parameter lists replaced by the three families below (no parameter of the frozen per-fastball kind and no
   other-series parameter), with the gap function and the convention gate.
2. Reading: the frozen columns plus `spin_axis`; the fill table prints it; `math` imported for the plain loop.
3. The fill gate maps each parameter to spin_axis, pfx_x and pfx_z (a parameter is dropped at a level if any is under
   90% filled in any season).
4. The per-type fastball window loop and the other-parameter loop in `params_for` replaced by the computation below.
5. U1 / U2 need 2 of the 3 short z (frozen: 7 of 14); the fastball-only U2 (the frozen "delivery only" U2) needs its 1
   short z (frozen: 5 of 10). Smoke value of MIN_Z: 2.
6. The z selection per family by its own primary type, and the league-typical scale of a family taken, in the league
   season (MLB 2025, Triple-A 2023), from each pitcher-season's whole-season most-thrown type of that family (the
   machinery of `FREEZE_pitch_mix.md`, items 4 and 5 there; the frozen design takes the fastball's league scale from
   the whole-season most-thrown fastball type in the same way). Added: the number of pitcher-seasons behind each
   league-typical scale is printed, and the real run (not the code test) stops if any scale is missing. The thinnest
   cell is Triple-A 2023 starters' long changeup series: 6 pitcher-seasons with at least 20 defined values in the full
   2023 file (md5 equal to the frozen one; before the roster and rehab filters), none in the 100-pitcher code-test
   sample (for scale, the starters' long fastball-gap series has 18 in the same count).
7. The centre kind recorded per row (`base_kind`) is taken from "Fastball gap (short)"; the planted S4 checks use
   "z_Fastball gap (short)".
8. Group labels: "delivery" becomes "fastball", "results" becomes "secondary"; the note "results, not delivery" becomes
   "secondary, not fastball", its Triple-A qualifier "(a roster move by secondary pitches)".
9. Title, text, test labels S1-S4 and output file names (`ssw_*`); library versions and the median short values by role
   printed.
10. Added: the equality gates (after the labels and the frozen "effective sample" line, before any test).
11. The frozen look-ahead plant (every pitch from the middle day on becomes a cutter, its pfx_x and pfx_z set to 999)
    now also sets spin_axis to 30 + pitch_number; the frozen check that nothing up to the day changes then covers the
    new series and the three primary-type choices. The frozen requirement that an other-series parameter changes after
    the day (there is none here) is replaced by: none exists, and at least one long and one short gap series changes
    after the day. Added after it, at each level: the six series and the three choices are recomputed by a plain loop
    over days (`ssw_plain`, its own trigonometry with `math`, its own window table and an assertion that the windows
    are 150 / 50 and 100 / 30) for the plant's pitcher-season, the first
    five (in sorted order) with two outings on one date, for each gap series the first (in sorted order) with a defined
    value, and 60 drawn with seed 0, all among this level's pitcher-seasons of 2024-2026 and each counted once; they must
    equal the vectorised series (same names, choices identical, same NaN pattern, difference at most 1e-9). By
    construction every gap series with a defined value somewhere at the level is compared (the draw includes, for each,
    the first pitcher-season with one); the fastball (FF), changeup (CH) and breaking (SL) series must be defined at each
    level, long and short. The Triple-A 2023 series (previous-season centre and scale for 2024) come from the same
    function and are not compared.
11b. Added in `build_rows` before the scores, on the same pitcher-seasons (Ohtani 2026 skipped): each row's value of
    each of the six parameters must equal the plain loop's series of that day's primary type of the family (NaN where
    the family has no pitch yet), and no z may be defined where its value is not. The plant and the plain loop see only
    what `params_for` returns; this gate sees the value selected from it. The z is taken with the same type index in
    the same lines and is not recomputed by the gate.
12. Added: the subset "same role as the previous appearance" in the subset tables of S1-S4, the interpretation note
    "role change" (below), and a label-free print of the share of rows with each short z defined, by role.
13. The 24-way bounds and summary column.

**Equality gates (real run only):** the run stops unless these equal the frozen run's printed values: rows 61,966
(MLB, 1,180 pitchers) and 55,686 (Triple-A, 1,853 pitchers); usable rows 53,384 with 3,553 positive (MLB) and 45,453
with 1,884 positive (Triple-A); MLB 2026 AUC(B0) 0.5644912856858865 (tolerance 1e-6). The in-outing, mix, command and
reach-back runs passed the same gates on Kaggle.

## The six parameters (long and short windows, only pitches strictly before the day, same season and level)

For each pitch with spin_axis, pfx_x and pfx_z: **gap** = wrap(spin_axis - (180 - degrees(atan2(pfx_x, pfx_z)))), wrapped
to [-180, 180) degrees. Each pitch type is its own series (the frozen per-type rule): the value of a window is the mean
gap of the last w pitches of the type strictly before the day, defined when the window is full and at least half of its
pitches have a gap (the frozen window rule).

1. **Fastball gap**: types FF, SI, FC; w = 150 / 50 (the frozen fastball windows).
2. **Changeup gap**: types CH, FS, FO; w = 100 / 30 (the frozen off-speed windows).
3. **Breaking gap**: types SL, ST, SV, CU, KC, CS; w = 100 / 30.

The z used on a day is that of the family's **primary type**: the type of the family with the most pitches this season
before the day (ties in the listed order); none before the family's first pitch of the season. Six z per row. Groups:
fastball (1) and secondary (2-3). No circular mean is used: per-series pitch SD of the gap is 19-33 degrees around
medians near 15, far from the wrap (gate above).

## Primaries (Bonferroni over four: 0.625% and 99.375%)

The frozen tests, with the new parameters:

- **S1** MLB 2024-2026, all placements: AUC of the daily within-role rank of U2. SUPPORTED if the lower bound > 0.5 and
  the AUC is above all 200 rotations (k uniform in [m, n - m], m = max(2, ceil(n / 6)), n >= 7).
- **S2** MLB 2024-2026, arm placements only (negatives as S1): as S1.
- **S3** Triple-A 2024-2026, all placements: as S1.
- **S4** MLB 2026, trained on 2024-2025: AUC(M2) - AUC(B0), M2 = B0 + the 6 z + U1 + U2. SUPPORTED if the lower bound
  > 0 and the gap is above all 20 gaps-fixed rotation floors (each z column rolled over its defined values by
  round(k * n_c / n), U1 / U2 recomputed).

A supported S4 means the gap z add to the rival; M2 does not contain the earlier studies' z.

## Reported, not tested

All items the frozen notebook reports, with the new parameters: subsets, fastball-only U2 (the frozen "delivery only"),
pooled-role rank, IL30, U1, Triple-A 2026 M2 - B0, M3 (with season-to-date means of the new z), arm placements in 2026,
permutation importance by group (fastball / secondary / rival / summary scores) and its top 10, the event level and the
alarm rule.

## Interpretation rules (fixed now)

As the frozen study, with "results, not delivery" replaced by **"secondary, not fastball"** (written when the
fastball-only AUC is not above its floor mean, or, for S4, when the fastball group's permutation importance is not
positive). A supported S3 is "before a Triple-A placement, not necessarily an injury" (frozen wording).

Added: **"role change"**, written for S1-S3 when the AUC on the subset "same role as the previous appearance" is not
above that subset's floor mean, and for S4 when M2 - B0 there is not above its floor mean (a move between roles changes
the pitch mix and the grip demands of a pitcher, while the daily role and the role-matched scale lag such a move).

Wording across families: the notebook prints the 24-way reading as its own column of the summary; in every write-up of
the result, a 4-way SUPPORTED whose 24-way column is NOT SUPPORTED is written "supported within this family, not across
the six pitcher families".

## Code test

Before the real run, the notebook runs on the code-test sample with random placements (`SMOKE_FAKE_LABELS=1`), in a
virtual environment with Kaggle's library versions (numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1), and must finish with
every gate and planted check passing. Before freezing, planted defects in the new code were each run through the code
test and had to stop it at the intended gate (checked from each log's assertion, not only its exit code), each built
from the final notebook (md5 above) on the code-test sample; the checker also requires each log to be newer than its
mutant and each mutant newer than the notebook, each mutant's text to equal the notebook with its one replacement, each
log to start with its mutant's md5 and to print the same replacement count as the unmutated run:

| defect | stopped by |
|---|---|
| s1 windows include the day's own pitches | look-ahead check (`('SI', 'Fastball gap')`) |
| s2 mirrored convention of the movement-implied axis (180 + atan2) | convention gate (MLB) |
| s3 no wrap to [-180, 180) | wrap gate (MLB) |
| s4 primary type chosen over the whole season | look-ahead check (`('choice', 'Fastball')`) |
| s5 the gap of each pitch uses the movement of the pitcher's next pitch (crosses into the next day at each outing end) | look-ahead check (`('CH', 'Changeup gap')`) |
| s6 one series per family instead of per pitch type | plain-loop gate (`(670970, 2026)`, `('FF', 'Fastball gap')`) |
| s7 a missing gap counted as zero | plain-loop gate (`(640448, 2024)`, `('FF', 'Fastball gap')`) |
| s8 long and short windows swapped | plain-loop gate (`(670970, 2026)`, `('FF', 'Fastball gap')`) |
| s9 a family's row value taken by the frozen primary fastball index | selection gate (`(670970, 2026)`, `'Changeup gap'`) |
| s10 a family's row value taken by the whole-season most-thrown type | selection gate (`(502043, 2024)`, `'Fastball gap (short)'`) |
| s11 fastball windows of the off-speed size (100 / 30) | plain-loop gate (`(670970, 2026)`, `('FF', 'Fastball gap')`) |

The code test of the final notebook finished with every gate passing (`exit 0`, 47 replacements applied): convention
gate 15.7 / 86.6 degrees (MLB) and 14.9 / 91.9 (Triple-A); gap defined on 0.9943 / 0.9855 of pitches; planted timing
beats its floors (MLB +0.466 against a floor maximum of +0.081, Triple-A +0.414 against +0.052); the floors keep a
planted trait (floor mean +0.3165 against +0.3064, +0.2465 against +0.2519); the role artefact does not pass S1's
interval (lower bound 0.486); the plain loop equals the vectorised series on 64 MLB pitcher-seasons (one with two
outings on one date; 11,140 defined values; never defined in the sample: CS and FO) and 64 Triple-A pitcher-seasons
(none with two outings on one date; 8,660 defined values; never defined: CS long, SV); the selection gate passed on the
same pitcher-seasons. Pitcher-seasons behind the league-typical scales in the sample (reliever / starter): MLB 2025
at least 10 / 16; Triple-A 2023 as few as 2 / 0 (the code-test exception above). Short z defined on MLB code-test rows
(reliever / starter): fastball 0.665 / 0.666, changeup 0.396 / 0.548, breaking 0.621 / 0.628.

An earlier version of the notebook (md5 `44c44415d634b687e8768e9e96dd59b5`) passed the same code test and stopped all
eight of its planted defects (s1-s8) at the intended gates; a code audit then found that no gate covered the
per-family selection in `build_rows` and that the plain loop shared the window table with the vectorised code, so
defects like s9-s11 would have passed (the audit's reading of the code; s9-s11 were not run on that version). Items 11 and 11b and the league-scale count were added, and every planted
defect above was re-run from the final notebook. A second audit found that the league-scale gate, as first written,
stopped the code test (the Triple-A 2023 cell above); it was limited to the real run.

## Known limits

The gap is computed from Statcast's tracked spin axis and movement; each park's Hawk-Eye installation, mound and air
differ, and the measured pitcher x park part (above) is noise this design does not remove. A change of grip or pitch
design made on purpose moves the gap without any injury. Pitch classification changes (slider / sweeper) move a pitch
between types and can change the primary type. The window of a primary type that the pitcher stops throwing keeps its
last value until another type of the family has more pitches (the mix study's limit). The gap of a pitch with little
movement (a gyro slider) has an unstable direction; such pitches widen the breaking-ball series.

## Not done

No comparison of a new parameter with any placement before this file is on main. No change after the run except a dated
amendment that says whether it came before or after any result was printed. The earlier pitcher studies' verdicts are
not revisited.

## AMENDMENTS
