# Pre-registration: confirming the fastball spin-axis gap before arm placements on unseen seasons (MLB 2021-2023)

Status: FROZEN when merged to main (notebook `pitcher-ssw-confirm/pitcher_ssw_confirm.py`, md5 `4c51bcd69ed95212adb89d348f555ff6`; input
dataset `yasunorim/mlb-statcast-2021-2023-pitch-subset`, files below); the merge commit is recorded in the README line
of this study.

## Why this test

`FREEZE_pitch_ssw.md` (commit `e83a56e`, run once, GHA `38435663378`) was NOT SUPPORTED on all four primaries. Two of
its numbers stood out only after the run: on MLB 2024-2026 arm placements (S2) its AUC 0.533 was above all 200
rotation floors (maximum 0.528) with a Bonferroni lower bound of 0.489; the reported-only fastball-gap score on arm
placements was 0.5586 (Bonferroni 0.516-0.601; floor maximum 0.5587). A target chosen after seeing 2024-2026 cannot be
tested on 2024-2026 again. It can only be **confirmed** on seasons no pitcher study has read: MLB 2021-2023. Statcast
reports the spin axis from 2020; all six pitcher studies loaded MLB pitches from 2024 only.

This is a confirmation of one target picked after a null run, not a new family. A null here is the likely result even
if a small effect is real (the sample is about the size of the run that produced the target).

## The one primary (fixed before any 2021-2023 pitch feature meets a label)

**C1** MLB 2021-2023 pooled, arm placements (the frozen arm rule; negatives as the frozen P3/S2), injured list within
14 days: AUC of the daily within-role rank of the fastball-gap-only score (`U2_deld`: the mean of z² of the short
fastball-gap z of the day's primary fastball type; the frozen S2 "fastball only" score).

**SUPPORTED** if both hold:
1. the bootstrap-over-pitchers 95% lower bound (2.5th percentile of the frozen bootstrap, 2,000 draws, seed 2) is
   above 0.5, and
2. the rotation p-value is at most 0.025: p = (1 + #{floor >= AUC}) / 1001 over 1,000 rotations (seeds 5000-5999) in
   which each pitcher-season with at least 7 values (as frozen) is rolled by k drawn uniformly from its whole cyclic
   group 0..n-1, shift 0 included (`rotated_group`).

Printed beside it, not part of the verdict: the frozen rule's 200 rotations (seeds 2000-2199, as the frozen S2) with
their 97.5th percentile and maximum, the 99% lower bound, and the frozen Bonferroni (0.625%) bound.

### Why C1 does not use the frozen rotation rule (found before freezing, before any 2021-2023 label met a feature)

The frozen rule draws k uniformly on [m, n - m], m = max(2, ceil(n / 6)): shift 0 and the shifts near it are never
drawn. Those are the shifts whose positive rows overlap the real alignment's (positives come in blocks of consecutive
rows, and the scores are autocorrelated), so every rotation it draws is less like the real alignment than a random
member of the whole group would be, and the real AUC is too extreme against them: a test against them rejects too
often when labels carry no timing at all. This was found in the code test: with random labels (the frozen generator, seeds 99 to
102, 2021-2023 sample of 100 pitchers) C1's AUC ranked 200, 4, 40 and 199 among its 200 rotations, and seed 102 met
the planned rule (95% lower bound 0.5375, above the 97.5th percentile). A label-only diagnostic then built the rows once
and redrew the random placements many times, comparing the frozen rule with the whole-group rule (C1's rows of the
100-pitcher samples, 200 rotations each, seeds 2000-2199 for the frozen rule and, in the 1,000-redraw rows, 5000-5199
for the whole-group rule, not C1's 1,000; rejection = p <= 0.025 with p = (1 + #{floor >= AUC}) / 201; "z sd" is the sd over redraws of
(AUC - floor mean) / floor sd, about 1 for a floor of the right width):

| sample, redraws | frozen rule: rejection (planned "above the 97.5th percentile"), z sd | whole-group rule: rejection (at 0.05), z sd |
|---|---|---|
| 2021-2023 sample, 200 (`diag_floor2.py`) | 7.0% (8.0%), 1.38 | 3.0% (5.0%), 1.04 |
| same 200, score shuffled across all rows | 4.5% (5.0%), 1.23 | 3.5% (4.5%), 1.02 |
| 2021-2023 sample, 1,000 (`diag_floor3.py`) | 7.9% (8.7%), 1.39 | 2.5% (5.4%), 1.03 |
| 2024-2026 sample (the frozen families' seasons), 1,000 | 6.1% (6.6%), 1.37 | 2.5% (4.3%), 1.02 |

The threshold stays 0.025: the whole-group rate measured on 1,000 redraws is at the nominal level (binomial SE about
0.5 points). Had it been above 0.025 beyond that error, the threshold would have been lowered to the largest p whose
measured rate is at most 0.025 (stated here after 1,000 redraws had been read, so it is a description, not a
pre-commitment).

What the diagnostic does and does not show. The random placements fall on uniformly random appearance days and the
pitcher keeps pitching, so the labels are close to cyclically uniform and the whole-group p is valid whatever the
scores do; the diagnostic checks this mechanism and shows the frozen rule failing it. Real arm placements differ: a
long injured-list stint often ends a pitcher's rows for the season, so positive blocks sit near the end of a series,
and a rotation wraps them onto early-season values. If the score drifted within a pitcher-season for reasons unrelated
to injury, the floor would then be biased, in a direction set by the drift; the daily within-role rank removes
league-wide calendar trends but not drift inside one pitcher's season. This cannot be measured without linking 2021-2023
features to labels, so three checks are fixed now and reported beside C1 (not part of the verdict): (a) labels only,
the AUC of the position in the series with the same whole-group p (how far the labels are from cyclic uniformity);
(b) features only, the mean score by position decile over all usable rows; (c) C1 with the score minus its decile mean
from (b), with the same whole-group p. If (c) is far from C1, the report says that C1 rests on position; (c) close to
C1 does not rule out drift specific to some pitchers, which a pooled decile mean cannot remove.

The six frozen families' verdicts (all NOT SUPPORTED) are not changed by this, since a too-narrow floor only makes
SUPPORTED easier, but "above all 200 rotations" there is weaker evidence than it reads; this is added to the README.

## Reported, not tested

The full six-parameter U2 on arm placements and on all placements; the fastball-gap-only score on all placements;
IL30 (arm, fastball-gap-only); C1's subsets under the frozen rule's rotations (each season, starters, relievers, same role as
the previous appearance, centre kinds, more than 30 days after a return or the other level, **without 2021 from 3 June**
and **2022-2023 alone**, below); the frozen model test over the rival with TRAIN 2021-2022 and TEST 2023, its floors,
subsets and IL30.

## Data (new input; fixed now)

- MLB 2021-2023 regular-season pitch-by-pitch from Baseball Savant's `statcast_search` CSV (keyless, one request per
  day), built on GitHub Actions by `pitcher-confirm-statcast-2021-2023/build.py` (it imports the fetch and parse
  functions of `dataset4_statcast_bat_tracking/build.py`), with the 23 columns the notebook reads, and published as a
  **private** Kaggle dataset (created by GHA `38483678623` from `build.py` at `492e865`; downloaded back and checked):
  2021 `51c22a049453bb1ba5102d66e728a304` (712,320 pitches, 2,429 games), 2022 `ac2f365b4d00f0ab139b0a844615a3fc`
  (710,210, 2,430), 2023 `4bba35933470f5a8d6549059a486d59a` (720,684, 2,430). Its gates (nothing is written if one
  fails): every day's answer has Savant's header, the requested date, regular-season games of that season, known pitch
  descriptions, integer columns and zones; every Final regular-season game of the StatsAPI schedule has pitches and no
  game outside it does; no duplicate pitch key within or across seasons and no two seasons with the same content;
  per season the rows that are not pitch-clock automatic balls or strikes equal Savant's own season total and are at
  least 700,000; pitch_type filled on at least 90%; the last date in [Sep 15, Oct 6]; no kept column except arm_angle
  empty for a whole day, game or season, except games without pitch tracking, which are listed in `tracking_gaps.csv`
  (18 games; a season fails above 5%); the Arrow types of the frozen 2024-2026 files. A first build (GHA
  `38474797934`) failed on the two Field of Dreams games (2021-08-12, 2022-08-11: a temporary park without tracking,
  where woba_denom is empty too); they were then allowed to have woba_denom empty, as neutral-site games of the
  schedule. Fill: pitch_type 99.6 / 99.7 / 99.6%, spin_axis 99.2 / 99.4 / 99.0%, arm_angle 96.5 / 99.1 / 98.7%.
- Label-free probe (single days, 2020-2023): spin_axis filled on 99.1-99.7% of pitches, integer-valued in every
  season; median per pitcher-type |median gap| 10.8-12.7 degrees, within-type sd 7.3-8.1 degrees; ST and SV already
  present in 2021-2022 (retroactive classification); arm_angle filled 99.5% on 2022-06-15.
- MLB only. Triple-A Statcast does not exist for 2021 and is 42% filled for 2022. The other-level rule and the workload
  items still use Triple-A **game logs** from the StatsAPI (2020-2023; there was no minor-league season in 2020).

## Code: the frozen sixth-family notebook, MLB only

The notebook downloads `pitcher-ssw-il/pitcher_ssw_il.py` at commit `e83a56e` (md5 `5fa9c4a0...`), executes it up to
its last line (which downloads the frozen pitcher notebook at `2c6b01d`, md5 `100c853c...`, and applies its 47
replacements) and takes the resulting source. On that source it applies these replacements (14 in the regression run,
17 in the confirmation run) and nothing else:

| replacement | regression run (2024-2026) | confirmation run (2021-2023) |
|---|---|---|
| row seasons, TRAIN, TEST | unchanged (2024-2026, 2024-2025, 2026) | 2021-2023, 2021-2022, 2023 |
| MLB md5 | unchanged | the three files above |
| league-typical scale season | unchanged (2025) | 2022 (the middle season, as 2025) |
| placement history years | `range(first row season - 2, TEST + 1)` = 2022-2026 (unchanged) | 2019-2023 |
| game-log years | `[first row season - 1] + row seasons` = 2023-2026 (unchanged) | 2020-2023 |
| Triple-A pitches | an empty table | an empty table |
| Triple-A first/last dates (rehab windows) | StatsAPI schedule, completed regular-season games | same (2021 ends 10-03, the Final Stretch included) |
| fill table, convention gate, fill gates, labels, random code-test labels, placement counts (and their printed year range) | MLB only | MLB only |
| league-scale gate | also in the code test (the frozen code skipped it there because of a Triple-A 2023 cell) | same |
| Triple-A rows, labels, P4/S3, Triple-A model test, interpretation notes, figures, summary | removed (the source is cut before the frozen equality gates) | removed |
| rehab table dtype | `dtype=bool` (an empty Series is float otherwise and `~` fails; no effect on a non-empty table) | same |

With no Triple-A pitches, the frozen rehab check on removed Triple-A pitches (`lost`) is true by construction; the
rehab windows themselves still decide which Triple-A game-log days the other-level rule ignores. For MLB rows,
Triple-A pitches reach the frozen code only through `season_last["aaa"]` (the stop of a rehab window), which the
schedule replaces; `ids`, `people`, `prev_pitches`, `app_days` and `game_pitches` are per pitcher (code audit).

Rows, labels, the rival B0, the parameters, scales, z, scores, ranks, the rotation rule, the bootstrap and the model
test are the frozen code's. Ohtani's rows of the test season are left out by the frozen rule (`b == OHTANI and y ==
TEST`), which here removes his 2023 rows (his 2022-2023 pitches were read in an earlier project).

### Regression gate (runs first, in the same notebook, before any 2021-2023 file is opened)

The notebook first runs, on the frozen 2024-2026 MLB files and the frozen Triple-A 2023-2026 kernel output, (a) the
frozen sixth-family code with **no** replacement and (b) the same code with the MLB-only replacements above (seasons
unchanged). Both read the StatsAPI minutes apart. (b) must equal (a) exactly: MLB rows, pitchers, usable rows and
positives, arm positives, the Triple-A first and last dates (schedule in (b), Triple-A Statcast in (a)), MLB 2026
AUC(B0), and for S1, S2 and their fastball-gap-only scores the AUC, the rotation-floor maximum, the Bonferroni lower
bound and the positives (floats to 1e-12). Rows, pitchers, usable rows and the Triple-A dates do not depend on the
labels and are compared in the code test too. This shows that removing Triple-A pitches, the schedule dates and the
generalised history and game-log year expressions change nothing on the MLB side. The confirmation-only constants
(row seasons, TRAIN/TEST, md5, league-scale season 2022) cannot be run on 2024-2026; the season gates below and the
frozen gates cover them. If the gate fails, the notebook stops before the confirmation run.

Why not against the frozen run's printed values: a pre-freeze run of the first version of this gate (GHA
`38475417132`, about five hours after the frozen run) reproduced the frozen MLB rows, usable rows, positives, B0 (to
16 digits) and S1 (AUC, floor maximum and Bonferroni bound to 16 digits) but not S2: arm positives 1,042 against 1,045,
AUC 0.53186 against 0.53276. Two StatsAPI fetches minutes apart then returned the same 10,702 placements in the same
order with the same descriptions, but 15 placement groups of 2024-2026 (same pitcher, same effective date) disagree on
the arm flag, and the frozen merge keeps the one the API lists first; reversing the input order flips the arm flag of
those 15. The frozen code's arm labels can therefore move between runs hours apart. The notebook prints (a) against
the frozen run's values as drift; drift is reported, not gated. The same tie rule applies, unchanged, to 2021-2023.
A pre-freeze run of this gate (section 1 only, GHA `38480274516`, no 2021-2023 file attached) passed: (a) and (b)
were equal on every number listed above; against the frozen run's values the drift was zero for rows, positives,
B0 and S1, and for S2 -0.000892 in AUC and -3 arm positives (the fastball-only scores differ from the frozen
printout by its four-decimal rounding, and S2 fastball-only by -3 positives). Its printed 2024-2026 values were read.

### Season gates (both runs, label-free)

Placement years read include the two seasons before the first row season; the rows' seasons are exactly the row
seasons; at least 99% of the Statcast appearance days of each season are in the StatsAPI MLB game logs; every row
season (and the season before, except 2020) has pitchers with Triple-A game-log days. The frozen gates (md5,
duplicates, fill, convention, look-ahead plant, plain loop, selection, league scales, rotation share) apply unchanged.

## Threats stated in advance

- **Sticky-stuff enforcement, June 2021.** League four-seam spin on single days: 2,326 rpm on 20 May, 2,256 on 8 June,
  2,244 on 25 June 2021. A pitcher-level step in spin, possibly in the gap; the in-season centre (median up to t - 30
  days) lags it, which can inflate z from late June 2021. Enforcement may also have changed injury risk. Reported:
  C1 without 2021 rows from 3 June, and 2022-2023 alone (about 1,280 arm pre-IL rows).
- Pitcher IL minimum 10 days in 2021, 15 days from 2022; 60-day lists were more common in 2021.
- Injury text missing on 19% / 18% of 2021 / 2022 MLB placements (mostly unnamed); arm labels need text, so the arm
  target is less affected than "all placements".
- 2023 pitch clock; possible Hawk-Eye recalibrations between seasons (the convention gate prints per season).
- The rival B0 is trained on 2021-2022 only in the reported model test.
- COVID-19 injured-list placements match the frozen placement rule; they carry no arm text, so they are dropped from
  the arm target, and the frozen merge (placements within 10 days collapse to the first) can absorb an arm placement
  that follows one. The label-free count found 0 (2021), 1 (2022) and 0 (2023) such MLB placements.
- The frozen roster filter keeps pitchers by today's StatsAPI `primaryPosition` (P or TWP), a look-ahead that is a
  little larger for 2021-2023 than for 2024-2026.
- The C1 line printed by the frozen test function ends with that function's own verdict (Bonferroni bound and all
  200 floors, the six families' rule); C1's verdict is the next line.

## Power (label-free counts)

Placements with injury text and at least one pre-IL MLB row (appearance dates only; the counter reproduces the frozen
2024-2026 count of 3,553 pre-IL rows exactly): 2021 1,909 rows (arm 693), 2022 1,502 (640), 2023 1,254 (642) = 4,665
(arm 1,975), against 2024-2026 3,553 (arm 2,048). Expected usable arm positives about 1,050 (frozen S2 fastball-only
1,092). Bootstrap SE about 0.017, so the 95% half-width is about 0.034. If the 2024-2026 excess over its floor mean
(+0.035) were real, the power to clear both conditions is about 77%; at half that effect about 27% (audit estimate;
the floor sd is inferred, not measured, and from the frozen rule; the whole-group floor is wider, by about the ratio
of the z sds above, 1.39 / 1.03, and it includes the shifts near 0, which carry part of any real effect and so raise
its mean under a true effect; both lower the power below these figures, which were not recomputed).

## Disclosures (before any 2021-2023 pitch feature met a label)

- Read: the complete printed output of the six frozen pitcher runs (2024-2026 and Triple-A 2023-2026) and of the
  pre-freeze regression checks (2024-2026 only).
- Read for 2021-2023, without pitch features: StatsAPI placement counts by year, level, list and arm (above), MLB
  appearance dates (to count pre-IL rows), the MLB season ends; 17 single days of 2020-2024 Savant pitches read without
  any label (the probe above).
- Read in earlier projects: Triple-A 2023 pitches, MLB and Triple-A 2023 game logs, 2022-2023 placements as history
  counts in B0 (not joined to 2021-2023 features), MLB 2023 season aggregates without spin or IL, Ohtani's 2022-2023
  pitches.
- The code test and the floor diagnostics (both below the rule) read the 2021-2023 pitch features of a sample of
  pitchers with random placements only; the code test also printed that sample's label-free score by position decile
  (0.508 in the first tenth of a pitcher-season to 0.484 in the last).

## Code test

Run on Raspberry Pi 5 in a virtual environment with Kaggle's package versions (numpy 2.1.3, pandas 2.3.3,
scikit-learn 1.6.1), random placements from the frozen generator (seed 99), on two 100-pitcher samples: the frozen
2024-2026 sample and 100 pitchers drawn with seed 0 from those present in all three of 2021-2023 (with the frozen
Triple-A 2023-2026 sample files). Notebook md5 `4c51bcd69ed95212adb89d348f555ff6`: exit 0.

- Section 1: reference 0, regression 14, confirmation 17 replacements; the label-free parts of the gate agreed; Statcast
  days in the StatsAPI game logs 100% in every season; last Triple-A game-log day equal to the schedule's in every
  season (2021 2021-10-03, 2022 2022-09-28, 2023 2023-09-24).
- C1 under random labels: AUC 0.6075, 95% lower 0.4885, whole-group p 0.0180 (floor mean 0.5121, 97.5th percentile
  0.5971), so condition 2 was met by a random label and condition 1 was not: NOT SUPPORTED. This seed is the one that
  started the floor diagnostic; over 1,000 redraws the whole-group rule rejected 2.5% (table above). Floor structure:
  8,900 group draws, 31.9% of shifts in the band the frozen rule excludes, shift 0 drawn 466 times. Position checks:
  labels' position AUC 0.5064 (p 0.40); score minus its decile mean 0.6058 (p 0.018).
- Before the floor change, seeds 99 to 102 gave C1 ranks 200, 4, 40 and 199 among the frozen rule's 200 rotations
  (above).

Planted defects: each is the notebook with one replacement (`conf_mut.py`); each must stop with its own message and not
exit 0, checked against the mutant's md5 in the first line of its log and the log being newer than the mutant. All 8
stopped as intended:

| mutant | replacement | stopped at |
|---|---|---|
| k1 | placement history from 2022 instead of two seasons before the rows | placement history does not start two seasons before the rows |
| k2 | game logs without the first row season | Statcast days and MLB game logs disagree |
| k3 | bootstrap seed 3 | the bootstrap is not the frozen one |
| k4 | league-scale season left at 2025 | a league-typical scale is missing |
| k5 | C1's floor not rotated | C1's floor cannot be beaten by a planted signal |
| k6 | C1's groups by pitcher instead of pitcher-season | C1's groups are not pitcher-seasons in time order |
| k7 | C1's floor from the frozen rule | C1's floor is not the whole-group rotation |
| k8 | C1's shift drawn from 1..n-1 | shift 0 is never drawn |

Audits: an opus code audit of the notebook before the floor change (no BLOCKER or SERIOUS; minor points applied), one
of the floor change (header, validity wording, structural asserts, the position checks, power note: applied) and one of
those fixes (two lenient floors passed every assert: the group and floor-equality asserts and mutants k6 to k8 added;
wording applied).

## Not done

No 2021-2023 pitch feature was compared with any placement before this file is on main. The notebook is run once. No
change after the run except a dated amendment that says whether it came before or after any result was printed. The
six frozen pitcher studies' verdicts are not revisited.

## AMENDMENTS
