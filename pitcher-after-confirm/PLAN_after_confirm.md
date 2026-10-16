# Plan (draft v1): confirm the last-outing P1 on unseen MLB 2021-2023

## Why
The eighth pitcher study (frozen `5bbe6ed`, notebook md5 a96dcbef) gave P1 SUPPORTED on MLB 2024-2026:
the daily within-role rank of A (primary-fastball velocity of the day's outing minus the mean of the last 60
of that type, on max(own MAD, 0.5 x league scale), sign so that slower = higher) for the last appearance 1-14
days before an arm placement, outings without an early hook: AUC 0.6059 (Bonferroni lower 0.5629), whole-group
rotation floor mean 0.533, p 0.001; position-adjusted 0.6065. Independent rebuild matched. Its FREEZE says
"MLB 2021-2023 can later confirm P1 (A only): no outing-day velocity has met a label there."

## Data
- Existing private dataset `yasunorim/mlb-statcast-2021-2023-pitch-subset` (23 columns; md5s in
  FREEZE_pitch_ssw_confirm.md) lacks `inning_topbot`, `bat_score`, `post_bat_score`, which the frozen notebook reads
  (exit flags, runs; early hook needs mid-inning exit and left-mid-at-bat).
- Do not version that dataset (the sixth-family confirmation pins its md5s). Build a NEW private dataset
  `mlb-statcast-2021-2023-outing-columns` with key columns (game_pk, at_bat_number, pitch_number) plus pitcher,
  game_date, inning_topbot, bat_score, post_bat_score, by a new `build.py` that imports the same dataset4
  fetch/parse functions (one Savant CSV per day, regular season). Gates: key set identical to the 23-column subset
  (exact, both directions; same pitcher and game_date on every key); inning_topbot in {Top, Bot}; scores integers
  >= 0, post >= bat; per-day row counts equal the subset's. Nothing about placements is read. Precedent: the
  Triple-A location columns (PR #45) joined by key with a zone-agreement gate.

## Notebook (pitcher-after-confirm)
- Fetch the frozen after notebook at 5bbe6ed (md5 a96dcbef). Same replacement pattern as pitcher-ssw-confirm:
  apply the 2021-2023 replacements of the frozen base used by pitcher-ssw-confirm (seasons, MLB md5, no Triple-A,
  history years, game-log years, ...), and in the after notebook: ROW_SEASONS/TRAIN/TEST, `load_ctx` for MLB joins
  the outing columns by key (assert one-to-one and complete), Triple-A context and `attach_after(rows_a)` removed
  (no Triple-A pitches for 2021), equality gates replaced by printed counts.
- League scale: A's definition uses the league scale measured on MLB 2025 (relievers/starters). The confirm run
  uses the exact floats measured by the regression run of the same session on the frozen 2024-2026 files (so A is
  the frozen function with frozen constants); reported alternative: the 2022 scale.
- Previous-season mode: 2021 rows have no 2020 pitches loaded -> choice -1 when no pitch this season yet, as 2024
  rows were in the frozen run.
- Section 1, regression gate (same session, before any 2021-2023 row meets a label): the replaced code in
  "regression" mode on the frozen 2024-2026 files must reproduce the frozen run exactly: rows/labels tuple, B0 AUC
  0.5644912856858865, P1 AUC 0.6059334572407536, positives 314, rows 30,492, position-adjusted 0.6065369360478149,
  game-log left out 1, and the frozen floor's p.
- Section 2, confirmation run on 2021-2023.

## The one primary (C2)
P1's statistic on MLB 2021-2023 (same target, same early-hook rule, same score): SUPPORTED iff
1. pitcher bootstrap 95% lower bound (2.5th percentile, 2,000 draws, seed 2) > 0.5, and
2. whole-group rotation p <= 0.025 (1,000 rotations, seeds 5000-5999, p = (1 + #{floor >= AUC}) / 1001), and
3. both also hold for the position-adjusted rank (apps_season deciles by role, label-free means).
Floor diagnostic before freezing: random placements redrawn 1,000 times on the code-test sample / full data
(label-free), rejection rate at 0.025 must be near nominal (as for the sixth-family confirmation).

## Reported only
Effective the next day vs 2-14 days later (the 2024-2026 descriptive split 0.629 / 0.552); starters/relievers;
each season; all outings (manipulation check); the 2022 league scale; model over B0+ in 2023 (TRAIN 2021-2022).

## Power
2021-2023 arm placements with an MLB appearance in 14 days ~ (from transactions, label counts only); expect ~250-350
positives after the rules. With 300 positives the 95% half-width is near 0.035, so an AUC of 0.606 would clear;
an AUC of 0.55 would be borderline.

## Threats / disclosures
- Already read from 2021-2023: placement dates and gaps, the spin-axis-gap confirmation vs labels (C1), a 2023 model
  test with before-day velocity z, the position AUC (0.659). No outing-day velocity has met a label there.
- Velocity calibration changed across seasons (Hawk-Eye from 2020); the daily rank within day and role absorbs
  league-wide shifts.
- 2021 had a shortened spring and many IL placements (COVID IL is non-arm, left out by the frozen arm rule).

## v2 (after the opus plan audit, each item checked in the code)
1. BLOCKER fixed: `LEAGUE[level]` feeds two constants: the league scale `lsc` (A) AND the league median pitch count by
   start flag `lg` (A:267, the shortfall fallback -> early hook -> P1 mask). Both are passed as exact values from the
   regression run of the same session (2024-2026 files, season 2025), and asserted finite. Reported alternative: both
   measured on 2022 (stated now: cannot change the verdict).
2. Data: the new dataset holds all 23 subset columns + the 3 (builder runs the subset's builder unchanged with three
   more kept columns; its md5 and dataset4's md5 593784729763616cdb2519cd9ff7ee5a are asserted). In the notebook:
   every key of the new files equals the pinned subset's and every one of the 23 columns is equal key by key
   (exact, NaN = NaN). Structure gates in the builder: topbot one per at-bat; bat_score never decreasing within a
   half; post_bat_score = next pitch's bat_score within a half (share measured on a sample before the threshold is
   fixed). New md5s pinned in the notebook and FREEZE.
3. The regression run goes through the same join: the frozen 2024-2026 files are split into the 23 columns and the
   3 outing columns and joined by the same function before the frozen code runs; it must reproduce 0.6059334572407536.
4. Power reframed: the floor sits above 0.5 (2024-2026: mean 0.533, 97.5th 0.559; seasons 0.507-0.552), so the
   rotation condition binds; ~330-370 positives expected (label-count ratio 2021-23 / 2024-26 ~1.17); AUC SE ~0.017;
   rough power ~1.0 at 0.606, ~0.9 at 0.58, ~0.5 at 0.56, ~0.3 at 0.55.
5. Floor diagnostic: random placements on 2021-2023 rows, label-free, drawn at realistic positions (position template
   of the 2024-2026 real positives: the share of the pitcher-season elapsed and the gap to the next appearance), 1,000
   redraws, rejection rate at p <= 0.025; the real 2021-2023 labels are never built in the diagnostic.
6. Replacement list rebuilt against the raw base and the after notebook (audit's line list); MLB-only asserts kept
   (choice/FB-velocity 100%, previous outing equality, exit-rate plausibility, game-log <= 10%).
7. Frozen seeds kept (7000-7999 for the rotation floor) so the code is identical; the regression gate checks the floor
   mean 0.5334 and 97.5th 0.5587 to the printed digits, not p.
8. The two conditions are a conjunction; no multiplicity. Condition 1 uses the 2.5th percentile of the frozen
   bootstrap (seed 2, 2,000 draws).
9. Disclose: the 2023 model with before-day velocity z (M2-B0 -0.004) and the label position AUC were read from
   2021-2023. 2021 seven-inning doubleheaders and the extra-inning runner may shift the early-hook rate (printed).

## v3 (2026-10-15, before any output of the full diagnostic was read)

From the opus code audit: gates added (the 95% bound against the frozen `boot()`; planted +/-0.3 signals must make the
verdict reachable only when both lines carry it; the outing columns' builder gates re-asserted in the notebook; the
confirm-only base replacements declared exactly; the verdict line labelled in fake mode; `game_log_out` dropped from
FROZEN as never asserted). Planted defects c9-c14 added. The plan's promise to print both 2022 constants is withdrawn:
only the 2022 median pitch count is printed (the 2022 velocity scale is not computed in the confirmation).

The diagnostic now tests the conjunction (A_rank and A_rank_pos) as C2 does, and a drawn positive inside another
positive's left-out window is left out. **Decision rule fixed now:** if, among random positives placed like the real
2024-2026 positives, the share passing C2 (both lines) is at most 0.05, C2 is frozen with the rule as written. If it is
above 0.05, C2 is not frozen with this floor: the floor is redesigned to match positions (rotation within the template
bins) and the diagnostic is run again on the redesigned rule before any freeze. (The first diagnostic run, GHA
38703478080, tests only A_rank on the pre-audit notebook; it is read for the real-label regression gate and the
template only.)

## v4 (2026-10-15 19:20 JST; after reading the first diagnostic, before any 2021-2023 outing-day feature met a label)

**What the first diagnostic showed** (GHA 38703478080, pre-audit notebook md5 bd0c206b, A_rank line only): on
2021-2023 rows with random placements, 1,000 draws of random positives placed in the template bins of the real
2024-2026 positives (fifth of the pitcher-season x days to his next MLB row: <=7, 8-30, later, none) gave AUC mean
0.5632 against a whole-group rotation floor mean 0.5205, and passed the A_rank line (p <= 0.025 and lower bound > 0.5)
in 0.951 of draws. The template: 0% of real positives but 97.3% of negatives have a next row within 7 days; 42.0% of
positives are the pitcher's last row of the season. On 2024-2026 (labels already read), the last outing before a
NON-arm placement had AUC 0.5569 (arm P1 0.6059). So the frozen P1 floor did not control for "the last outing before
a long absence"; much of P1's 0.606 is that, and only the excess over it is a candidate for an arm-specific signal.
This does not change the frozen verdict of FREEZE_pitch_after.md (its rule was met); it is disclosed in the README with
the confirmation.

**C2 redesigned (rule v3 above: share passing > 0.05 -> position-matched floor):**
- Same rows, target and scores (A_rank, A_rank_pos) as the frozen P1, on 2021-2023.
- Floor: bin-matched label draws. Bins as in the diagnostic (positions from all MLB rows of the pitcher-season:
  fifth of the season by row order x class of days to his next row). Each draw (seeds 7000-7999) picks, in every bin,
  as many rows as there are real positives in that bin, uniformly without replacement from the C2 rows of that bin,
  and takes them as positives against all other C2 rows. p = (1 + #{draws with AUC >= real AUC}) / 1001.
- **SUPPORTED** iff p <= 0.025 on both lines (A_rank and A_rank_pos). The frozen rule's numbers (rotation floor,
  bootstrap bound) are reported, not tested.
- Reported (labels already read): the same bin-matched test of the frozen P1 on 2024-2026, in the regression section.
- Diagnostic 2 (random placements only, 2021-2023): (a) random positives in the template bins -> share passing the
  new rule (must be near 0.025, at most 0.05, else the floor is wrong); (b) the same with a planted shift of +0.05 in
  the drawn positives' scores -> share passing (power at an arm-specific excess near 0.04-0.05 AUC). The old rule's
  pass share is printed beside them.

## v5 (2026-10-15, after the opus audit of v4; still before any 2021-2023 outing-day feature met a label)

- **Bins:** fifth of the pitcher-season by row order x class of days to his next MLB row: <=7, 8-15, 16-30, 31-60, >60,
  none with the row more than 13 days before the season's last regular-season MLB date (none-early), none within the
  last 14 days (none-late). Bins are merged before any score is used: while some bin with real positives k has fewer
  than 4k eligible C2 rows, the smallest such bin (by fifth*7 + class) joins the next shorter class of the same fifth
  (none-late -> none-early -> >60 -> 31-60 -> 16-30 -> 8-15 -> <=7), and a <=7 bin joins the <=7 bin of the previous
  fifth (the next one for the first fifth); stop when no bin violates or one bin is left. The table of eligible rows and
  positives per merged bin is printed before any score is compared.
- **Floor 2 (new):** exact stratified label permutation within merged bins: in each, as many rows as real positives,
  uniformly without replacement from all of that bin's C2 rows (real positives included), 1,000 draws (seeds
  7000-7999), p = (1 + #{draw AUC >= real}) / 1001.
- **SUPPORTED iff, on both lines (A_rank, A_rank_pos), the frozen whole-group rotation p <= 0.025 AND the bin-matched
  p <= 0.025.** The bootstrap lower bound is reported, not tested. Reported, not tested: a within-bin stratified
  Mann-Whitney (permutation p over the same draws); arm vs non-arm last outings.
- **Diagnostic 2 (fixed before it runs):**
  (a) on 2021-2023 with random placements: pseudo-positives drawn finer than the floor's bins (for each of 367 targets
  drawn from the 2024-2026 real positives' (position, days to next row, season-end) values, a random one of the 5
  eligible rows with the same days-to-next class key (exact days up to 120, else the none-early/none-late flag) nearest
  in position), left-out windows as the target does; share passing the new rule must be at most 0.05.
  (b) 2024-2026 last outings before a NON-arm placement (labels already read) under the new rule: if they pass, the
  floor does not control "last outing before a long absence" and C2 is not frozen.
  (c) power: the (a) draws with +0.05 added to the pseudo-positives' scores (both lines); the AUC excess it makes and
  the share passing are printed.
  If (a) > 0.05 or (b) passes, C2 is redesigned again before any freeze.
- Fake-mode gates added: a pure position score (the merged bin's key) must give bin-matched p = 1 exactly; every draw
  must hold the real positive count of every bin; no merged bin may violate the 4x rule unless only one is left.

## Erratum (2026-10-16 01:00 JST; found by the opus audit of the FREEZE draft, before diagnostic 2 was read)

v4 above misread the first diagnostic's template table (bin = fifth x 4 + gap class; class 0 = <=7 days, 3 = none):
0.3% of real positives (bin 16, 1 of 314), not 0%, and 96.6% of negatives, not 97.3%, had a next row within 7 days;
44.0% of positives (bins 15 and 19), not 42.0%, were the last row of the season. No decision used these three numbers
(v4's redesign followed the 0.951 pass rate); FREEZE and the README disclosure carry the corrected values.

## Diagnostic 2 result (2026-10-16 02:15 JST read; Kaggle kernel COMPLETE, GHA 38732450611 success; RPi5 diag2_out/log.txt)

- Regression section reproduced the frozen run (rows/labels (61966, 1180, 53384, 3553), AUC(B0) 0.5644912856858865,
  P1 0.6059334572407536). Arm P1 2024-2026 under the v5 rule: rotation p 0.0010 / 0.0010; position-matched floor mean
  0.5234 / 0.5210 (97.5th 0.5530 / 0.5510), p 0.0010 / 0.0010 -> passes. Template: 314 positives; none-late 0.000,
  none-early 0.439, median gap 35.
- (b) NON-arm last outing 2024-2026: 252 positives, AUC 0.5569 / 0.5580; rotation p 0.0060 / 0.0070; position-matched
  floor mean 0.5206 / 0.5193, p 0.0180 / 0.0150 -> PASSES.
- (a) 1,000 draws on 2021-2023 random placements, pseudo-positives mean 236 (exact gap key, nearest position): AUC
  0.5578 / 0.5585 against a position-matched floor mean 0.5440 / 0.5433; share passing the v5 rule 0.133 (rotation
  floors only 0.802). (c) +0.05 on the pseudo-positives: AUC excess +0.0474 / +0.0474, passes 0.865.
- **By the rule fixed in v5: (a) 0.133 > 0.05 and (b) passes -> C2 is NOT frozen; redesign.**
- Reading: the v5 bins pool gaps within a class (8-15, 16-30, ...); pseudo-positives that match the exact gap and
  position beat a floor that only matches the class by +0.014 AUC on average. So the floor still leaks "how long the
  absence after this outing is". (b) cannot be read as "injury in general slows the last outing" while the floor leaks.

## v6 draft (2026-10-16, after diagnostic 2; still before any 2021-2023 outing-day feature met a label)

1. **Floor M (matched sets, replaces the v5 bins):** for each real positive i, S_i = i plus the 9 eligible C2 rows of the
   same role (start flag of the row) and the same exact gap key (days to his next MLB row capped at 120; none-early;
   none-late), nearest in fraction of the pitcher-season (stable ties), outside every positive's left-out window. A draw
   replaces each positive by a uniform member of its set (sequential redraw on collision); AUC on the C2 rows with the
   drawn labels; 1,000 draws (seeds 7000-7999); p = (1 + #{draw >= AUC}) / 1001. The frozen rotation floor stays a
   conjunct. SUPPORTED iff both floors p <= 0.025 on both lines.
2. **Null check that is not the floor's own construction (diagnostic 3a):** pseudo-positives placed by the real
   positives' joint template on variables the floor does NOT match, in addition to (role, exact gap, position): the
   outing's pitch count relative to his usual, month, season, apps_season decile (all label-free). Share passing must
   be <= 0.05. A second variant uses pseudo-positives = last rows before real 2021-2023 gaps that end in a return to MLB
   (no IL placement read), sampled to the template's gap distribution.
3. **Non-arm rival, reading fixed now:** the 2024-2026 non-arm last outing under floor M is reported. Arm-specific
   wording is used only if the arm-minus-non-arm excess over floor M (2021-2023, arm vs non-arm matched sets, bootstrap
   over pitchers) has a 95% interval above 0; otherwise a SUPPORTED C2 is reported as "slower than outings before
   absences of the same length", not as arm-specific.
4. **Power (3c):** +0.05 on pseudo-positives under floor M; the 2024-2026 arm P1 under floor M (labels already read).
5. Diagnostic 3 runs before any freeze; if 3a > 0.05, redesign again.

## Audit of the v6 draft (opus, 2026-10-16 02:40 JST) and what I checked myself

- BLOCKER, confirmed in the code: in the real run the 0 rows are `use14 & ~il14` (`after/pitcher_after_il.py:369`), so
  every row before any placement except the arm positive is NaN, and rows with a Triple-A outing within 14 days are
  dropped (`pitcher_full.py:653` `use14 = ok14 & ~other14`). Long-gap 0 rows are therefore rare: diagnostic 2's table
  (`diag2_out/log.txt:80-82`) shows 68-439 rows in the long-gap bins and the last fifth's 184 positives merged into the
  <=7 bin (bin 28, 3,950 rows). **v5's floor compared 59% of the positives with rows followed by an outing within 7
  days**; v6's 9-row exact-gap sets would mostly not exist.
- BLOCKER, confirmed: with fake labels the real IL rows stay 0-eligible (`after_confirm_diag.py:106-108` uses `use14`
  from the fake placements), so fake-mode pools differ from the real run's; diagnostic 2's 0.133 does not describe the
  real floor.
- BLOCKER (disclosure): diagnostic 2(a) placed pseudo-positives on 2021-2023 rows with long exact gaps, most of which
  are real IL last outings, so its AUC 0.558 carries some of the 2021-2023 IL signal (all causes mixed, arm not
  separated). Disclosed in FREEZE; "no 2021-2023 outing-day feature has met an arm label" stays true, "met any label"
  does not.
- SERIOUS, accepted: exact gap is post-treatment; overlapping sets with sequential redraw are not an exact permutation;
  left-out windows asymmetric; 3a's same-day pitch count is a mediator; non-arm rival statistic undefined (power about
  0.4-0.5 for a 0.05 arm-minus-non-arm difference); q uses the season's row count (post-treatment).

## v7 (2026-10-16, replaces the v6 draft; still before any 2021-2023 outing-day feature met an arm label)

1. **Pool (label-free):** every MLB row (ok14, no early hook, A and A_rank_pos finite) whose next MLB row is 8 or more
   days later or absent in that season, whatever the cause: arm IL, non-arm IL, no IL, demotion (Triple-A outings
   allowed). Positives = the frozen P1 positives that are in the pool (count of positives with gap <= 7 printed and
   left out; 1 of 314 in 2024-2026). All other pool rows are 0 (non-arm IL last outings included). Rows before an arm
   placement 1-14 days later that are not the last row are NaN as frozen.
2. **Strata (disjoint, label-free):** role (start flag) x calendar fifth of the regular season (day_of_season over the
   season length, not row order) x gap class (8-15, 16-30, 31-60, >60, none). Merged before any score by a fixed rule
   using only counts (a stratum with k positives and fewer than 4k pool rows joins the next shorter gap class of the same
   role and fifth; 8-15 joins the same class of the previous fifth); the merged table is printed and asserted before A
   is read.
3. **Statistic:** stratified AUC = sum over strata of (positives in s) x AUC_s / positives (A_rank and A_rank_pos lines).
   **Floor P:** exact stratified label permutation within merged strata, 1,000 draws (seeds 7000-7999), p over 1001.
   **Conjunct:** the frozen whole-group rotation floor on the frozen C2 rows (frozen statistic). SUPPORTED iff both
   p <= 0.025 on both lines.
4. **Within-class gap residual check (2024-2026 only, labels already read):** among pool 0 rows, the Spearman of A_rank
   with exact gap inside each stratum, and the stratified AUC of a pure "exact gap" score; if the gap score passes
   floor P at p <= 0.025, the classes are refined before freezing.
5. **Validity on 2024-2026 (labels read):** under floor P, the arm P1 and, as pseudo-positives, (i) non-arm IL last
   outings (removed from the 0 rows), (ii) last outings before a demotion with a Triple-A outing within 14 days, (iii)
   covariate-only scores (rest days, 7- and 14-day pitch load, n_prev_il, n_prev_arm, days_since_act, number of primary
   fastballs in the window) as the score with arm positives; each printed. (iii) passing means positives differ on
   that covariate within strata; A's correlation with it within strata is printed beside it.
6. **2021-2023 before the freeze:** fake mode only, positives drawn uniformly from the pool (the pool does not depend on
   labels, so fake and real pools are the same); the share passing must be <= 0.05 (code test of exactness).
7. **Secondary, arm-specific (reported, fixed now):** pool limited to IL last outings (COVID-related placements left
   out), arm vs non-arm, the same stratified AUC and floor; "arm-specific" is written only if its p <= 0.025 on both
   lines. Power about 0.4-0.5 is stated.
8. **Power:** planted shift calibrated to the 2024-2026 arm excess over floor P (and half of it), at the 2021-2023 pool
   size.

## v7.1 (2026-10-16 03:10 JST; after the opus audit of v7, each item checked in the code; before any 2021-2023 outing-day feature met an arm label)

Checked: arm needs injury text (`pitcher_full.py:167-170`, no-text placements have arm False); role is the pre-day start
share (`:517`) and A_rank ranks within (day, role) (`:449-450`); `ok14` equals "not within the last 14 days", so the
none class is always none-early (`:581` vs `pitcher_after_confirm.py:261`); FAKE replaces `il_lab` by random
placements (`:611-620`) while `rule_days`/`other14` come from the real transactions.

1. **Pool:** `ok14 & ~other14` (the same filter as the positives), no early hook, A_rank and A_rank_pos finite, next MLB
   row >= 8 days later or none. **Every row with an arm placement 1-14 days later (or 0) that is not a positive is NaN**
   (gap 0, failed use14, drop rules, non-last rows) and so are rows before a no-text placement within 14 days (arm
   unknown). The pool's 0 rows therefore depend on the placements; fake mode goes through the frozen FAKE path and the
   frozen `last_outing_target`, so its pool is built by the same code as the real run's.
2. **Role** = the frozen `role`. **Strata** = role x calendar fifth (day_of_season / season length) x gap class
   (8-15, 16-30, 31-60, >60, none). **Merge** (counts only, positives included in the row count): while a stratum with k
   positives has fewer than 4k rows, take the smallest violating key (role, class, fifth) and merge it with the adjacent
   fifth of the same role and class (the next one; the previous one for the last fifth of that chain); when a class of a
   role is one stratum and still violates, it joins the next shorter class of the same role (8-15 joins 16-30); never
   across roles; stop when none violates or a role has one stratum. Printed and asserted before A is read.
3. Statistic, floor P and the rotation conjunct as v7 (an intersection-union test; size at most 0.025).
4. **Null check for residual position (decides the freeze):** pseudo-positives placed on pool rows with NO placement of
   any kind within 14 days, by the 2024-2026 positives' template (exact gap key, calendar fifth, role; 1 of the 5
   nearest in calendar position), count = the real positive count, 1,000 draws, each tested by the full rule (200
   rotations, 200 stratified draws, p over 201): on 2024-2026 AND on 2021-2023 (only no-placement rows are used, so no
   A meets a placement). Each share passing must be <= 0.05, else redesign.
5. **Covariates (decides the strata):** on 2024-2026, for rest days, 7- and 14-day pitch load, n_prev_il, n_prev_arm,
   days_since_act and number of primary fastballs in the window, the covariate as a score under floor P, and its
   within-stratum Spearman with A_rank among 0 rows. If p <= 0.025 and |Spearman| >= 0.05, its tercile (cut on the pool
   of that season set) is added to the strata before freezing. Non-arm IL and demotion last outings as pseudo-positives
   are printed (reported; their reading is item 7).
6. 2021-2023 fake mode is a code test only.
7. **Secondary (arm-specific):** pool limited to last outings before text placements, COVID-related placements left out
   (description contains "COVID"), arm vs non-arm, same stratified AUC and floor; "arm-specific" only if p <= 0.025 on
   both lines; power about 0.4-0.5 stated.
8. Power: shift calibrated to the 2024-2026 arm excess over floor P and half of it, at the 2021-2023 pool size; the
   2021 count of positives with gap <= 7 is printed.

### v7.1 implementation notes (2026-10-16 04:00 JST, before diagnostic 3 runs)

- "Number of primary fastballs in the window" is not a row column; `own_scale` (the pitcher's own velocity scale, which
  reflects how precise A is) is used in its place. The exact gap (capped 120, none = 121) is tested as an item-5
  covariate. Covariates are tested two-sided (p over 1001 by distance from the floor mean).
- Diagnostic 3 decision: C2 may be frozen only if item 4's share passing is at most 0.05 on BOTH 2024-2026 and
  2021-2023 AND no item-5 covariate is added; if a tercile is added, the strata change and diagnostic 3 is run again
  before any freeze.
- Item 4 detail: pseudo-positives take each template positive once (random order), the 5 unused no-placement pool rows
  of the same role and exact gap key nearest in calendar position (nearest key of the same kind if none), one at
  random; the template's days to placement sets the left-out window; the rotation is evaluated only when the pool
  floor passes (the rule needs both), so the share passing is exact.
- The 2021-2023 build in diagnostic 3 runs with stdout discarded; only rows with no placement within 14 days enter
  (asserted). Power: +0.02/+0.05/+0.1 on the pseudo-positives' scores (first 200 draws), beside the 2024-2026 arm
  excess over the pool floor.

## v7.2 (2026-10-16 05:15 JST; after the opus code audit and the fake-mode test, before diagnostic 3 ran on Kaggle)

- The fake-mode test stopped in the secondary: arm vs non-arm is about 1:1, so the 4x rule collapsed each role into one
  stratum and a role with arm rows only reached the statistic. **The secondary's merge rule is now "a stratum with arm
  rows needs at least one non-arm row"** (the primary keeps 4x); a stratum of one class only (possible only where a
  role is left as one stratum) is dropped from the statistic for the real labels and every draw alike, and counted.
- The non-arm last row is taken as the frozen arm rule does: the last row before the placement, then 1-14 days (it had
  filtered 1-14 days first, which moved the "last row" back when the true last row was on the placement day).
- NaN covers rows with ANY arm placement, and any placement without injury text, 0-14 days later (not only the next
  placement), in the primary pool and the secondary.
- Fake-mode gates: fake placements are random days the pitcher keeps pitching after, so few reach the pool; the gates
  use 30 random pool rows as the pool's positives. The position-only gate now also asserts that the plain AUC of the
  stratum score is the same in every draw (it fails if draws ignore the strata). Merge unit cases added (last fifth,
  two violators, the secondary rule).
- Disclosure: A_rank ranks within (day, role) over all rows, so a no-placement row's rank depends slightly on the A of
  placement rows on the same day (diagnostic 3's 2021-2023 null check).

## Diagnostic 3 result (2026-10-16 08:09 JST; Kaggle `yasunorim/pitcher-last-outing-confirmation-diagnostic-3`, GHA 38811390746 success, notebook d1289f4 md5 10199285)

Printed decision: "item 4 share passing 2024-2026 0.000, 2021-2023 0.000; item 5 terciles to add: ['_gap'] -> C2 is NOT
frozen: redesign". By the v7.1 implementation note, C2 is not frozen.

What the log also shows (read before any 2021-2023 arm label met A):

1. **The 4x merge left one stratum per role in the real run** (strata 0 and 100: relievers 161 positives in 761 rows,
   starters 152 in 844). The calendar fifths and the gap classes of v7.1 item 2 were all merged away, so the v7.2 pool
   floor controlled role only. The arm P1 2024-2026 under that floor: stratified AUC 0.5748 / 0.5752, floor mean 0.5011,
   p 0.001 / 0.001.
2. **Item 5:** the exact gap within those two strata has stratified AUC 0.8789 (p 0.001) and within-stratum Spearman
   with A_rank among 0 rows +0.0673 -> its tercile is added by the rule. The other covariates did not qualify
   (n_prev_il p 0.017 but Spearman -0.007; days_since_act p 0.021, Spearman +0.043).
3. **Item 4 did not test the region where the positives sit.** Of 313 template positives per draw, 222 were skipped
   (no unused no-placement pool row of that role and exact gap key) and only about 91 were placed; the template is 44%
   "none" with median gap 35 days. The share passing 0.000 therefore covers mostly short-gap positions.
4. **The pseudo-positives are not centred on the floor:** pool AUC 0.5105 (2024-2026) and 0.5388 (2021-2023) against
   floor means 0.4987 / 0.4989. In 2021-2023 a +0.05 shift passes 0.985 of draws, in 2024-2026 0.000 (about 106
   and 91 placed). The 2021-2023 residual (+0.040) is about half of the 2024-2026 arm excess over the pool floor (+0.0737).
5. Reported rivals (2024-2026): non-arm last outings as positives 235, stratified AUC 0.5391 (p 0.027); last outings
   before a Triple-A outing within 14 days 630, AUC 0.4534 (p 1.000); secondary arm vs non-arm 313 vs 235, AUC 0.5063
   (p 0.42).

Reading: v7's position control was absent in practice (point 1), the gap still separates positives from zeros almost
completely inside the strata that remained (point 2), and the null check could not place pseudo-positives at long gaps
(point 3), where its residual bias shows (point 4). Adding a gap tercile under the 4x rule would be merged away again,
because positives are 20% of the pool and sit at gaps the zeros rarely reach. Diagnostic 4 (counts only) sizes the
zeros by role x gap class x calendar fifth on 2024-2026 and the no-placement rows on 2021-2023, and the strata each
candidate merge rule would leave, before v8 fixes the rule.

## Diagnostic 4 result (2026-10-16 10:20 JST; Kaggle `yasunorim/pitcher-last-outing-confirmation-diagnostic-4`, GHA 38829424438 success, branch commit ece9a0b, notebook d1289f4; counts only, no A statistic)

Also: planted defects v7 (notebook 6afb7897, `conf_chain7.log`): code test exit 0, 30 of 30 "stopped as intended YES".

1. **No arm positive has a gap of 8-15 days** (an IL stint is at least 15 days for pitchers): 2024-2026 positives by
   class 16-30 77, 31-60 54, >60 44, none 138; gap median 112 (relievers) / 88.5 (starters).
2. **The pool zeros at gaps of 16 or more (329) are mostly the last outings before a non-arm text placement (233)**;
   the rest are 94 rows with no placement of any kind within 14 days, of which 63 are "none" (no later MLB row that
   season). The 963 zeros of class 8-15 hold almost all of the no-placement zeros (947 of 1,041).
3. With any merge rule that keeps the gap classes (zeros >= 5: 36 strata; zeros >= 1: 46 strata, all positives in a
   one-class stratum), the positives' comparators are therefore almost only non-arm IL last outings: the primary becomes
   the arm vs non-arm secondary (2024-2026: AUC 0.5063, p 0.42). The 4x rule keeps role only; 2x and "zeros >=
   positives" keep relievers' classes only.
4. 2021-2023 no-placement pool rows: 1,174, of which class >= 16: 112 (none 69).

**Reading (the reason v8 changes the design, not only the merge rule):** the gap after the outing is a consequence of
the label (an IL placement makes the gap at least 15 days), so item 5 was bound to select it, and stratifying by it
compares arm IL with other IL. Item 5's within-stratum Spearman of gap with A among "0 rows" (+0.067) was computed on
zeros of which the long-gap part is 71% non-arm IL last outings, so it mixes the IL signal itself into the "confounder"
check. The alternative v5-v7 meant to remove is "an outing followed by a long break is slow for reasons other than an
injury list"; that can be measured directly on rows with no placement of any kind, without any IL label.

## v8 (2026-10-16 10:40 JST; before any 2021-2023 arm label met A; to be audited before diagnostic 5)

1. **Labels:** 1 = the frozen P1 arm positive in the pool (unchanged); **0 = pool rows with no placement of any kind
   0-14 days later** (non-arm IL last outings and every other row with a placement within 14 days are NaN in the
   primary). Pool filter unchanged (use14, no early hook, both scores finite, next MLB row >= 8 days later or none).
2. **Strata = role x calendar fifth** (gap is not a stratum: it is a consequence of the label). Merge rule: a stratum
   with positives needs at least as many zeros as positives; violators merge with the adjacent fifth of the same role
   (as v7.1, class step removed); never across roles. With about 1,000 zeros against 313 positives no merge is expected;
   the merged table is printed and asserted before A is read.
3. Statistic, floor P (exact stratified permutation, 1,000 draws, seeds 7000-7999) and the frozen rotation conjunct as
   v7 (both lines, both floors, p <= 0.025).
4. **Gate G (the alternative, label-free; decides the freeze):** on rows with no placement of any kind within 14 days,
   pseudo-label 1 = next MLB row 16 or more days later or none, 0 = 8-15 days; score A_rank and A_rank_pos; strata role x
   fifth; one-sided floor P. Run on 2024-2026 AND 2021-2023 (no IL label is used; rows with any placement are not read).
   If either season set gives p <= 0.025 on either line, "a long break is slow without an injury list" is present and
   C2 is not frozen as a test of the injury list (redesign; the arm vs no-placement contrast cannot be read as IL).
   Also printed: the same with "none" only and with "16+ excluding none" (reported, not deciding).
5. **Item 4 (null calibration; decides the freeze):** pseudo-positives drawn from the zeros of 2024-2026 and of
   2021-2023, placed by the 2024-2026 positives' template on role x calendar position (the 5 nearest unused rows in
   frac of the same role, one at random; no gap key), count 313 or as many as can be placed (printed), 1,000 draws,
   the full rule; share passing <= 0.05 on both, and the mean pool AUC of the pseudo-positives minus the floor mean is
   printed (diagnostic 3's +0.040 on 2021-2023 under role-only strata is the reference).
6. **Item 5 (covariates):** rest, p7, p14, n_prev_il, n_prev_arm, days_since_act, own_scale, as v7.1 (p <= 0.025 and
   |within-stratum Spearman with A among zeros| >= 0.05 -> tercile joins the strata), with zeros now no-placement rows
   only. **Gap is not an item-5 covariate** (it is gate G). A covariate that is defined after the outing is not
   eligible.
7. Reported rivals (2024-2026 and, after the freeze, 2021-2023): non-arm last outings vs the same zeros; arm vs non-arm
   (the v7.2 secondary, kept; "arm-specific" only if p <= 0.025 on both lines); demotion last outings vs zeros.
8. Power at the 2021-2023 pool size: shifts +0.02/+0.05/+0.1 on the pseudo-positives' scores (item 4 draws), beside the
   2024-2026 arm excess over floor P under v8.
9. What C2 then claims if it passes: the last outing before an arm IL placement is slower than outings of the same role
   at the same point of the season that are followed by no placement, in 2021-2023, and (gate G) outings followed by a
   long break without a placement are not; it does not claim that the slowdown is arm-specific (that is item 7) or
   that it precedes the injury (the frozen framing: a public signal after the outing, before the announcement).

## v8.1 (2026-10-16 11:30 JST; after the opus audit of v8, each finding checked against the logs; before any 2021-2023 arm label met A)

**Corrections to the Diagnostic 3 / 4 sections above (they stay as written; read them through these):**
- Diagnostic 3 point 3 ("item 4 did not test the region where the positives sit") is wrong. Item 4 drew by exact gap key
  among no-placement pool rows (diag3 `NullCheck.draw`), so the placed count tracks the long-gap no-placement rows
  available: about 91 placed against 94 such rows in 2024-2026, about 106 against 112 in 2021-2023. **Item 4 was in
  effect the "long break without a placement" contrast** (calendar position matched, role-only strata): pool AUC 0.5105
  (2024-2026) and 0.5388 (2021-2023) against floor means 0.4987 / 0.4989. That contrast has been read on 2021-2023
  no-placement rows (allowed; no arm label) and is disclosed in the FREEZE.
- Diagnostic 4 point 1 holds for 2024-2026 only: in 2020-2021 the pitchers' IL minimum was 10 days (15 again from
  2022), so 2021 positives can have gaps of 11-15 days. Gap medians there code "none" as 121.
- v8 "Reading": gap stratification was invalid because the long-gap comparators were other IL rows, not because the
  gap is a post-label variable as such; among rows with no placement the gap is a legitimate control, only a thin one.

**v8.1 replaces v8 items 1-9:**

1. **Primary contrast (deciding):** 1 = the frozen P1 arm positive in the pool; 0 = a **clean long break**: a pool row
   whose next MLB row is at least L days later or none (L = 11 in 2021, 16 in 2022-2026; fixed by the IL minimum, not by
   data), with **no placement of any kind (any list, any level, with or without text) from the day of the row to the
   day before the next MLB row, or to the season's last MLB date for "none" rows**. Every other row is NaN. The question
   is: is the last outing before an arm IL placement slower than the last outing before a comparably long break with
   no IL placement?
2. **Strata:** role x kind (none / not none). Merge (counts only): a stratum with positives needs at least 5 zeros, else
   its two kinds merge within the role; never across roles; printed and asserted before A is read. No calendar strata:
   A_rank is the rank of A within (day, role) (`daily_rank`, `pitcher_full.py:449-452`, applied to A
   in `after/fakecheck.py:354`) and A_rank_pos subtracts its label-free mean by role x apps decile (`:454-462`), so their level does not drift with the
   calendar; the calendar fifths of 1 and 0 rows are printed (reported).
3. **Statistic:** stratified AUC on A_rank and A_rank_pos; **floor P** = exact stratified label permutation, 1,000 draws
   (seeds 7000-7999), p over 1001; **SUPPORTED iff p <= 0.025 on both lines**. The rotation conjunct is dropped for this
   contrast (it was built for the frozen all-rows P1; the zeros here are a few rows per pitcher).
4. **Power (printed in diagnostic 5, from 2024-2026 design data):** the 2024-2026 primary statistic and its excess over
   floor P, and the share passing when that excess and half of it are planted on 2021-2023-sized label sets drawn from
   2024-2026 (positives and zeros resampled to the 2021-2023 counts of zeros, which are label-free, and the 2024-2026
   positive count). The power is written into the FREEZE whatever it is; low power does not reopen the design.
5. **Pre-declared claims:** SUPPORTED -> "in 2021-2023 the last outing before an arm IL placement is slower than the
   last outing before a comparably long break with no IL placement". NOT SUPPORTED -> "not confirmed: in 2021-2023 the
   slowdown before an arm IL placement is not distinguished from the slowdown before a long break without one (power
   stated)". **C2 is not redesigned after diagnostic 5** except for a code fault (a gate that fails), which is fixed and
   disclosed without reading any 2021-2023 arm statistic.
6. **Reported (not deciding), 2021-2023 after the freeze and 2024-2026 in diagnostic 5:** (a) the v8 contrast, arm vs
   short-break no-placement rows in role x fifth strata, with the long-break-vs-short-break excess (the diagnostic-3
   contrast, rebuilt with the v8.1 clean-break rule and role x fifth strata) printed beside it and a bootstrap 95%
   interval of their difference; (b) non-arm IL last outings vs the same clean long breaks; (c) arm vs non-arm (the v7.2
   secondary); (d) among clean long breaks only, the within-stratum Spearman of exact gap (none = 121) with A_rank
   (label-free); (e) the count of long-break rows removed by the "no placement until the next MLB row" rule.
7. **Diagnostic 5 (before the freeze) reads only:** 2024-2026 everything in items 1-6; 2021-2023 counts of the clean
   long breaks per stratum and per season, and counts (no A) of 2021-2023 arm positives by gap class per season, to
   confirm the IL-minimum reading. No 2021-2023 A statistic is computed in diagnostic 5.
8. Unit gates (label-free, synthetic rows): L per season (2021 gap 11 is long, 2022 gap 15 is not), a placement on the
   day before the next MLB row removes the row, a "none" row with a placement after the row's day removes it, the
   merge rule, and the permutation keeping strata; planted defects rebuilt for v8.1 before the freeze.

## v8.2 (2026-10-16 12:00 JST; after the opus audit of v8.1, checked in the code; before any 2021-2023 arm label met A)

Amends v8.1:

- **Item 1 wording:** "no placement of any kind" -> **"no IL placement at either level"** (`il_by` holds IL lists only:
  7/10/15/60-day and full-season; options, DFA, release and the restricted list are not in it). Unrecorded injuries
  remain among the clean long breaks. Unit gate and assert: clean long breaks are a subset of `pool_base & ~il14`.
- **Item 3, decision rule:** **SUPPORTED iff, on both lines (A_rank, A_rank_pos), floor P on the v8.1 contrast AND the
  frozen whole-group rotation `p1_stat` (`pitcher_after_confirm.py:469`) both give p <= 0.025** (the frozen `decide`,
  `:626`; an intersection-union test, so adding the rotation conjunct can only lower the size). Reason: floor P permutes
  labels across pitchers and does not protect against pitcher traits, and the clean-break zeros are a different set of
  pitchers (fringe, released, demoted). Balance printed (reported): age if available, n_prev_il, own_scale, starts share.
- **Item 2, merge edge:** a merged stratum that still has fewer than 5 zeros (a role with one stratum left) is **kept**
  and counted; the same for the real labels and every draw. Reported sensitivity: stratified AUC with the positives
  reweighted to the zeros' calendar-fifth distribution within role x kind (late-season "none" zeros include demotions
  after the Triple-A season ends, which `other14` cannot catch).
- **Item 4, power:** from 2024-2026, permute labels within strata (the null), then plant +E and +E/2 on the positives'
  A_rank (E = the 2024-2026 excess of the contrast over floor P), at the 2021-2023 counts (zeros from item 7; positives
  from item 7's count), for the conjunction with the rotation floor; also a bootstrap of the real 2024-2026 contrast at
  those sizes without planting. Printed and written into the FREEZE; it does not reopen the design.
- **Item 7, leakage:** 2021-2023 positives are counted by gap class and season **before** the finite-A and early-hook
  filters (no A-derived column is printed; asserted by printing only counts from columns not derived from A). 2021
  positives with gap < 11 are counted and kept.
- **Disclosure in the FREEZE:** diagnostic 3 read the aggregate A of about 106 of the 112 2021-2023 long-gap
  no-placement rows (pool AUC 0.5388 against those rows as pseudo-positives, role-only strata); the cleaning rule (no IL
  placement until the next MLB row) was added after that read; it removes long breaks that end in an IL placement, which
  should make the zeros faster, i.e. it moves toward SUPPORTED. Reported sensitivity: the same contrast without the
  cleaning rule (zeros = long-gap pool rows with no IL placement within 14 days), beside the verdict.

## Diagnostic 5 result and the FREEZE audit (2026-10-16 15:30 JST; before any 2021-2023 arm label met A)

Diagnostic 5 (Kaggle `yasunorim/pitcher-last-outing-confirmation-diagnostic-5`, GHA 38845620755 success, notebook
`e82f5e7` md5 0cd200db; RPi5 `v8/diag5_out/log.txt`): 2024-2026 deciding contrast 313 vs 89, stratified AUC 0.5923 /
0.5924, floor P p 0.0250 / 0.0240, rotation p 0.001 (both pass); E +0.0908; 2021-2023 counts only (clean long breaks
182, of which 2021 110; positives 525 before any filter, N1 433); power of the conjunction at those sizes: +E 0.93,
+E/2 0.28, unplanted bootstrap 0.82, null 0.000. No code fault. By v8.1 item 5 the design is not reopened.

Planted defects on md5 0cd200db (`conf_chain8.log`): code test exit 0, 40 of 41 at the intended gate; k5 stopped at the
new clean-contrast draw gate one step earlier; k5b (that gate off) stopped at the intended gate.

The opus audit of the FREEZE draft (no code fault; text fixes taken into the FREEZE, each checked in the code and logs)
asked for two label-free additions to the notebook, made without reading any 2021-2023 A: a printed count of 1 and 0
rows of the clean contrast by calendar fifth (v8.1 item 2 said they are printed; nothing printed them), and two gates on
`clean_test` in the fake-label code test (the per-stratum AUC equality and the NaN path, which had been tested on the
old pool test only). `patch_audit4.py`; new md5 b6a7dbc8; planted defects rerun in full (`conf_chain9.log`) with three
added: k5b, k9c (clean_test NaN fails open), k6c (clean_test unstratified).
