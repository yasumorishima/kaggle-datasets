# Pre-registration: do pitchers' pitch mix or secondary-pitch quality drift before the injured list? (MLB and Triple-A, 2024-2026)

Status: FROZEN when merged to main (notebook `pitcher-mix-il/pitcher_mix_il.py`, md5 4bd6bc6f828612ccc192d954e2bef65e); the merge commit is recorded in the README line of this study.

## Why this study

Two frozen pitcher studies found no support for any of their primaries:

- `FREEZE_pitch.md` (commit `2c6b01d`): window **levels** of 14 parameters (fastball velocity, top velocity, spin and
  movement, extension, release point, arm angle, pooled off-speed velocity, zone, whiff, K - BB, hard-hit) against each
  pitcher's own normal (MLB AUC 0.518, arm placements 0.535, Triple-A 0.543, each inside its rotation floor; model over
  the rival -0.022).
- `FREEZE_pitch_inouting.md` (commit `76537f0`): fastball fade and scatter **inside an outing** (0.513, 0.520, 0.462,
  each inside its floor; -0.002).

Both measured the fastball. The other pitches entered only as one velocity pooled over every non-fastball. Not measured:
**which pitches a pitcher throws** (a pitcher who protects an arm may throw fewer breaking balls, or lean on them when
the fastball is down) and **how his breaking ball and changeup move and spin** (each type against its own normal). This
study measures them with the same rows, labels, rival, floors and tests.

## Disclosures (before any new parameter met a label)

- **Read before this plan:** the complete printed output of both frozen pitcher runs (tests with floors, subsets,
  reported items, model tables, permutation importance by group and its top 10, the event level and the alarm rule).
  In the in-outing run the top delivery-type item of the permutation importance was the short arm-angle scatter z; in the
  level run, the short fastball velocity z. They motivated looking beyond the fastball.
- **Families on these placements:** this is the third pre-registered family on the same pitcher placements (MLB and
  Triple-A clubs, 2024-2026), after `FREEZE_pitch.md` and `FREEZE_pitch_inouting.md`, and the parameters were chosen
  after reading permutation importances computed on the MLB 2026 test labels that R4 reuses. The Bonferroni correction
  below covers this family only (four-way over three families is about 14% familywise). Each verdict is therefore also
  printed at a **12-way reading across the three families** (0.208% and 99.79%), and a 4-way SUPPORTED is written as
  "supported within this family, not across the three pitcher families" unless the 12-way reading also holds.
- **What the level run printed for the overlapping parameter:** the frozen "Off-speed velocity" was in the level run's
  delivery group, whose MLB permutation importance was -0.0192 (Triple-A -0.0028); it was not in the MLB top 10.
- **Overlap:** the frozen level study's "Off-speed velocity" pooled all non-fastballs. Here velocity is per type of the
  primary breaking ball and of the primary changeup or splitter, so a mix shift does not move it.
- **Label-free probe** on the code-test sample (100 pitchers per level, 2024-2026, pitch columns only; MLB / Triple-A):
  breaking balls (SL, ST, SV, CU, KC, CS) are 29.8% / 30.6% of pitches and changeups or splitters (CH, FS, FO) 15.8% /
  15.0%; speed, spin and movement of both families are 98.8-100% filled; 92.5% / 93.8% of pitcher-seasons have at least
  30 breaking balls and 73.2% / 82.5% at least 30 changeups or splitters; the primary breaking type (most pitches before
  the day) changes during the season in 22.4% / 23.1% of pitcher-seasons, the primary changeup type in 1.4% / 2.3%; the
  median sd of 30-pitch block means within a pitcher-season-type is 0.63 mph (velocity), 40 / 48 rpm (breaking spin),
  0.08 ft (movement), larger than independent pitches would give (part of it is drift across the season), so block
  means do move.
- **Considered and dropped before freezing:** the fastball share (one minus the two shares and the rare other pitches),
  per-type usage shares (a type switch moves them mechanically), spin axis and release point of secondary pitches
  (delivery dimensions already measured on the fastball), and any directional item (no direction is fixed by a prior
  result).

## Design: the frozen code with only the parameters replaced

The notebook downloads the frozen pitcher notebook from commit `2c6b01d`
(`pitcher-anomaly-il/pitcher_anomaly_il.py`, md5 `100c853ccb8e42f386f547ead97f0b95`), checks the md5, applies a fixed
list of exact text replacements (each must match exactly once, else the run stops) and runs it. Everything not replaced
is the frozen study unchanged: the inputs and their md5 gates, the 90% fill rule, players and game logs, labels
(placements by MLB or Triple-A clubs within 14 days, rows with an appearance at the other level in the window dropped,
rehab windows), rows (one per pitcher and appearance day, 2024-2026; Ohtani 2026 left out), the primary fastball choice
(still used by the rival's previous-season fastball velocity and the "no primary fastball" count), own-normal centre and
scale (role-matched own previous-season MAD, else the league-typical scale by role), daily ranks within role, the rival
B0, the models, the bootstrap over pitchers, the rotation floors and their gates, the planted checks under fake labels
(including the look-ahead check, unchanged), subsets, reported items and figures.

The replacements, all in the notebook:
1. The parameter lists replaced by the ten parameters below (no fastball parameters).
2. The fill gate maps each parameter to its source column (the shares: pitch_type).
3. The per-type fastball window loop and the other-parameter loop in `params_for` replaced by the computation below.
4. The per-type z selection (frozen: per fastball type, chosen by the primary fastball) generalised to the two families,
   each chosen by its own primary type; the league-typical scale uses each family's full-season most-thrown type, as
   the frozen code used the full-season most-thrown fastball type (label-free).
5. U1 / U2 need 5 of the 10 short z (frozen: 7 of 14; short z only, unchanged); the quality U2 needs 4 of its 8 short z
   (frozen delivery U2: 5 of 10). Smoke value of MIN_Z: 3.
6. The centre kind recorded per row (`base_kind`) is taken from "Breaking share (short)" (an all-pitch window, as the
   frozen "Zone rate (short)").
7. The planted P2 checks use "z_Breaking share (short)" instead of "z_FB velocity (short)".
8. Group labels: "delivery" becomes "pitch quality", "results" becomes "mix"; the interpretation note "results, not
   delivery" becomes "mix, not pitch quality", its Triple-A qualifier "(a roster move by mix)".
9. Title, text, test labels R1-R4 and output file names (`mix_*`); library versions printed first.
10. Added: the equality gates (after the labels and the frozen "effective sample" line, before any test).
11. Added inside the frozen look-ahead check (label-free, real and code-test runs): each family's primary-type choice is
    recomputed independently for every pitcher-season and day (per-type count of pitches dated before the day, first
    maximum, -1 if none) and must equal the notebook's; and a second planted change on the 20 pitcher-seasons with both
    families and the most appearance days (from the middle day on, pitches alternate between each family's least-used
    type before that day, with extreme speed, spin and movement) must leave every series up to that day unchanged and
    change at least one per-type series of each family after it. The frozen cutter plant removes family pitches and adds
    none, so on its own it catches few leaks in the family choice (label-free probe on the code-test sample: a same-day
    leak 3 of 215 pitcher-seasons, a full-season leak 13 of 215; the second plant on all 215, without requiring both
    families, 52 and 211 of 215). As shipped (the 20 pitcher-seasons of the MLB code-test sample), the second plant
    stops a same-day choice leak in 2 of 20, a full-season choice leak in 19 of 20 and a same-day value leak in 20 of
    20, and the correct code in 0 of 194 eligible pitcher-seasons; the same-day choice leak is stopped by the
    independent recomputation.
12. Added: the subset "same role as the previous appearance" (the previous appearance's start flag equals the day's
    role) in the subset tables of R1-R4, the interpretation note "role change" (below), and label-free prints of each
    family's primary-type changes and rows without a short family velocity z.
13. Text and printed labels: the section text and the `params_for` docstring describe the family series; the frozen
    "delivery only" reports print as "pitch quality only".

**Equality gates (real run only):** with B0 independent of the parameters and fixed seeds, the run stops unless these
equal the frozen run's printed values (a changed Kaggle image, library or StatsAPI correction stops the run here): rows
61,966 (MLB, 1,180 pitchers) and 55,686 (Triple-A, 1,853 pitchers); usable rows 53,384 with 3,553 positive (MLB) and
45,453 with 1,884 positive (Triple-A); MLB 2026 AUC(B0) 0.5644912856858865 (tolerance 1e-6). The in-outing run passed
the same gates on Kaggle.

## The ten parameters (long and short windows, only pitches strictly before the day, same season and level)

Families: **breaking** = SL, ST, SV, CU, KC, CS; **changeup** = CH, FS, FO.

Mix (all-pitch windows of 300 / 100 pitches, the frozen zone-rate windows; a pitch with no type or of type FA, PO, IN
or EP is missing, the frozen exclusions from off-speed; every other typed pitch, including fastballs and the rare KN,
SC and UN, counts in the denominator of both shares):
1. **Breaking share**: share of breaking balls.
2. **Changeup share**: share of changeups and splitters.

Quality (per type, windows of the last 100 / 30 pitches of that type, the frozen off-speed windows; mean of the column):
3-6. **Breaking velocity, spin, vertical movement, horizontal movement** (release_speed, release_spin_rate, pfx_z,
     pfx_x) of each breaking type.
7-10. **Changeup velocity, spin, vertical movement, horizontal movement** of each changeup type.

Each type has its own series, own normal and z; the z used on a day is the one of that family's **primary type** (the
type of the family with the most pitches before the day in the season, as the frozen fastball choice; no fallback to the
previous season, so a family with no pitch yet that season has no z). A window value needs its window full and at least
half of it defined (the frozen rule). Twenty z per row. Groups: mix (1-2), pitch quality (3-10). A pitcher who throws
only one of the two families has 6 short z, enough for U1 / U2; the share of the family he does not throw is 0 in every
window, so its z is exactly 0 and pulls his U2 toward the middle.

## Primaries (Bonferroni over four: 0.625% and 99.375%)

The frozen tests, with the new parameters:

- **R1** MLB 2024-2026, all placements: AUC of the daily within-role rank of U2. SUPPORTED if the lower bound > 0.5 and
  the AUC is above all 200 rotations (k uniform in [m, n - m], m = max(2, ceil(n / 6)), n >= 7).
- **R2** MLB 2024-2026, arm placements only (negatives as R1): as R1.
- **R3** Triple-A 2024-2026, all placements: as R1.
- **R4** MLB 2026, trained on 2024-2025: AUC(M2) - AUC(B0), M2 = B0 + the 20 z + U1 + U2. SUPPORTED if the lower bound
  > 0 and the gap is above all 20 gaps-fixed rotation floors (each z column rolled over its defined values by
  round(k * n_c / n), U1 / U2 recomputed).

A supported R4 means the mix and secondary-pitch z add to the rival; M2 does not contain the earlier studies' z, so it
does not mean "beyond the fastball levels".

## Reported, not tested

All items the frozen notebook reports, with the new parameters: subsets, quality-only U2 (the frozen "delivery only"),
pooled-role rank, IL30, U1, Triple-A 2026 M2 - B0, M3 (with season-to-date means of the new z), arm placements in 2026,
permutation importance by group (mix / pitch quality / rival / summary scores) and its top 10, the event level and the
alarm rule.

## Interpretation rules (fixed now)

As the frozen study, with "results, not delivery" replaced by **"mix, not pitch quality"** (written when the
quality-only AUC is not above its floor mean, or, for R4, when the quality group's permutation importance is not
positive). A supported R3 is "before a Triple-A placement, not necessarily an injury" (frozen wording).

Added: **"role change"**, written for R1-R3 when the AUC on the subset "same role as the previous appearance" is not
above that subset's floor mean, and for R4 when M2 - B0 there is not above its floor mean. A pitcher moving between
starting and relieving changes his mix mechanically, and the daily role (more than half of the season's earlier
appearances were starts) lags such a move.

Wording across families: the notebook prints the 12-way reading as its own column of the summary; in every write-up
of the result, a 4-way SUPPORTED whose 12-way column is NOT SUPPORTED is written "supported within this family, not
across the three pitcher families".

## Code test

Before the real run, the notebook runs on the code-test sample with random placements (`SMOKE_FAKE_LABELS=1`) and must
finish with every gate and planted check passing (planted timing beats the floors; the floors keep a planted trait; a
role artefact does not pass R1's interval; the look-ahead check, which sets every pitch from a day on to a cutter with
extreme values and requires every series up to that day, including each family's primary-type choice, to be unchanged;
the independent recomputation of the family choice; and the second planted change, item 11). Before freezing, three
planted leaks in the family code (the family choice counting the day's own pitches, the family choice taken from the
full season, the per-type windows counting the day's own pitches) were each run through the code test and had to stop
it: the first two stopped at the independent recomputation, the third at the frozen cutter plant (the frozen check
runs first, so the second plant did not get to stop any of them in the code test; its strength is the probe above).

## Known limits

Pitch types are Statcast's classification, which can relabel a pitch (a slider becoming a sweeper) without a change in
the pitch; a relabel within a family moves no family share but starts a new per-type series, and a relabel across
families (a slider becoming a cutter, which is a fastball here) moves the breaking share. When a pitcher stops throwing
his primary type, its window keeps its last value until another type of the family has more pitches (on the MLB
code-test sample, 2.3% of quality rows use a primary type last thrown more than 14 days before, 1.0% more than 30). Spin and
movement come from park-specific tracking; Triple-A calibration differs more by park. Movement is pfx (relative to a
spinless pitch), not the arm-side-corrected break. Pitchers with one family carry 6 short z, those with both 10, so U2
averages over different sets.

## Not done

No comparison of a new parameter with any placement before this file is on main. No change after the run except a dated
amendment that says whether it came before or after any result was printed. The earlier pitcher studies' verdicts are
not revisited.
