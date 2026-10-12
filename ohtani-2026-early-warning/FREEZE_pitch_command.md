# Pre-registration: does a pitcher's command drift before the injured list? (MLB and Triple-A, 2024-2026)

Status: FROZEN when merged to main (notebook `pitcher-command-il/pitcher_command_il.py`, md5
`af13e96ac1d3a6c8e4f7dc83bcd72ad8`); the merge
commit is recorded in the README line of this study.

## Why this study

Three frozen pitcher studies found no support for any of their primaries:

- `FREEZE_pitch.md` (commit `2c6b01d`): window **levels** of 14 parameters (fastball velocity, top velocity, spin and
  movement, extension, release point, arm angle, pooled off-speed velocity, zone, whiff, K - BB, hard-hit) against each
  pitcher's own normal (MLB AUC 0.518, arm placements 0.535, Triple-A 0.543, each inside its rotation floor; model over
  the rival -0.022).
- `FREEZE_pitch_inouting.md` (commit `76537f0`): fastball fade and scatter of velocity, release point, arm angle and spin
  axis **inside an outing** (0.513, 0.520, 0.462, each inside its floor; -0.002).
- `FREEZE_pitch_mix.md` (commit `f7e2a53`): **pitch mix** and **secondary-pitch quality** (0.501, 0.501, 0.496, each
  inside its floor; -0.023).

All three measured how a pitch leaves the hand or which pitch it is. **Where the pitch goes** entered only as the zone
rate (share of pitches in zones 1-9). Not measured: **command**, how far pitches miss around where that pitcher's pitches
of the type went in the outing, how often they land over the heart of the plate or far outside the zone, and how often
plate appearances reach three balls. A pitcher protecting a sore arm, or one whose mechanics are off, may keep his
velocity and spin and still lose the location. This study measures command with the same rows, labels, rival, floors
and tests.

## Disclosures (before any new parameter met a label)

- **Read before this plan:** the complete printed output of the three frozen pitcher runs (tests with floors, subsets,
  reported items, model tables, permutation importance by group and its top 10, the event level and the alarm rule).
  In all three runs only the rival B0 (n_prev_il and other workload and history items) put AUC above 0.5. In the level
  run the zone rate was one of the results items, whose group importance on the MLB 2026 test labels was +0.0015
  (Triple-A +0.0011); the short zone-rate z was not in the MLB top 10, the short K - BB z was (fifth, +0.0042).
- **Families on these placements:** this is the fourth pre-registered family on the same pitcher placements (MLB and
  Triple-A clubs, 2024-2026), after `FREEZE_pitch.md`, `FREEZE_pitch_inouting.md` and `FREEZE_pitch_mix.md`, and the
  parameters were chosen after reading permutation importances computed on the MLB 2026 test labels that C4 reuses. The
  Bonferroni correction below covers this family only. Each verdict is therefore also printed at a **16-way reading
  across the four families** (0.156% and 99.84%), and a 4-way SUPPORTED is written as "supported within this family, not
  across the four pitcher families" unless the 16-way reading also holds.
- **Overlap with frozen parameters:** the frozen zone rate (zones 1-9) is not a parameter here. The heart share (zone 5)
  and the waste share are subsets of in-zone and out-of-zone pitches, so they move with the zone rate in part; the frozen
  K - BB counts walks, and the three-ball share counts plate appearances that reach three balls whatever their end.
- **New input:** the frozen Triple-A pitching files (md5 in `FREEZE_pitch.md`, unchanged and still checked) do not hold
  the pitch location or the count. A separate label-free fetch, `aaa-statcast-location-fetch/` (Kaggle
  `yasunorim/triple-a-statcast-pitch-location-2023-2026-fetch`, the frozen pitching fetch with only the kept columns and
  file names changed; kaggle-datasets PR #45, GitHub Actions run 38203571126), keeps the pitch keys, pitcher, game
  date, batter side, plate_x, plate_z, sz_top, sz_bot, balls and strikes. md5 of its files (printed by the fetch and
  checked by the notebook): 2023 `ddc88cbdb25d87cd356890b5934a1f69`, 2024 `0eb84f0af52298597404accf72a318a5`, 2025
  `9652d14a943e76d3d772437554110dc4`, 2026 `921600911144b7e89c2fe58d8f87aef3`. The notebook left-joins them to the frozen files by (game_pk, at_bat_number,
  pitch_number), stops unless at least 99% of Triple-A pitches of every season find a location row and at least 99.9% of
  those agree with the frozen file on pitcher, batter side and date, and blanks the location of the rest. Those gates
  cannot see a location moved to another pitch of the same plate appearance, so a third gate checks the location against
  the frozen `zone`: in every season at least 93% of the joined Triple-A pitches with a zone and all four location
  columns must agree on in-zone (zone 1-9) versus inside the box |plate_x| <= 0.83 ft and sz_bot <= plate_z <= sz_top.
  Measured label-free on the code-test sample before freezing: Triple-A 0.952 / 0.956 / 0.955 / 0.954 (2023-2026), MLB
  0.960 / 0.959 / 0.953 (2024-2026); with each location keyed to the next pitch number (the first pitch of each plate appearance left
  unjoined) 0.490-0.500; with locations rotated by one pitch inside the plate appearance (mutation c5) 0.511 (2023). On
  the full Triple-A files (all pitches, before the notebook's row filters): 0.9515 / 0.9551 / 0.9553 / 0.9543. A code test
  of the fetch on two days each of 2023 and 2026 matched every key, pitcher and batter side of the frozen files. MLB
  location columns come from the same md5-checked MLB files as before (columns not read by the frozen notebook).
- **Label-free probe** on the code-test sample (100 pitchers per level, pitch columns only, smoke windows):
  - lowest season fill of the new columns: MLB 0.9971 (plate_x, plate_z, sz_top, sz_bot; stand and balls 1.0),
    Triple-A 0.9935 after the join (stand and balls 1.0); Triple-A location joined to 100% of the sample's pitches, with
    the same pitcher, batter side and date on 100% of them; no parameter dropped by the fill rule at either level.
  - fetch coverage: every finished regular-season Triple-A game of 2023-2026 present (2,224 / 2,232 / 2,232 / 2,223).
  - share of rows with each short z defined (reliever / starter): MLB FB horizontal and vertical 0.646 / 0.667,
    secondary 0.690 / 0.703, heart and waste 0.698 / 0.681, three-ball 0.678 / 0.681; Triple-A 0.692 / 0.596,
    0.732 / 0.662, 0.735 / 0.633, 0.711 / 0.622.
  - three-ball logic (MLB sample, 96,475 plate appearances): every walk reaches three balls (7,548 of 7,548); the 195
    intent_walk rows all carry balls = 3 and are excluded; 4 plate appearances have more than one pitcher and 1 starts
    at three balls.
- **Other couplings:** more three-ball plate appearances mean more pitches, so the three-ball share is mildly coupled
  to the pitch counts in B0. The 16-way bounds are the 0.156th and 99.84th percentiles of 2,000 bootstrap draws, about
  the fourth value from each end (linear interpolation at index 3.12 of 0-1,999), so they are noisier than the 4-way
  bounds.
- **Considered and dropped before freezing:** mean location (a target, chosen by pitch plan and scouting, not command);
  edge or shadow shares (depend on a zone model and on intent); first-pitch strike share (a subset of the zone rate and
  count); called-strike share (umpire and, from 2026, ABS challenges in MLB); scatter normalised by the batter's zone
  height (sz_top and sz_bot are estimated per batter; the raw height in feet is used); a mirrored horizontal location in
  one group over both batter sides (pitchers target different sides by batter side, so the groups are split by side
  instead); any directional item (no direction is fixed by a prior result).

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
1. The parameter lists replaced by the six parameters below.
2. Reading: the MLB files are read with the frozen columns plus batter side, plate_x, plate_z, sz_top, sz_bot and balls;
   the Triple-A files with the frozen columns plus batter side (already in them), then the location join above (with its
   md5 check and gates).
3. The fill gate maps each parameter to its source columns (a parameter is dropped at a level if any of them is under
   90% filled in any season); the fill table prints the new columns.
4. The per-type fastball window loop and the other-parameter loop in `params_for` replaced by the computation below.
5. U1 / U2 need 3 of the 6 short z (frozen: 7 of 14); the scatter U2 needs 2 of its 3 short z (frozen delivery U2: 5 of
   10). Smoke value of MIN_Z: 2. In-outing groups need at least 3 pitches (`IN_MIN = 3`). The two fastball scatter z
   come from the same pitches and, because plate_x and plate_z are missing together in these files (0 pitches with
   one missing in the four Triple-A location files and the MLB sample; the code does not require it), are defined
   together (code test: 0.646 and 0.646), so in practice the scatter
   U2 needs the fastball pair, and U1 / U2 need the fastball pair and one more z.
6. The centre kind recorded per row (`base_kind`) is taken from "Heart share (short)" (an all-pitch window, as the frozen
   "Zone rate (short)").
7. The planted C4 checks use "z_FB horizontal location scatter (short)" instead of "z_FB velocity (short)".
8. Group labels: "delivery" becomes "scatter", "results" becomes "zone outcomes"; the interpretation note "results, not
   delivery" becomes "zone outcomes, not scatter", its Triple-A qualifier "(a roster move by zone outcomes)".
9. Title, text, test labels C1-C4 and output file names (`command_*`); library versions printed first.
10. Added: the equality gates (after the labels and the frozen "effective sample" line, before any test).
11. The frozen look-ahead plant (every pitch from the middle day on becomes a cutter with extreme values) also sets
    plate_x, plate_z, sz_top and sz_bot to 999 and balls to 3; the frozen check that nothing before the day changes then
    covers the new series. The added requirement that a fastball series changes after the day only confirms the plant
    reached the new code (it holds whenever the late pitches become cutters, so it is not a test of the scatter code).
    sz_top / sz_bot are planted because the audit found that a waste share reading the next pitch's zone edges passed
    the plant without them (it stops it now: mutation c6 below). A leak this plant cannot catch: a balls maximum taken
    over the whole season (3 is already the real maximum).
12. Added: the subset "same role as the previous appearance" (the previous appearance's start flag equals the day's
    role, as in `FREEZE_pitch_mix.md`) in the subset tables of C1-C4, the interpretation note
    "role change" (below), and a label-free print of the share of rows with each short z defined, by role.
13. The 16-way bounds and summary column.

**Equality gates (real run only):** with B0 independent of the parameters and fixed seeds, the run stops unless these
equal the frozen run's printed values: rows 61,966 (MLB, 1,180 pitchers) and 55,686 (Triple-A, 1,853 pitchers); usable
rows 53,384 with 3,553 positive (MLB) and 45,453 with 1,884 positive (Triple-A); MLB 2026 AUC(B0) 0.5644912856858865
(tolerance 1e-6). The in-outing and mix runs passed the same gates on Kaggle.

## The six parameters (long and short windows, only pitches strictly before the day, same season and level)

A **group** is the pitches of one appearance (game_pk and date) thrown to one batter side; for secondary pitches also of
one pitch type. A pitch's squared miss is its squared distance from its group's mean location (plate_x and/or plate_z,
in feet) times k / (k - 1), k the group's pitches with the coordinates and batter side known; groups with fewer than 3
such pitches give no value. A scatter is the square root of the window mean of squared misses.

Scatter (the frozen fastball and off-speed windows):
1. **FB horizontal location scatter**: plate_x, per fastball type (FF, SI, FC), windows of the last 150 / 50 pitches of
   that type; the z used on a day is that of the day's primary fastball (frozen choice).
2. **FB vertical location scatter**: plate_z, as 1.
3. **Secondary location scatter**: plate_x and plate_z together, over all secondary pitches (a typed pitch that is not a
   fastball and not FA, PO, IN or EP, the frozen off-speed set), groups split by pitch type, windows of 100 / 30 such
   pitches.

Zone outcomes:
4. **Heart share**: share of pitches in zone 5 (the middle ninth of the zone), windows of 300 / 100 pitches (the frozen
   zone-rate windows); a pitch with no zone is missing.
5. **Waste share**: share of pitches more than 1 ft outside the zone (|plate_x| > 1.83 ft, or plate_z more than 1 ft
   above sz_top or below sz_bot), windows of 300 / 100 pitches; a pitch with any of the four missing is missing.
6. **Three-ball share**: share of the pitcher's plate appearances (the frozen K - BB set, intentional walks removed) in
   which a pitch was thrown with three balls, windows of 100 / 30 plate appearances (the frozen K - BB windows).

A window value needs its window full and at least half of it defined (the frozen rule). Twelve z per row. Groups:
scatter (1-3), zone outcomes (4-6).

## Primaries (Bonferroni over four: 0.625% and 99.375%)

The frozen tests, with the new parameters:

- **C1** MLB 2024-2026, all placements: AUC of the daily within-role rank of U2. SUPPORTED if the lower bound > 0.5 and
  the AUC is above all 200 rotations (k uniform in [m, n - m], m = max(2, ceil(n / 6)), n >= 7).
- **C2** MLB 2024-2026, arm placements only (negatives as C1): as C1.
- **C3** Triple-A 2024-2026, all placements: as C1.
- **C4** MLB 2026, trained on 2024-2025: AUC(M2) - AUC(B0), M2 = B0 + the 12 z + U1 + U2. SUPPORTED if the lower bound
  > 0 and the gap is above all 20 gaps-fixed rotation floors (each z column rolled over its defined values by
  round(k * n_c / n), U1 / U2 recomputed).

A supported C4 means the command z add to the rival; M2 does not contain the earlier studies' z.

## Reported, not tested

All items the frozen notebook reports, with the new parameters: subsets, scatter-only U2 (the frozen "delivery only"),
pooled-role rank, IL30, U1, Triple-A 2026 M2 - B0, M3 (with season-to-date means of the new z), arm placements in 2026,
permutation importance by group (scatter / zone outcomes / rival / summary scores) and its top 10, the event level and
the alarm rule.

## Interpretation rules (fixed now)

As the frozen study, with "results, not delivery" replaced by **"zone outcomes, not scatter"** (written when the
scatter-only AUC is not above its floor mean, or, for C4, when the scatter group's permutation importance is not
positive). A supported C3 is "before a Triple-A placement, not necessarily an injury" (frozen wording).

Added: **"role change"**, written for C1-C3 when the AUC on the subset "same role as the previous appearance" is not
above that subset's floor mean, and for C4 when M2 - B0 there is not above its floor mean. A pitcher moving between
starting and relieving changes his command series mechanically (a reliever's outings are short, so his groups are
smaller, and he faces batters in different counts and orders), while the daily role (more than
half of the season's earlier appearances were starts) and the role-matched scale lag such a move.

Wording across families: the notebook prints the 16-way reading as its own column of the summary; in every write-up of
the result, a 4-way SUPPORTED whose 16-way column is NOT SUPPORTED is written "supported within this family, not across
the four pitcher families".

## Code test

Before the real run, the notebook runs on the code-test sample with random placements (`SMOKE_FAKE_LABELS=1`) and must
finish with every gate and planted check passing (planted timing beats the floors; the floors keep a planted trait; a
role artefact does not pass C1's interval; the look-ahead check with item 11; the location join gates). Before freezing,
planted leaks in the new code were each run through the code test and had to stop it at the intended gate (checked
from each log's assertion, not only its exit code); all six did, each built from the final notebook (md5 above) on
the code-test sample with the real Triple-A locations:

| leak | stopped by |
|---|---|
| c1 fastball horizontal scatter window includes the day's own pitches | look-ahead check (`('SI', 'FB horizontal location scatter')`) |
| c2 fastball miss measured from the season mean to that batter side | look-ahead check (`('FF', 'FB horizontal location scatter (short)')`) |
| c3 three-ball window includes the day's own plate appearances | look-ahead check (`Three-ball share`) |
| c4 location file keyed to the next plate appearance | join gate (location missing for more than 1%) |
| c5 locations rotated by one pitch inside each plate appearance (keys, pitcher, side, date still match) | zone-agreement gate (Triple-A 2023: 51.1%) |
| c6 waste share reads the next pitch's sz_top / sz_bot | look-ahead check (`Waste share`) |

The code test of the final notebook (location md5s filled in) finished with every gate passing (`exit 0`); Triple-A
zone agreement 95.21% / 95.59% / 95.53% / 95.44% (2023-2026).

## Known limits

Locations are Statcast's tracked plate crossing; Triple-A tracking differs more by park. A group mean is the pitcher's
location for that pitch type to that batter side in that outing, not his intended target: a pitcher who mixes targets
(up and down, in and out) within an outing has a large scatter without missing, and a scatter change can be a plan
change. Relievers' groups are small, so their scatter is noisier and more often undefined (see the probe). zone 5 and
sz_top / sz_bot are Statcast's zone model. The three-ball share depends on the count when a pitcher enters mid
plate appearance.

## Not done

No comparison of a new parameter with any placement before this file is on main. No change after the run except a dated
amendment that says whether it came before or after any result was printed. The earlier pitcher studies' verdicts are
not revisited.
