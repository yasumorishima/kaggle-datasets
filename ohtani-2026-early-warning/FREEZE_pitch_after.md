# Pre-registration: does the last outing before an arm placement look slow? (pitchers, MLB 2024-2026)

Status: FROZEN when merged to main (notebook `pitcher-after-il/pitcher_after_il.py`, md5 `a96dcbef03d67e4cfc9404279ac2e566`; inputs: the frozen
MLB 2024-2026 files of `yasunorim/mlb-statcast-bat-tracking-2024-2025` and the frozen Triple-A 2023-2026 files, md5
as in FREEZE_pitch.md); the merge commit is recorded in the README line of this study.

## Why this study

Seven pre-registered pitcher studies were NOT SUPPORTED: six families of "deviation from the pitcher's own normal"
(levels, in-outing fade, pitch mix, command, reach-back, spin-axis gap) and the confirmation of the sixth on MLB
2021-2023. Every one of them built a pitcher-day's features from pitches thrown **strictly before** that day
(`wmean`/`wpct` in the frozen notebook use `searchsorted(..., side="left")`). So the row of a pitcher's last appearance
before a placement never contained that appearance. Label-free, from the StatsAPI transactions and appearance dates
(2021-2026, counts only): arm placements with an MLB appearance in the 14 days before follow the last appearance
within 0 days 0.4%, 1 day 73.0%, 3 days 90.3%, 7 days 97.9% (n = 1,139); 47.8% of them end the pitcher's season.

This study scores **the outing that just ended**. Framing: a post-outing, pre-announcement public signal, not a claim
that an injury is seen before it happens.

## Rows, labels, rival

The notebook runs the frozen pitcher notebook (commit `2c6b01d`, md5 `100c853ccb8e42f386f547ead97f0b95`) up to its
tests and takes its test definitions (B0's features `CTX`, the frozen gradient boosting, `auc`, `boot`, `daily_rank`),
so rows, labels and the rival B0 are the frozen ones. Equality gates (real run only): rows and labels (61966, 1180,
55686, 1853, 53384, 3553, 45453, 1884) and MLB 2026 AUC(B0) 0.5644912856858865 (to 1e-6).

**Target (new).** Positive = the last appearance row before an arm placement (frozen arm rule) whose effective date is
1 to 14 days after the row's day; the other rows of that pitcher before the same placement are left out; a placement
whose last row is on its effective day (`days_to_il` 0) or is unusable (frozen `use14`) contributes no positive.
Negatives = usable rows with no placement within 14 days. Rows before non-arm placements are left out (frozen arm
rule). Gates: every positive is 1 to 14 days before its placement; no row of the pitcher lies between a positive and
its placement (a consistency check on the same rows).

**Game-log rule.** A positive with a StatsAPI game-log appearance at the same level after its day and on or before its
placement's effective date is left out (its last outing has no Statcast row, e.g. a pitcher hurt before his first
pitch): the count is printed at both levels, and MLB asserts that at most 10% of positives are left out this way
(Triple-A only prints). Label-free, on the full files: of the rows' pitchers' game-log appearance days in the row
seasons, 6 of 61,986 (0.01%) at MLB and 19 of 57,368 (0.03%) at Triple-A have no Statcast pitch of his that day (all of them log 0 pitches). The pre-check's
random placements were all effective the day after their row (k = 1), so it could not show injury-tied cases; it
found none to leave out (0 of 326 MLB and 0 of 521 Triple-A positives); a game-log day planted the day after one
positive left exactly that positive out (325 kept).

## The outing's score

The outing data are read again from the same md5-checked files with an allowlist of columns (the frozen
future-column assert applies to it): `pitcher, game_date, game_pk, at_bat_number, pitch_number, inning,
inning_topbot, pitch_type, release_speed, game_type`, and for MLB `bat_score, post_bat_score`.

For row day d, T = the frozen primary-fastball choice for d (most pitches of FF/SI/FC this season before d, else the
previous season's most thrown; gate: agrees with the frozen `no_primary` flag on 100% of MLB rows):
- x_d = mean release_speed of the type-T pitches on d (at least 5 with a value);
- normal_d = mean of the last 60 type-T release_speed values strictly before d, same season and level (frozen window
  rule: window full, half filled);
- dev_d = x_d - normal_d; undefined on doubleheader days (two games on d; gate: A undefined there);
- scale_d = max(own, 0.5 x lscale): own = 1.4826 x MAD of the pitcher-season's earlier dev values (at least 5,
  strictly before d); lscale (of the row's role) = median over the league season's pitcher-seasons of that season role
  (more than half starts over the whole season; MLB 2025, Triple-A 2023, as the frozen league scales) with at least 20
  dev values of their own MAD; no own -> lscale. The league constants use a whole season (2025 also for 2024 rows), as
  the frozen league scales do; they hold no pitcher's own later outings beyond his one share of the median;
- **A_d = -(dev_d / scale_d)** clipped to [-10, 10] (higher = slower than his own recent normal); the score is the
  frozen daily within-role rank of A.

Gates on the score (both levels unless stated):
- look-ahead: in the largest pitcher-season that has a day t0 followed by an appearance the next calendar day, pitches
  after t0 set to 999 and to type FC change nothing up to t0 (choice, x, normal); pitches on t0 set to 999 change x at
  t0 but neither the normal at t0 nor the own scale at t0 (t0 is chosen where adding the day's dev to the earlier
  ones would move their MAD, and the scale with the day's dev is compared with the scale with the day's dev missing);
  the usual pitch count of a day is the same with the day's count set to 9999 and set to missing, on a day where a
  count of 9999 added to his earlier same-flag counts would move their median;
- x equals the mean of the chosen type's pitches of the day recomputed from the full pitch table (count >= 5);
- the normal equals a direct loop (last 60 of the type strictly before the day, half filled) on 200 random rows with
  a defined normal;
- MLB: the chosen type's 150-pitch mean equals the frozen row column `FB velocity` on every row (the choice is the
  frozen one); the previous outing's pitch count and start flag equal the frozen rows' `prev_app_pitches` and
  `prev_start` on every pair of consecutive single-game rows. Triple-A prints these (99.76% / 99.22% / 100% / 100% in
  the pre-check): its outing table keeps the rehab pitches the frozen code removes from Triple-A (reported only).

## What anyone watching saw (the rival's additions)

From all pitches of the game, in game order (`game_pk, at_bat_number, pitch_number`), per pitcher and game: start
(first pitch in inning 1, frozen definition), pitch count, batters faced, pitches in his last inning, runs during his
outing (sum of post_bat_score - bat_score over his pitches; MLB; runs scoring between pitches without a pitch
record are missed), **mid-inning exit** (the next pitch of the game is
in the same inning and half and is thrown by another pitcher; no next pitch = not an exit), **left mid-at-bat** (the
next pitch has the same at-bat number and another pitcher), **shortfall** (usual minus today's count; usual = median of
his earlier single-game outings this season with the same start flag, at least 3, else the league season's median for
the flag). **Early hook** = left mid-at-bat, or a mid-inning exit with a shortfall of 15 or more (undefined when a
mid-inning exit has no shortfall). Records are per pitcher, game and date (a suspended game resumed later gives two).
All undefined on doubleheader days (gate). A synthetic-game self-test checks the exit flags (between innings, mid-inning after a completed
at-bat, mid-at-bat, the game's last pitch). Real run gates (MLB only; Triple-A rates are printed): the mid-inning
exit rate is between 0.05 and 0.95 for each role, and left mid-at-bat between 0 and 0.05.

Label-free on the full MLB 2024-2026 files (2026-10-14, with this code): starts 14,673 outings, mid-inning exit 0.419,
left mid-at-bat 0.0049, early hook 0.077, dev defined 0.859; relief 47,650 outings, 0.178, 0.0021, 0.005, 0.536;
league scale of dev 0.678 (relievers) / 0.648 (starters) mph; 19 doubleheader days.

## Primaries (two; each at 0.0125)

- **P1** MLB 2024-2026 pooled, the target above, rows without an early hook: AUC of the daily rank of A. SUPPORTED iff
  the frozen bootstrap (seed 2, 2,000 draws, pitchers resampled) lower bound at 0.625% is above 0.5, and the
  whole-group rotation p is at most 0.0125 (each pitcher-season with at least 7 rows rolled by k uniform on 0..n-1,
  shift 0 included; 1,000 rotations, seeds 7000-7999; p = (1 + #{floor >= AUC}) / 1001), **and** both hold for the
  daily rank of A minus its mean by role and decile of the pitcher's appearances so far this season at this level
  (the frozen `apps_season`, known on the day; means over all rows, no label; the means are printed). The daily rank
  is within day and role, so it already removes the calendar day; this guard removes where the day sits in the
  pitcher's own season.
- **P2** MLB 2026 (TRAIN 2024-2025), same target: AUC(M+) - AUC(B0+), frozen gradient boosting. B0+ = B0's features
  plus the eight items of the previous section (start, pitch count, batters faced, pitches in the last inning, runs,
  mid-inning exit, left mid-at-bat, shortfall). M+ = B0+ plus A and dev. SUPPORTED iff the bootstrap lower bound at
  0.625% of the difference is above 0 and the difference is above all of 80 floors in which A and dev are rolled
  together by the same whole-group rule (seeds 8000-8079) before training and scoring.

Floor structure gates for P1: the groups cover the rows once and are pitcher-seasons in time order; the floor values
are those of `rotated_group`; shift 0 is drawn; about a third of shifts fall in the band the frozen rule excluded.
Planted signals under random labels: P1's floor must be beaten by the rank plus 0.3 on positives (p <= 0.0125), and
P2's by A set to 4 on positives and 0 on the other rows with A (above 10 floors). The game-log rule: under random
labels a game-log appearance planted the day after one positive must leave exactly that positive out (planted
defect a17, which makes the rule inert, stops here).

Floor diagnostic before freezing (`diag_after.py` on this notebook, md5 a96dcbef03d67e4cfc9404279ac2e566; the 100-pitcher code-test
sample): the rows and A built once, the random placements redrawn 1,000 times with the frozen generator's form, the
game-log rule applied, P1's rows and rule each time with 200 whole-group rotations (seeds 7000-7199, p over 201, so the
discrete nominal rate at 0.0125 is 2/201 = 1.0%): over the 1,000 redraws (mean 30.9 positives) P1 rejects at p <= 0.0125 in 1.0%, at 0.025 in 1.8%, at 0.05 in
4.1%; z-sd 1.01.

## Reported, not tested

P1 on all outings (a manipulation check: an injury exit should show); the two-sided A^2; IL14 over all usable arm rows
(the frozen target); each season; starters and relievers; early-hook shares among positives and negatives; Triple-A
2024-2026 (A only); P2's permutation importance.

## Power (stated now)

Supply: 571 MLB arm placements with an appearance in the 14 days before (2024-2026, transactions). Measured with
random placements on the full files (the pre-check; every random placement effective the day after its row): of 556 random "arm placements", 326 give a positive (35
gap 0, 195 last row unusable, mostly the frozen 14-day end-of-season rule), and 184 of them have A defined and no early
hook (the plan's "about 300" was an inference and too high). With about 180 positives the bootstrap Bonferroni
half-width is near 0.055, so P1 needs an AUC near 0.56 to 0.57 or more; P2 had 139 positives in 2026 in the pre-check.

## Threats

- An injury inside the outing shows in the outing: hence the early-hook rule for P1 and the visible items in B0+.
- Effective dates may be backdated; the transactions do not give the announcement time.
- Park, weather and tracking calibration move velocity; no adjustment (as every frozen family).
- Real placements sit late in a pitcher's season (position AUC 0.659 on 2021-2023): the position guard in P1
  (deciles of his appearances so far this season).
- Relievers often lack a 60-pitch normal (A defined on 54% of relief rows, 88% of starter rows in the pre-check): the
  target is weighted to starters and long relievers.

## Disclosures

- Read before this file: every printed output of the seven frozen pitcher runs; the planning audit's label-free
  probes (`audit7b/`: transactions counts and gaps 2021-2026, outing velocity features of the code-test sample without
  labels, coverage); the label-free full-file measurements above.
- No outing-day feature has been compared with any placement. The code test and the floor diagnostic used random
  placements only.
- MLB 2021-2023 can later confirm P1 (A only): no outing-day velocity has met a label there. Already read from it:
  placement dates and gaps (this study's motivation), the spin-axis-gap confirmation vs labels, a 2023 model test with
  before-day velocity z, and the position AUC.

## Code test

Random placements only (`SMOKE_FAKE_LABELS=1`), notebook md5 a96dcbef03d67e4cfc9404279ac2e566.

- 100-pitcher sample (RPi5, Kaggle package versions): exit 0. P1 0.6273 [0.4765, 0.7695], whole-group p 0.0110,
  26 positives -> does not pass (lower bound below 0.5); P2 +0.0112 with 8 positives -> NOT SUPPORTED. Planted timing in
  A: +0.3236 against a floor maximum of +0.0104. Planted game-log day: 1 of 38 positives left out, 37 kept.
- Planted defects a1 to a17 (each a one- or two-line change of this notebook, run on the same sample): every one exits
  non-zero. Sixteen stop at the gate they target. a15 (the primary-fastball choice reversed) stops earlier, at the
  look-ahead gate (the reversed choice leaves no usable day for it); with that gate made to return (a15b) it stops at
  the intended gate, "the primary-fastball choice is not the frozen one" (the chosen type's 150-pitch mean then equals
  the frozen FB velocity on 36.1% of MLB rows).
- Full files on Kaggle (GHA run 38594006099): every gate passed (MLB choice and FB velocity 100%, look-ahead gates at
  both levels); P1 0.4715 [0.4168, 0.5283], p 0.871, 184 positives; position-adjusted 0.4718, p 0.876; P2 -0.0149
  [-0.0464, +0.0203] with 139 positives; planted timing +0.3576 against +0.0330. Label-free mean of A_rank by
  apps_season decile: 0.48 to 0.51 except the relievers' first decile (0.6175) and the starters' first (0.4709).

## Not done

No outing-day feature was compared with any placement before this file is on main. The notebook is run once. No change
after the run except a dated amendment that says whether it came before or after any result was printed.

## AMENDMENTS
