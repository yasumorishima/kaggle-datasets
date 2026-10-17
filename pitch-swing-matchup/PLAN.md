# PLAN (draft v0, 2026-10-17): pitch shape x swing shape matchups

## Question

When a manager picks a pinch hitter, a lineup against today's starter, or a reliever for a hitter, the usual inputs are each player's quality and handedness. **Does the shape of the pitcher's pitches, read against the shape of the batter's swing, predict the plate appearance beyond what each player's own quality (and handedness) already predicts?**

"Shape" means, on the pitching side: per pitch type the velocity, induced vertical break, horizontal break, spin rate and spin axis, release height and side, extension, arm angle, vertical approach angle (VAA), the usage mix and how the pitches relate to each other (velocity gap and movement separation between the fastball and the secondary pitches, release consistency), and the pitcher's form earlier in the same game. On the batting side: bat speed, swing length, attack angle, swing path tilt, attack direction, contact depth (intercept), swing and chase tendencies, and where in and around the zone, and against which pitch shapes, he has done well or badly.

Handedness and quality are **controls**, not the subject.

Literature checked 2026-10-17: Bayesian/ML plate-appearance models (log5, Singlearity, BYU arXiv 2511.17733) use player quality, handedness and recent form; one non-peer-reviewed project (LKohn4/mlb-matchup-projections) reports that pitch-mix matchup features change log loss by less than run-to-run noise. Bat-path articles (Driveline, MLB.com, BP 2025) argue that a swing whose attack angle matches the pitch's plane widens the margin for good contact. No pre-registered held-out test of shape x swing interaction on plate-appearance outcomes was found.

## Data

The public Kaggle dataset `yasunorim/mlb-statcast-bat-tracking-2024-2025` (every regular-season MLB pitch 2024, 2025 and 2026 with bat tracking). A plate appearance (PA) is (`game_pk`, `at_bat_number`); its result is on its last pitch row.

**Eligible PAs** (fixed before any result is read):
- `woba_denom == 1` and not an intentional walk (a decision, not a matchup);
- the pitcher is not a position player and the batter is not a pitcher (by `players.csv` position);
- the batter has at least `NB` tracked swings and the pitcher at least `NP` pitches **on dates strictly before the game** (counted from the first date in the dataset; `NB`, `NP` set from diagnostic 0 counts, not from results).

**Outcome**: 6 classes = strikeout, walk or hit by pitch, out on a ball in play (errors and fielder's choice counted here), single, double or triple, home run. The run-value weight of each class is the mean `woba_value` of that class in the training seasons.

## Split

- **Development** (anything may be tried): features and models trained on 2024 PAs, evaluated on 2025 PAs.
- **Final** (once, after the freeze): trained on 2024 + 2025 PAs, evaluated on **2026** PAs. No 2026 result column is read by any diagnostic before the freeze.

## Features (every one causal)

All are computed from pitches on **dates strictly before** the game, except the pitcher's in-game form, which uses only pitches of **earlier PAs of the same game** (`at_bat_number` smaller). Horizontal quantities (`pfx_x`, `release_pos_x`, `plate_x`, attack direction, intercept x) are mirrored to the batter's frame (multiplied by +1 for a right-handed batter, -1 for a left-handed one), so "toward / away from the batter" means the same thing for everyone.

- **C, context** (in both sides of every model): batter side, pitcher hand, platoon flag, home or away, home team (park), inning, outs, runners on base, score difference, the pitcher's times through the order in this game, starter or reliever, month.
- **BQ, batter quality**: shrunk prior rates of the 6 classes (all pitchers and same-hand pitchers), prior PA count.
- **PQ, pitcher quality**: the same for the pitcher (allowed), prior batters faced.
- **BS, batter swing shape**: mean and spread of bat speed, swing length, attack angle, swing path tilt, attack direction, intercept x and y; swing, chase, zone-contact and whiff rates overall and by pitch family (fastball, breaking, offspeed); prior run value per pitch by zone region (9 in-zone cells and 4 chase regions), shrunk to his overall.
- **PS, pitch shape**: per pitch type (usage against this batter side) the means of velocity, induced vertical break, horizontal break, spin rate, spin axis, VAA, release height, release side, extension, arm angle; usage shares; fastball-to-secondary velocity gap and movement separation; release-point spread; location shares by zone region against this batter side; in-game form so far (primary fastball velocity minus his own norm, pitch count before this PA).
- **X, explicit matchup features** (built from both sides at once):
  - X1 shape-neighbour deviation: for each pitch type k of the pitcher, the batter's prior pitches seen (from any pitcher) are weighted by a Gaussian kernel on the standardized shape vector (velocity, induced vertical break, mirrored horizontal break, release height, mirrored release side, extension, VAA) around type k. The weighted whiff per swing, called-strike-plus-whiff per pitch, run value per pitch and xwOBA on contact, each **minus the batter's own overall value**, shrunk by the kernel weight sum; averaged over k with the pitcher's usage against this batter side. Four features.
  - X2 plane match: usage-weighted mean of (batter's mean attack angle + VAA of type k), signed and absolute. Two features.
  - X3 familiarity: kernel-weighted share of the batter's prior pitches whose release point, extension and arm angle are close to this pitcher's. One feature.
  - X4 location match: sum over zone regions of (pitcher's location share against this side) x (batter's shrunk run value per pitch in that region minus his overall). One feature.

## Models

LightGBM multiclass, identical hyperparameters across arms, averaged over 3 seeds. Hyperparameters and the number of rounds per arm are chosen on development (2025 early stopping); the final fit uses the development rounds x 1.2 (more training rows), fixed in the freeze.

- **ADDQ (rival)**: features C, BQ, PQ, BS, PS with LightGBM `interaction_constraints` = { C + BQ + PQ, C + BQ + BS, C + PQ + PS }. Quality may interact with quality across the two players (so a non-log5 combination of the two qualities is in the rival), and each side's shape may interact with its own quality and the context, but **no tree may use a shape feature of one player together with any feature of the other player**.
- **FULL**: features C, BQ, PQ, BS, PS, X, no interaction constraint.
- Reported only: **ADD** (constraint sets C + BQ + BS and C + PQ + PS, the log5-like additive model), **FULL0** (FULL without X, to split the gain between free trees and explicit matchup features).

## Hypotheses (2026 test, decided once)

- **H1 (information)**: mean multinomial log loss of ADDQ minus FULL over eligible 2026 PAs is > 0.
- **H2 (run value)**: with δ = FULL's expected run value minus ADDQ's for each PA and r = actual `woba_value` minus ADDQ's expected value, the slope of r on δ (with intercept) is > 0.

Inference for both: **two-way (pigeonhole) cluster bootstrap over batters and pitchers**, 2,000 resamples, and separately a game-cluster bootstrap; the verdict uses the **wider** of the two. Each hypothesis is SUPPORTED if its one-sided lower bound at α = 0.0125 (Bonferroni over the two) is above 0.

**Placebo gate (both hypotheses)**: 5 placebo FULL models whose X and the cross-side use of shape are built against a **decoy pitcher** (each PA's pitcher replaced by another pitcher of the same season, hand, role and prior-quality tercile; the real pitcher's PQ and PS stay, only the cross pairing for X is broken) and re-fit with the same settings. A hypothesis can be SUPPORTED only if its real statistic exceeds all 5 placebo statistics. (Placebo design to be refined in the audit: the point is to separate "flexible trees fit better" from "the specific pairing carries information".)

Reported, not judged: ADD vs ADDQ (non-log5 quality combination), FULL0 vs ADDQ, per-class log loss, decile table of δ with realized r in wOBA points and runs per 100 PA, the gain by pitcher role (starter vs reliever) and by batter history length, SHAP or gain importance of X, and a worked example (for one 2026 game: the ranked pinch-hit options).

## Diagnostics before the freeze

- **D0 (label-free, running)**: column coverage by season and month, PA and pair counts, prior-history counts at each PA, VAA sanity check, within- vs between-batter spread of swing descriptors.
- **D1 (development, labels of 2024-2025 only)**: build features, fit all arms on 2024 -> 2025, estimate H1/H2 statistics, their bootstrap spread and the placebo distribution in development; check that each feature is causal (planted-future tests: a feature built with the game's own pitches must make a gate fail).
- **D2 power**: from D1, plant a known matchup effect of the size seen in development and of half that size into 2025 labels and measure how often the frozen test would SUPPORT it.

Code tests on RPi5 with fake data and planted defects (future leak in each feature family, a broken mirror sign, an interaction constraint that lets shape cross, a bootstrap that ignores clusters, a placebo that keeps the real pairing) must each stop at the gate meant for it, before the freeze.

## Claim wording (fixed in advance)

- If H1 and H2 are SUPPORTED: "in 2026, knowing how this pitcher's pitches are shaped relative to how this batter swings improved prediction of the plate appearance beyond both players' quality and handedness, by X log-loss units and Y wOBA points between the top and bottom deciles of the matchup adjustment." Not: "managers should", not causal ("matchups cause"), and selection of who faces whom by managers is disclosed as a reason the size may not transfer to new decisions.
- If not: the null is reported as the size bound the bootstrap allows, without "matchups do not exist".

---

# v1 (2026-10-17, after the opus plan audit; v0 above is kept verbatim)

Audit points checked against the definitions above and adopted unless noted.

## What v0 got wrong
- **B1** FULL vs ADDQ changed two things (X added and the constraint lifted), and lifting the constraint also lets one player's *quality* cross the other's *shape*.
- **B2** Shape is a better quality proxy than shrunk outcome rates (Stuff+-like), so FULL could win by a non-log5 combination of the two qualities expressed through shape.
- **B3** Every X carried one-player content: X1 averaged over batters is "how hitters do against pitches like this" (a stuff rating of the pitcher), X3 is mostly how rare the arm slot is, X2 contains VAA and attack angle on their own.
- **B4** The placebo kept the real pairing of PS with BS inside an unconstrained model, and "beat all 5" has no known level.

## Revised design

**Stage 1, rival R** (one player at a time plus quality x quality): LightGBM **regression on xRV** (below) with `interaction_constraints` {C, BQ, PQ, Bscore, Pscore}, {C, BQ, BS}, {C, PQ, PS}. `Bscore` / `Pscore` are each player's own-side predicted xRV from a batter-only model (C, BQ, BS) and a pitcher-only model (C, PQ, PS), out-of-fold by date blocks on the training rows, so shape-based quality may cross the other player's quality (B2) in the rival. Training-row predictions of R used as stage-2 offsets are out-of-fold by date blocks.

**Stage 2, test arm T**: a small LightGBM with `init_score` = R's prediction and features = **X_dc + C only** (no player's own features), so the only new content is the pairing.

**Double-centred matchup features X_dc.** X1, X3 and X4 are inner products of a pitcher vector and a batter vector:
- X1: a_p = Σ_k u_pk W_pk (usage-weighted soft assignment of the pitcher's pitch-type shapes, mirrored to this batter side, to K shape clusters frozen from 2024 pitches, label-free) and D_b = the batter's shrunk per-cluster deviation from his own overall (whiff per swing, CSW per pitch, run value per pitch, xwOBA on contact: one X1 per metric).
- X3: look-cluster assignment of the pitcher's release height, mirrored release side, extension and arm angle, times the batter's share of prior pitches seen per look cluster.
- X4: pitcher's location share per zone region against this side, times the batter's shrunk run value per region minus his overall.
For X = a_p · D_b the double-centring against reference means is exact and cheap: **X_dc = (a_p − ā) · (D_b − D̄)**, with ā the mean of a over pitchers of the same hand eligible as of that date (against this batter side) and D̄ the mean of D over batters of the same side eligible as of that date. Averaged over batters or over pitchers it is zero (up to reference-set membership), so it carries no one-player content.
- X2: the **signed** form Σ_k u_k (AA_b + VAA_k) = AA_b + VAA_p is a sum of one-player terms; **double-centred it is identically zero**, so it is dropped (the audit's version would have been an empty column). The squared form Σ_k u_k (AA_b + VAA_k)² double-centres to **2 (AA_b − ĀA)(VAA_p − V̄AA)** (usage-weighted VAA), which is kept as X2: the attack-angle x approach-angle product.

**Outcome (power, audit S5)**: primary per-PA **xRV** = `woba_value` for strikeout, walk, hit by pitch; `estimated_woba_using_speedangle` for balls in play (`woba_value` where it is missing; the missing count is reported). This removes fielding and park luck from the target. Actual `woba_value` is used in H2.

**Hypotheses, fixed sequence (audit S4)**:
- **H1** on eligible 2026 PAs: mean squared error of R minus that of T on xRV is > 0. One-sided α = 0.025.
- **H2**, tested only if H1 is SUPPORTED: slope of (actual `woba_value` − R) on δ = T − R is > 0. One-sided α = 0.025.
- Reported only: the same on whiff per swing aggregated per PA (mechanism), per-class log loss of multiclass versions, FULL and FULL0 (v0 arms) against R, gain by history-length band, by starter / reliever, and on **starting-lineup batters against the starting pitcher the first time through the order** (least manager selection, audit S6).

**Inference**: one product-weight bootstrap (independent Exp(1) weights per batter, per pitcher and per game; each PA weighted by the product), 2,000 draws, for the MSE difference and the weighted-OLS slope; conditional on the fitted models (stated). AND a **decoy floor**: X_dc recomputed against a decoy pitcher (same date, hand, role and prior-quality tercile, vectors as of the **real** PA's date), stage 2 refit, 99 draws, p = (1 + #{decoy ≥ real}) / 100. A hypothesis is SUPPORTED only if both pass.

**Level check (development only)**: synthetic outcomes with no pairing effect = R's out-of-fold prediction + batter and pitcher shocks + resampled residuals (variances from D1). Bootstrap level from ≥ 200 sets (one stage-2 fit each); decoy-floor level from as many sets as the time budget allows (each needs its own decoys), reported with its binomial interval. If either rejects above α, it is fixed before the freeze.

**Other audit points adopted**: PA state from the PA's **first** pitch row (S2); no `players.csv` position rule (position-player pitchers fall under the NP history threshold; reported); every standardization, cluster centre, bandwidth, shrinkage strength and league mean is estimated on training seasons and frozen, and shrinkage targets use strictly earlier data (S2); a file-access gate fails the run if any season-aggregate CSV is opened (S2); rounds for the final fit chosen by early stopping on 2025 H2 after fitting 2024 + 2025 H1, rehearsed in development (S3); `dump_model()` gate that every root-to-leaf feature set of R lies inside one constraint set, shown to fail when a set is dropped (S1); tunnel distance (pairwise separation of pitch types at ~23.8 ft versus at the plate, from vx0..az) added to PS; batter's bat speed in earlier PAs today versus his norm added to BS.

**Power (D2)**: planted pairing effects of logit-SD-equivalent 0.015 / 0.025 / 0.04 in 2025, run through stage 2, bootstrap and decoy floor; the frozen test's power is written down, not used to reopen the design.

---

# v2 (2026-10-17 night, after development runs D1-D2; v0 and v1 above are kept verbatim)

## What development showed (2024 -> 2025, labels of 2024-2025 only)
- **PA level (D1, D1c, D1d)**: no double-centred pairing feature, including the arsenal-averaged pitch-level model x5, moves PA xRV beyond R (MSE difference -8.2e-6, decoy p 0.75; correlations with R's residual within +-0.006). A small consistent whiff channel (x1_whiff ~ +0.009 with per-swing whiff residual in both seasons).
- **Pitch level (D2)**: three LightGBM models on the same rows and features (pitch shape + location 9, count 2, batter quality rates 5, batter swing shape 7; profile from strictly earlier dates), differing only in interaction constraints: A = {pitch+count}, {count+quality+shape}; Q = {pitch+count+quality}, {count+quality+shape}; S = unconstrained. Q - S on 2025 (positive = S better):
  - whiff | swing (log loss): +1.41e-3 [95% +4.2e-4, +2.30e-3] (0.31% of Q); 5 shape-swap worlds all negative (-2.3e-4 .. -5.2e-4); corr(S-Q, y-Q) +0.070 vs swap +0.025..0.029.
  - xwOBA | contact (MSE): +4.21e-4 [+4.2e-5, +7.7e-4] (0.31%); swap max +2.2e-5; corr +0.060 vs swap +0.018..0.029.
  - run value per pitch (MSE): +4.7e-7 [-3.2e-5, +3.6e-5]: not visible (as at PA level).
  - quality x pitch (A - Q) on whiff: +1.55e-3, the same size as shape x pitch.
- Conclusion carried into v2: the swing-shape x pitch-shape interaction is visible where the mechanism acts (swing-and-miss, contact quality) and drowned where the outcome is mostly noise (run value per pitch, PA xRV). **The confirmatory test moves to the pitch level.**

## Rival explanation that development has NOT excluded
The 7 swing-shape numbers (cumulative means to date) are close to unique per batter, so S can use them as a **batter identifier** and learn individual zone/pitch responses in the training season (memorised hot/cold spots), not a mechanism that transfers between batters with similar swings. The D2 swap floor re-draws partners every date, so it destroys the identifier as well as the mechanism and cannot separate the two.

## D3 (development, before the freeze; same rows as D2)
1. **ID arm**: S_id = S with the 7 shape columns replaced by a 7-dimensional random vector fixed per batter (N(0,1), drawn once, seed fixed), same constraints as S. If S_real is not better than S_id on 2025 (bootstrap), the shape claim is dropped and the plan is rewritten before any freeze.
2. **Unseen batters**: 2025 eval pitches of batters with zero rows in the training season, reported with n; Q - S, its bootstrap interval, and the swap floor on that subset (same swap worlds, subset losses).
3. **Swap floor with 19 worlds** (p resolution 0.05) to see where the real value sits in a larger floor, and the bootstrap lower bound inside each swap world (level: how often a world without the shape link would pass the bootstrap criterion).
4. **Final-fit rehearsal**: fit on all of 2024 except the last 20% of its dates, early-stop there (as D2), rounds recorded; this is the exact procedure the final fit uses with 2024 + 2025.
5. Counts for power: 2025 swings and contacts of unseen batters; projected 2026 counts are not read (2026 stays closed by G1/G6).

## Final test (decided once, after the freeze)
- Fit A, Q, S (and S_id if D3 keeps it as a gate) on all eligible pitches of 2024 + 2025, early stopping on the last 20% of 2025 dates; evaluate on all eligible 2026 pitches.
- **H1** whiff | swing: mean log loss of Q minus S > 0.
- **H2** (tested only if H1 is SUPPORTED) xwOBA | contact: mean squared error of Q minus S > 0.
- **H3** (tested only if H1 is SUPPORTED; judged or reported is fixed after D3 from its power): H1 on 2026 pitches of batters with zero rows in 2024-2025.
- Each hypothesis is SUPPORTED only if (i) the one-sided 97.5% lower bound of the product Exp(1) bootstrap (batter x pitcher x game, 2,000 draws, conditional on the fitted models) is > 0, AND (ii) p = (1 + #{swap world >= real}) / (K + 1) <= 0.05 with K = 39 shape-swap worlds (same date, side, prior-whiff tercile; Q and S refitted per world), AND, if D3 keeps the ID arm as a gate, (iii) the bootstrap lower bound of loss(S_id) - loss(S) > 0.
- Reported only: A - Q (quality x pitch), run value per pitch, per pitch family (fastball / breaking / offspeed) and per count, and the decision translation: for each 2026 PA, the arsenal-averaged S - Q whiff adjustment (the pitcher's prior usage over (shape, location) clusters against this side, evaluated at the batter's profile) in deciles vs realised strikeouts and whiffs per PA, with the bootstrap interval of the top-minus-bottom decile.

## Claim wording (fixed in advance)
- SUPPORTED H1: "in 2026, letting the pitch's shape and location interact with the shape of the batter's swing improved prediction of swing-and-miss beyond letting it interact with the batter's swing and contact rates, by X log-loss units (Y% of the rival)." With H3: "... including for batters the model had never seen." Not causal, not "managers should".
- Otherwise: the bound the bootstrap allows, without "swing shape does not matter".
