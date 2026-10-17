# Pitch shape x swing shape matchups (development, not yet frozen)

**Question.** Lineups, pinch hitters and reliever choices are usually made from each player's quality and handedness.
Does the *shape* of today's pitches (velocity, movement, spin, release, approach angle) interact with the *shape* of
a batter's swing (bat speed, attack angle, swing path tilt, timing / contact point from MLB bat tracking) beyond what
the batter's and pitcher's quality already explain?

**Status.** Development only: models are fitted on 2024 and scored on 2025. 2026 is never read before the freeze;
the confirmatory test is 2024+2025 -> 2026, run once. The plan, including every revision, is [`PLAN.md`](PLAN.md)
(earlier versions are kept verbatim above later ones). Code for each development run sits in the sibling
`pitch-swing-matchup-d*` directories; each one is a Kaggle kernel on the dataset
`yasunorim/mlb-statcast-bat-tracking-2024-2025`.

## Design in one paragraph

Pitch-level LightGBM models that differ only in which features may interact (`interaction_constraints`):
**Q+** lets pitch shape interact with the batter's quality block (27 causal as-of quality columns plus a quality score
built from the swing shape) but not with swing shape; **S** is unconstrained. The quantity is the out-of-sample loss
difference Q+ - S on 2025 pitches (log loss for whiff given swing, squared error for xwOBA given contact), with a
product Exp(1) bootstrap over batter x pitcher x game. Rivals that development has to exclude, each with its own check:
quality proxy (Q+), approach to the pitch (swing shape residualised on pitch location, velocity, movement and count),
batter identity (a donor floor that gives each batter another batter's whole swing-shape trajectory, matched on the
k-th tracked swing), seed noise, a level check with labels drawn from Q+, and planted effects that must pass (attack
angle x approach angle) or must fail (a per-batter effect with no shape link).

## Development results so far (2024 -> 2025)

Loss difference Q+ - S on 2025 pitches with 95% bootstrap intervals (positive = the swing-shape interaction helps).

| run | whiff given swing | xwOBA given contact | run value per pitch |
|---|---|---|---|
| D2 (smaller quality block, raw swing shape) | +1.41e-3 [+4.2e-4, +2.30e-3] | +4.21e-4 [+4.2e-5, +7.7e-4] | +4.7e-7 [-3.2e-5, +3.6e-5] |
| D3 (rival = Q+, residualised shape, seed 1 of 5) | +9.4e-5 [-7.3e-4, +8.5e-4] | +1.3e-5 [-2.8e-4, +2.8e-4] | -6.7e-7 [-2.8e-5, +3.2e-5] |

- Against the richer quality rival (Q+), the shape x shape gain seen in D2 is no longer separable from zero on any target.
  Across 5 seeds the whiff difference ranges +9.4e-5 to +8.9e-4 (sd 3.0e-4); xwOBA -9.5e-5 to +1.1e-4.
- Quality x pitch shape (A - Q+) is clearly present: whiff +3.07e-3 [+1.81e-3, +4.29e-3], xwOBA +7.4e-4 [+3.4e-4, +1.14e-3].
- Batters unseen in 2024 (75 batters): whiff +6.0e-4 [-2.9e-3, +3.8e-3].
- Level check: 0/40 (whiff) and 0/30 (xwOBA) null label sets gave a bootstrap lower bound above zero.
- D3's donor floor and planted effects are not read: a code audit found the D3 donor floor favoured the real world
  (donor shapes measured on fewer swings) and the no-shape plant was unlearnable. D3b fixes both; its planted-defect
  test (8 of 8 defects stop at their intended gate; large attack-angle x VAA plant PASS, large no-shape plant FAIL) is
  done and D3b itself has not run yet.

Nothing here is a claim. The only claim this study will make comes from the frozen 2026 test, worded as fixed in
`PLAN.md`.
