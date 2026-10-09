# Does Ohtani's pattern hold for every MLB hitter? Frozen 2026-10-09, before any all-MLB pitch data is fetched

The Ohtani notebook (FREEZE.md, same folder) found two things in one player:
(a) a bat-speed drop of 3.07 mph over ten game days (ending 2026-08-04), larger than any 10-day change in
    his and Seiya Suzuki's healthy seasons (minimum -1.80), a month before his injured list; it recovered;
(b) level flags on process signals (EV90, hard-hit, bat speed) went up about as often as the results flag
    (wOBA), sometimes earlier, sometimes with nothing following.
One player and two events cannot say whether either is a pattern. This file fixes the test on all MLB
hitters before any of their data is looked at. Nothing below is changed after the data is seen; any
change goes in an AMENDMENTS section with its date and reason, before the run.

## Data
- Pitch-by-pitch Statcast, regular season 2024, 2025, 2026 (bat tracking era), all batters: the Kaggle
  dataset "MLB Statcast + Bat Tracking (2024-2026)" (yasunorim/mlb-statcast-bat-tracking-2024-2025).
- Injured-list placements: MLB StatsAPI transactions for every batter in the data; the outcome date is
  the effective date (retroactive date when given). Any IL length (7/10/15/60-day).
- Excluded: Shohei Ohtani 2026 (the source of the hypotheses). Pitchers' plate appearances are not used
  (batters whose primary position is P are excluded).

## Signals (same definitions as the Ohtani notebook; day t uses only events strictly before t, same season)
- BS100(t): mean bat_speed of the last 100 swings with bat_speed >= 50.
- D10(t) = BS100(t) - BS100 on the batter's game day 10 game days before t (both defined).
- Level flags: EV90 (last 40 batted balls), hard-hit (last 40), bat speed (BS100), whiff (last 150
  swings), wOBA (last 100 PA); line = 5th percentile (whiff 95th) of the same signal over every day of the
  batter's previous season (>= 300 PA there); flag = worse than the line on 3 game days in a row.

## Outcome
IL(t) = the batter has an IL placement with effective date in [t, t + 30 days].

## H1 (primary): a sharp bat-speed drop precedes the injured list
- Exposure: D10(t) <= -2.0 mph (fixed here: between the healthy-reference minimum -1.80 and Ohtani's -3.07).
- Statistic: relative risk RR = P(IL | exposed) / P(IL | not exposed) over batter-days, all three
  seasons pooled; 95% CI by bootstrap over batters (2,000 resamples, seed 0).
- Supported if the CI lower bound is above 1. Also reported, whatever the result: RR per season,
  positive predictive value, the share of IL placements with an exposed day in the 30 days before
  (sensitivity), exposure episodes per batter-season with no IL in the next 30 days (false alarms),
  and the AUC of -D10 for IL(t) (threshold-free).
- Secondary (direction fixed now): RR is larger for upper-body injuries (description contains shoulder,
  elbow, wrist, hand, finger, thumb, biceps, triceps, forearm, back, oblique, rib, intercostal, neck)
  than for lower-body ones (hamstring, quad, calf, knee, ankle, foot, hip, groin, toe, heel, achilles).
  Other or unspecified descriptions are in the primary only.

## H2: do process flags warn earlier than the results flag?
- For each level flag: RR = P(IL | flag up on t) / P(IL | not up), same CI method, batter-seasons with a
  previous season of >= 300 PA.
- Ohtani's notebook predicts no clear lead: process RRs are not above the wOBA RR. Reported as "process
  leads" only if a process signal's CI lower bound is above the wOBA RR point estimate.

## Known limits, stated before the result
- The injured list is an administrative record; an acute injury (a pulled hamstring on one play) cannot
  be preceded by anything in swing data. The upper-body split is the closest this data gets to the
  "slowly worsening condition" idea.
- A batter playing hurt may also be benched before the IL, which removes his days from the data.
- Returning from an IL stint can itself lower bat speed; days within 30 days after an activation are
  reported both included (primary) and excluded (sensitivity).
