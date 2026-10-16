# Pre-registration: confirming the last-outing signal on unseen seasons (MLB 2021-2023)

Status: FROZEN when merged to main (notebook `pitcher-after-confirm/pitcher_after_confirm.py`, md5 `b6a7dbc8669c3d7abc1cd6f8f84b8c5d`; input
datasets `yasunorim/mlb-statcast-2021-2023-pitch-subset` (md5s as in FREEZE_pitch_ssw_confirm.md) and
`yasunorim/mlb-statcast-2021-2023-outing-columns` (files below), and for the regression run the frozen MLB 2024-2026
files of `yasunorim/mlb-statcast-bat-tracking-2024-2025`); the merge commit is recorded in the README line of this
study. The design history (PLAN v1 to v8.2, five diagnostics, the opus audits) is in
`pitcher-after-confirm/PLAN_after_confirm.md`.

## Why this test

`FREEZE_pitch_after.md` (commit `5bbe6ed`, run once, GHA `38657482176`) found P1 SUPPORTED on MLB 2024-2026: for the
last appearance 1 to 14 days before an arm placement, outings without an early hook, the daily within-role rank of A
(the day's primary-fastball velocity against his last 60 of that type, on his own scale; slower = higher) had AUC
0.6059 (Bonferroni lower bound 0.5629), above all 1,000 whole-group rotations of its floor (mean 0.533, p 0.001), and
0.6065 after the position guard. An independent rebuild matched (AUC 0.605934). P2 (gain over what anyone watching
saw, MLB 2026) was not supported.

Diagnostics before this file showed that P1's floor did not control where the positives sit. The last outing before
a placement is almost always followed by a long absence (1 of 314 positives, against 96.6% of the other rows, had a
next MLB row within 7 days), and an outing followed by a long break is slower than other outings even when no injury
list follows it (diagnostic 3, read through PLAN v8.1: about 106 such 2021-2023 rows as pseudo-positives gave a pool
AUC 0.5388 against a floor mean 0.4989; 2024-2026 0.5105 against 0.4987). So this test asks once, on MLB 2021-2023:
**is the last outing before an arm IL placement slower than the last outing before a comparably long break with no IL
placement?**

## The one primary (fixed before any 2021-2023 outing-day feature meets an arm label)

**C2** MLB 2021-2023 pooled, scores `A_rank` (the rank of A within (day, role)) and `A_rank_pos` (that rank minus its
label-free mean by role and decile of `apps_season`), as frozen.

**Pool** (label-free filter, as in PLAN v7.1): every MLB row with `use14` (no other-level outing
within 14 days and 14 days before the season's end), no early hook, both scores finite, and a next MLB row 8 or more
days later or none.

**1** = the frozen P1 positives in the pool (the last row before an arm placement effective 1-14 days later).

**0 = a clean long break**: a pool row with no IL placement within 14 days (`~il14`) whose next MLB row is at least L
days later or none, where **L = 11 in 2021 and 16 in 2022-2026** (the pitchers' IL minimum plus one: 10 days in
2020-2021, 15 from 2022; fixed by the rule, not by data), **and with no IL placement at either level (7/10/15/60-day
and full-season lists; `il_by`) from the day of the row to the day before the next MLB row, or to the season's last
MLB date for a "none" row**. Options, designations, releases and the restricted list are not IL placements;
unrecorded injuries remain among the 0 rows. **Every other row is left out.**

**Strata**: role (the frozen pre-day start share) x kind (none / not none). Merge, by counts only and before any score
is read: a stratum with positives needs at least 5 zeros, else the two kinds of that role merge; never across roles; a
merged stratum that still has fewer than 5 zeros is kept and counted (for the real labels and every draw); a stratum
with positives and no zeros cannot enter an AUC and is dropped from the statistic, and the number dropped is printed
(diagnostic 5 counts every 2021-2023 role x kind cell at 18 or more zeros, so no merge and no drop is expected). No
calendar strata: A_rank is ranked within the day and role, so its level does not drift with the calendar (the counts of
1 and 0 rows by calendar fifth are printed, without A).

**Statistic**: stratified AUC = sum over strata of k_s x AUC_s / K (average ranks within strata).

**SUPPORTED** iff, on both lines (A_rank and A_rank_pos), both floors are beaten:
1. **floor P (the clean-break contrast)**: exact stratified label permutation (in each stratum as many rows as real
   positives, uniformly without replacement from that stratum's 1 and 0 rows), 1,000 draws, seeds 7000-7999,
   p = (1 + #{draw >= stat}) / 1001 <= 0.025 (a non-computable statistic gives p NaN, which fails);
2. **the frozen whole-group rotation floor** on the frozen C2 rows with the frozen statistic: p = (1 + #{floor >=
   AUC}) / 1001 <= 0.025 over 1,000 rotations by the frozen `rotated_group` and `floor_structure` (seeds 7000-7999).
   `p1_stat`, which computes it, is new code in this notebook, held to the frozen run by the regression gate (section
   1); the frozen P1 rule's bootstrap bound (lo > 0.5) and its alpha 0.0125 are not used here. The rule is `decide`
   (new): both p <= 0.025, NaN fails.

This is an intersection-union test: its size is at most 0.025 per line. Floor P permutes labels across pitchers and
does not protect against pitcher traits (the clean-break zeros are partly a different set of pitchers: fringe,
released, demoted); the rotation conjunct does.

**Pre-declared claims.** SUPPORTED -> "in 2021-2023 the last outing before an arm IL placement is slower than the last
outing before a comparably long break with no IL placement". NOT SUPPORTED, by which floor failed (printed per line):
floor P failed on a line -> "not confirmed: in 2021-2023 the slowdown before an arm IL placement is not distinguished
from the slowdown before a long break without one (power as below)"; floor P passed on both lines but the rotation
floor failed on a line -> "not confirmed: in 2021-2023 the contrast with clean long breaks passed, but the frozen
statistic did not exceed the whole-group rotation floor, so it is not separated from differences between pitchers";
both apply -> both sentences. Neither claim says the slowdown precedes the injury (placements are often backdated: a public signal after
the outing, before the announcement), and a SUPPORTED C2 is not called arm-specific unless (c) below passes.

## Reported, not deciding (2021-2023 after the run; 2024-2026 already in diagnostic 5)

(a) arm positives vs short-break no-placement rows (role x calendar fifth strata), beside clean long vs short breaks
(the diagnostic-3 contrast rebuilt with the cleaning rule), with a pitcher bootstrap 95% interval of the difference of
their excesses over floor P; (b) non-arm IL last outings vs the same clean long breaks; (c) arm vs non-arm (the v7.2
secondary; "arm-specific" only if p <= 0.025 on both lines); (d) among clean long breaks, the within-stratum Spearman
of the exact gap with A_rank; (e) the count of long breaks removed by the cleaning rule; the stratified AUC with the
positives reweighted to the zeros' calendar-fifth distribution within role x kind (late-season "none" zeros include
demotions after the Triple-A season ends, which `other14` cannot catch); balance of age, n_prev_il, own_scale and
starts share; **the same contrast without the cleaning rule** (zeros = long-gap pool rows with no IL placement within
14 days) beside the verdict. Also, as frozen: the frozen C2 statistic and its rotation floor alone, the bootstrap
lower bound, effective the next day vs 2-14 days later, each season, starters and relievers, all outings, mean dev
(mph), early-hook shares.

## League constants

A's scale uses a league-typical scale (MLB 2025, by season role) and the early-hook rule a league median pitch count
by start flag (MLB 2025) as the fallback of a pitcher's usual count. Both are taken, as exact floats, from the
regression run of the same session (section 1) and asserted finite; the confirmation asserts that it used them. The
2022 median pitch count is printed beside them and cannot change the verdict. Values in diagnostic 1 (the same
regression code on the frozen 2024-2026 files): scale relievers 0.6779583739316144, starters 0.6477022008547039 mph;
median pitch count relievers 16, starters 89 (2022: 16 and 89).

## Data (new input; fixed now)

`yasunorim/mlb-statcast-2021-2023-outing-columns`, built by `pitcher-confirm-outing-2021-2023/build.py` (GHA
`38675834805`; 2021 712,320 rows in 2,429 games, 2022 710,210 in 2,430, 2023 720,684 in 2,430): the subset's builder
run unchanged (its md5 `f5f7de1ee64e5ccb76ae7c9232eeff58` and dataset4's fetch/parse md5
`593784729763616cdb2519cd9ff7ee5a` asserted) with `inning_topbot`, `bat_score`, `post_bat_score` kept in addition to
the 23 columns. Added gates: Top/Bot only; scores filled, post >= bat; one half per at-bat; bat_score never decreasing
within a half; post_bat_score equal to the next pitch's bat_score within a half on all but 0.1% (45 of 712,320 pitches
in 2021, 34 of 710,210 in 2022, 52 of 720,684 in 2023; (game, inning, pitcher) in both halves 0 in each season). In the
notebook, every key and all 23 shared columns of the new files equal the pinned subset's. md5: `outing_2021.parquet`
4865ef17b5d991047404ca06ae5589a9, `outing_2022.parquet` 7a2f8eeb6336cc826369c4346d30a256, `outing_2023.parquet`
0ee0149b93c8cacca91e075fd3465202.

## Code: the frozen notebook, MLB only, through the join

The notebook fetches the frozen notebook at `5bbe6ed` (md5 a96dcbef03d67e4cfc9404279ac2e566) and applies
replacements: 9 in the notebook (base replacements passed in; equality gates made MLB-only; league constants; MLB
pitches by the join and no Triple-A context; no Triple-A outings, unseen days, target; target gates MLB only) and 12 or
15 in the frozen base (seasons, MLB md5s and the base's own league season 2022 in the confirmation only; no Triple-A
pitches, rows or labels; Triple-A season dates from the StatsAPI schedule; history and game-log years), each listed
and counted in the log. The notebook is cut before its P1 section and the frozen `rotated_group` and
`floor_structure` (with N_ROT) are appended; `p1_stat`, `decide`, the pool, the clean-break labels (`clean_labels`,
`il_between`), strata (`merge_kind`), statistic and permutation (`clean_test`) are new code in this notebook.

**Section 1, regression gate (before any 2021-2023 row meets a label):** the same code on the frozen 2024-2026 files
must reproduce the frozen run: MLB rows and labels (61966, 1180, 53384, 3553), AUC(B0) 0.5644912856858865, P1 AUC
0.6059334572407536, Bonferroni bound 0.5628716317318954, position-adjusted 0.6065369360478149 (each to 1e-12), 314
positives in 30,492 rows, floor mean 0.5334 and 97.5th percentile 0.5587 (printed digits); the 95% and Bonferroni
bounds also equal the frozen `boot()`.

**Section 2:** the confirmation on 2021-2023. The joined outing columns must again pass the builder's gates. The
confirmation may differ from the regression only by three declared base replacements, asserted.

## Diagnostics before freezing

**Diagnostic 1** (Kaggle, GHA `38703478080`): on 2021-2023 rows built with random placements only, random positives
placed in the bins of the real 2024-2026 P1 positives (fifth of the pitcher-season x days to his next MLB row) gave
AUC mean 0.5632 against a whole-group rotation floor mean 0.5205; in 2024-2026, 1 of 314 positives but 96.6% of
negatives had a next row within 7 days, and 44.0% of positives were the pitcher's last row of the season. The last
outing before a NON-arm placement 1-14 days later: 252 positives, AUC 0.5569.

**Diagnostic 2** (Kaggle, GHA `38732450611`, the v5 design): pseudo-positives matching the exact gap and position
passed the v5 rule in 0.133 of 1,000 draws (limit 0.05) and the 2024-2026 non-arm last outing passed it; v5 was not
frozen. Its pseudo-positives sat on 2021-2023 rows with long exact gaps, most of them real placement last outings, so
its AUC 0.558 carries some of the 2021-2023 IL signal (all causes mixed, arm not separated).

**Diagnostic 3** (Kaggle, GHA `38811390746`, the v7.2 design, notebook `d1289f4`): by the rule fixed before it ran,
C2 was not frozen: the exact gap was selected as a covariate (within-stratum AUC 0.879, Spearman with A among 0 rows
+0.067), and the 4x merge had collapsed the strata to role only. Its null calibration (item 4) drew pseudo-positives
among no-placement pool rows by exact gap key, i.e. in effect on the long breaks with no placement: pool AUC 0.5105
(2024-2026) and **0.5388 (2021-2023)** against floor means 0.4987 / 0.4989.

**Diagnostic 4** (Kaggle, GHA `38829424438`, counts only): no 2024-2026 arm positive has a gap of 8-15 days; of the
329 2024-2026 pool zeros with gaps of 16 or more, 233 are last outings before a non-arm IL placement and 94 have no
placement within 14 days (63 "none"); 2021-2023 has 1,174 no-placement pool rows, 112 of them with gap >= 16 or none
(69 none). Stratifying by gap therefore compared arm IL with other IL; this file's contrast replaces it.

**Diagnostic 5** (Kaggle, GHA `38845620755`, this design on notebook `e82f5e7`, md5 `0cd200dbe4763671b124a1c7fee95006`; the frozen
notebook differs from it only by what the FREEZE audit asked for, all without A on new data: a printed count of 1 and 0
rows by calendar fifth, and two gates on `clean_test` in the fake-label code test (its per-stratum AUC and its NaN path);
the diff is in the PR): on 2024-2026 (labels read before), the deciding contrast: 313 arm positives vs 89 clean long breaks (strata role x kind: 85/108, 76/127, 90/96, 62/71 positives/rows; 5 long breaks removed by the cleaning rule), stratified AUC 0.5923 / 0.5924 (A_rank / A_rank_pos), floor P mean 0.5014 / 0.5013, 97.5th 0.5891 / 0.5895, **p 0.0250 / 0.0240** (at the boundary); calendar-fifth reweighted 0.5863 / 0.5859; E = +0.0908; balance (zeros / positives) age 31.8 / 29.5, n_prev_il 2.39 / 2.37, own_scale 0.655 / 0.676, starter share 0.169 / 0.486;
the frozen rotation floor beside it: rotation p 0.0010 on both lines (both floors pass on 2024-2026, by a hair on floor P); reported items (a)-(e): without the cleaning rule 0.6070 / 0.6070, p 0.0170 / 0.0190 (313 vs 94); (a) arm vs short no-placement (role x fifth) 0.5854, excess +0.0845, clean long vs short 0.5243, excess +0.0271, difference +0.0573, pitcher bootstrap 95% [-0.0174, +0.1286] (A_rank; A_rank_pos +0.0559 [-0.0190, +0.1276]); (b) non-arm IL last outings vs clean long breaks 0.5773 / 0.5795, p 0.0729 / 0.0689 (235 vs 89); (d) Spearman of exact gap with A_rank among not-none clean long breaks -0.0094 (relievers, 23) / +0.0294 (starters, 6). On 2021-2023, counts only
(no A statistic was computed): clean long breaks by season x role x kind 2021: relievers 44 not none + 17 none, starters 47 + 2; 2022: 3 + 16, 6 + 7; 2023: 10 + 15, 6 + 9 (182 in all; 6 removed by the cleaning rule). **2021 holds 110 of the 182, 91 of them not none**. Two possible reasons, not checked: L is 11 there (gaps of 11-15 days count), and before May 4 there were no Triple-A games, so a pitcher optioned to the alternate site has no other-level outing and can enter as a clean long break. This was seen in counts only, after the design was fixed; by v8.1 item 5 it does not reopen the design. It is disclosed here, the per-season results are reported, and the rotation conjunct guards pitcher traits; arm positives by season x gap
class before any A-derived filter 2021: <=7 1, 8-15 21, 16-30 32, 31-60 33, >60 22, none 76; 2022: 0, 1, 37, 36, 24, 72; 2023: 0, 1, 35, 25, 34, 75 (525 in all before any filter; 433 with use14, no early hook and a gap of 8+ days or none). The one 2021 positive with a
gap below 11 days has a gap of 7 or less and is outside the pool (gap 8+); no 2021 positive has a gap of 8-10. The 2022
and 2023 positives with gaps of 8-15 days (1 + 1) are kept as positives, while a 0 row in those seasons needs a gap of
16 or more. **Power** at the
2021-2023 sizes (labels permuted within strata on 2024-2026, +E and +E/2 planted on the positives' A_rank, E = the
2024-2026 excess over floor P; the conjunction with the rotation floor; pitchers resampled from P1 and the contrast):
under the nulls the conjunction passed 0.000 of 200 draws (floor P 0.015, rotation 0.020); planted +E/2 (+0.0454): floor P 0.500, rotation 0.575, both **0.280**; planted +E (+0.0908): 0.950, 0.980, both **0.930**; the real 2024-2026 contrast bootstrapped over pitchers at the 2021-2023 sizes without planting: both 0.820 (100 draws). Zeros by (role, none): 57, 48, 59, 18; N1 = 433. N1 (the 2021-2023 positive count used) is an upper bound (counted before the finite-A filter), so this
power is optimistic. The rotation half of the power uses the full 2024-2026 P1 rows, not rows resized to 2021-2023, and
both floors in the power use 200 draws (p over 201), not 1,000. Low power does not reopen the design.

## Threats

- Placements are often backdated; the signal is a post-outing public signal, not a sign seen before an injury.
- The 0 rows are few per pitcher and include pitchers who were demoted or released for performance; floor P does not
  guard pitcher traits, the rotation conjunct does.
- "No IL placement" is not "no injury": unrecorded injuries stay among the 0 rows, which works against SUPPORTED.
- Late-season "none" rows include demotions after the Triple-A season ends (`other14` cannot see them); the
  calendar-reweighted AUC is reported beside the verdict.
- A_rank ranks within (day, role) over all rows, so a 0 row's rank depends slightly on the A of placement rows on the
  same day.
- 2021: seven-inning doubleheaders, the extra-inning runner, a 10-day IL minimum for pitchers (15 from 2022, hence L)
  and an alternate site before May 4 (no Triple-A games).
- The frozen base drops Ohtani's TEST season (2023 here), as in the sixth-family confirmation.

## Disclosures (before any 2021-2023 outing-day feature met an arm label)

Read from 2021-2023 before this file: placement dates and gaps; the sixth family's confirmation (C1 vs arm labels,
before-day fastball gap); a 2023 model test with before-day velocity z (M2 - B0 -0.004); the label position AUC 0.659;
**diagnostic 1's 2021-2023 A read** (pseudo-positives in bins of calendar fifth x days to the next MLB row,
mostly long gaps, so it carries 2021-2023 IL signal: AUC mean 0.5632 against a rotation floor mean 0.5205);
diagnostic 2(a) as above; **diagnostic 3 read the aggregate A of about 106 of the 112 2021-2023 long-gap no-placement
rows (pool AUC 0.5388 against those rows as pseudo-positives, role-only strata), and the cleaning rule (no IL
placement until the next MLB row) was added after that read**. The rule removes long breaks that end in an IL
placement, which should make the zeros faster, i.e. it moves toward SUPPORTED; the contrast without it is reported
beside the verdict. On 2024-2026 it went the other way: 0.5923 with the rule against 0.6070 without it (5 rows
removed). Counts of diagnostics 4 and 5 (no A). From 2024-2026: every printed output of the frozen run and
the independent rebuild, including the descriptive split (next day 0.629, 2 to 14 days 0.552), and the outputs of
diagnostics 1-5.

## Code test

On RPi5 in a venv with Kaggle's versions (numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1), on a 100-pitcher sample of
all six seasons with random placements (`SMOKE_FAKE_LABELS=1`; no verdict is written in that mode and the verdict line
is labelled FAKE LABELS): the notebook (md5 `b6a7dbc8669c3d7abc1cd6f8f84b8c5d`, LF line ends) exits 0. Label-free unit gates at load:
L per season (a 2021 gap of 11 is long, a 2022 gap of 15 is not), the clean-break labels on synthetic rows (r0-r13: an
IL placement on the day before the next MLB row removes the row, a "none" row with a placement after its day is
removed, only `il14` removes a 2021 row whose placement falls after the day before its next row; the same table without
the cleaning rule), the merge (`merge_kind`: never across roles, the 4/5-zero boundary, a stratum without positives
not merged, a role left as one stratum kept), and every pool and secondary label rule of the previous version. On the
sample: clean long breaks are a subset of `pool_base & ~il14` (and no `quiet` row has an IL placement within 14 days,
by `il_between`); a clean-contrast stratum never crosses roles and never mixes kinds unless its role merged; the
deciding call is `clean_test` (asserted by its reweighted output), not the old pool test. The subset assert holds by
construction on the real code and only guards changes to it (planted defect v5). With 30 random short no-placement
rows as the clean contrast's fake positives: a pure stratum score gives the same plain AUC in every draw, the
stratified AUC equals the weighted per-stratum AUCs, and with no zeros the statistic and p are NaN; the same three on
the old pool test with 30 random pool rows as positives; planted +0.3 gives p 0.001 on both floors and both lines and the verdict SUPPORTED, -0.3 and either line
alone give NOT SUPPORTED; the decision rule is asserted on fixed inputs including a NaN p; floors are 1,000 draws.

Planted defects (each run from its own mutant; md5 in the log's first line, log newer than the mutant; each must stop,
exit not 0, with the intended gate's message): `conf_chain9.log` on notebook md5 b6a7dbc8: code test exit 0; the 41 inherited defects plus k5b, k9c and k6c (44) were run, and **39 stopped at the intended gate** in that chain. **c7b, c8 and c9** hit the RPi5 network outage of 2026-10-16 16:28-18:50 JST (StatsAPI `/transactions` failed; DNS of raw.githubusercontent.com failed) before reaching any gate; rerun by `conf_chain9b.sh` on the same notebook md5 and the same mutant md5s (the first logs kept as `*.netfail.log`), all three **stopped at the intended gate** (`conf_chain9b.log`). **k5** and **k6** stopped one gate earlier, at the new clean-contrast gates ("a clean-contrast draw moves positives across strata"; "the clean-contrast stratified AUC differs from the per-stratum AUCs"), because both tests share the draw function and the `Strat` ranks; **k5b** (in chain9) and **k6b** (`conf_chain9b.sh`), the same defects with that earlier gate switched off, stopped at the intended gates ("a draw moves positives across strata"; "the stratified AUC differs from the per-stratum AUCs"). The new k9c and k6c stopped at their clean-contrast gates. Every planted defect (44, plus k6b) is therefore caught at its intended gate, judged as in the chain: the mutant's md5 in the log's first line, the log newer than the mutant, the intended message in the log, exit not 0. Before the audit additions, conf_chain8.log on notebook md5 0cd200db: code test exit 0; **40 of 41 stopped at the intended gate**: inherited c1-c9, c7b, c13, c14 (data, join, regression gate, league constants, bootstrap), k1-k4, k6-k18 (label rules, merges, statistic, decision rule, floor length; k18 now aimed at `clean_test`), and new v1-v11 (L per season, long boundary, cleaning to the day before the next row, no cleaning, zeros outside `pool_base & ~il14`, the 5-zero boundary, a stratum without positives merged, one stratum across roles, the verdict on the old pool test, the `quiet` gate off, kind ignored). **k5** (draws ignore strata, in the draw function both tests share) stopped one gate earlier, at the new clean-contrast draw gate ("a clean-contrast draw moves positives across strata"); **k5b**, the same defect with that gate switched off, stopped at the intended gate ("a draw moves positives across strata"; `k5b.sh`, judged as in the chain: md5 in the log's first line, log newer than the mutant, exit not 0).

## Not done

No 2021-2023 outing-day feature was compared with an arm placement before this file is on main. The notebook is run
once. No change after the run except a dated amendment that says whether it came before or after any result was
printed.

## AMENDMENTS
