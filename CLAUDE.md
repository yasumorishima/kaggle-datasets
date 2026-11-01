# CLAUDE.md — kaggle-datasets

Working rules for Claude Code (cloud or local). This file is the source of truth for this repo; the owner's local notes and the Raspberry Pi are not visible to cloud sessions.

## Session protocol (cloud)
This repo is worked on in Claude Code cloud. The owner starts a session with just 「続き」/「つづきから」 in the environment 「クラウド許可1」 (full network).
1. Read this file first, then continue from the first unfinished item in **Next tasks**. The active research (pitch shape x swing shape) lives on branch `matchup-diag`, not `main`: read `pitch-swing-matchup/PLAN.md` and `README.md` there before touching it.
2. Before ending, update "Next tasks" and add a dated line to "Progress log" **in this file**. It is the only memory that survives between cloud sessions.
3. Cloud can push only its session branch: open a PR to `main` and merge it yourself (own repo). Use a **merge commit (not squash)** when the PR contains a frozen pre-registration commit (`FREEZE_*.md`, an `AMENDMENT`, or the code it pins), so the frozen hash stays in `main`'s history. Report the merged commit.
4. No `gh` CLI in cloud: use the GitHub MCP tools. `workflow_dispatch` returns **403** from cloud, and **every workflow in this repo is `workflow_dispatch`-only today** (see "How to run"), so cloud cannot start a Kaggle kernel or a dataset build until a push-triggered path exists (Next task 1).
5. Reply to the owner in Japanese (です/ます). Never claim anything not run in that session; say where each number was measured (cloud / GitHub Actions / Kaggle). Numbers recorded up to 2026-10-18 were measured on Kaggle kernels, GitHub Actions and the owner's Raspberry Pi.
6. Anything that publishes (a new dataset version, `create`, a public notebook, a subtitle/tag change) needs the owner's approval first. Private development kernels (`"is_private": "true"`) do not.

## Progress log
- 2026-09-27..28: Savant Leaderboards, Pitcher Arsenal, Japanese MLB, ABS Challenges rebuilt in GitHub Actions with publish gates (`*/build.py`, `build.sh`); `update-dataset.yml` gained the build step, `dry_run`, `create`/`public`.
- 2026-10-10: Statcast + Bat Tracking 2024-2026 (version 2) published; hitter anomaly study (`FREEZE_anomaly.md`) not supported; Triple-A replication (`FREEZE_aaa.md` + 2 amendments) not supported.
- 2026-10-11..14: six pre-registered pitcher families (anomaly, in-outing, mix, command + amendment 1, reach-back, spin-axis gap) all NOT SUPPORTED; MLB 2021-2023 spin-axis confirmation NOT SUPPORTED. README updated per family.
- 2026-10-14..16: last-outing study `FREEZE_pitch_after.md` (`5bbe6ed`): P1 SUPPORTED (AUC 0.606), P2 not; independent check `41d840c`; confirmation on unseen MLB 2021-2023 `FREEZE_pitch_after_confirm.md` (`db13c49`, run once, GHA `38905732627`): SUPPORTED, README `e0b15e5`. The pitcher-injury series is closed.
- 2026-10-17 (branch `matchup-diag`): pitch shape x swing shape study: label-free D0, development D1/D1c/D1d/D2/D3 on Kaggle (2024 -> 2025 only), PLAN v0-v2 and results README `5b324c7`. D3: against the quality rival Q+, the shape x shape gain is not separable from zero. D3b (fixed donor floor and plants) passed its planted-defect test off-repo but has **not run**.
- 2026-10-18: ABS carry-over notebook published (`72dfaea`); frozen Triple-A 2026 -> MLB 2027 predictions committed (`b2a41c0`, md5 `90e2625b697ad91096bdc06ca584fd20`); ABS subtitle/tags synced (`5aa0b16`).
- 2026-11-01 (cloud): reachability measured (section below): Kaggle REST via `www.kaggle.com` 200 (11 datasets), `kaggle` CLI 2.2.5 fails with "Authentication required"; Savant and Stats API 200.

## Cloud reachability (measured 2026-11-01, cloud session)
| Check | Result | Proof |
|---|---|---|
| `curl --max-time 60 "https://www.kaggle.com/api/v1/datasets/list?user=yasunorim"` (no key handled; the environment's network secret is attached by the proxy) | 200, 0.28 s | JSON list of 11 datasets, incl. `yasunorim/mlb-statcast-bat-tracking-2024-2025` |
| `pip install kaggle` (2.2.5, kagglesdk 0.1.38) → `kaggle datasets list --user yasunorim` | **fails** | "Authentication required to call the Kaggle API." (no `~/.kaggle`, no `KAGGLE_*` variable in the session) |
| Same with a dummy `KAGGLE_API_TOKEN` (not a real key) | **fails**, same message | CLI 2.x first checks the token by sending it in the request *body* to `api.kaggle.com` (`introspect_token`), so the proxy's header injection cannot help; its endpoint is fixed by `KAGGLE_API_ENVIRONMENT` (prod = `api.kaggle.com`), no way to point it at `www.kaggle.com` |
| `curl --max-time 60 "https://baseballsavant.mlb.com/leaderboard/sprint_speed?year=2025&position=&team=&min=10&csv=true"` | 200, 0.18 s, 36,204 B | 579 CSV lines (header + 578 players), first row Trea Turner 30.3 |
| `curl --max-time 60 "https://statsapi.mlb.com/api/v1/teams?sportId=1"` | 200, 0.20 s | 30 teams |

## What this is
Baseball datasets the owner publishes on Kaggle (user `yasunorim`), each built in GitHub Actions and published only if its gates pass, plus the Kaggle notebooks that use them and pre-registered studies run as Kaggle kernels. Dataset list, links and every study result: `README.md` (main). Pre-registrations: `ohtani-2026-early-warning/FREEZE_*.md`; confirmation plan `pitcher-after-confirm/PLAN_after_confirm.md`.

## Non-negotiable discipline
1. **Pre-register, freeze, run once.** A `FREEZE_*.md` (or `PLAN.md` version) is committed before any outcome is read. A gap found before the run goes into an `AMENDMENT` before any parameter meets a label. Never change code, definitions or the target after seeing results; never sweep windows/thresholds/variants looking for a pass.
2. **Gates must be shown to fail.** Every gate gets a planted defect that stops at *that* gate (check the mutant's log came from the current version, and that a mutant is not stopped by an earlier gate). Planted values must differ from the real ones. A floor must keep its level under null labels: redraw labels many times and count the rejection rate (one seed is not enough). An assert that is true by construction checks nothing.
3. **Matchup study:** the 2026 pitch file is never opened before the freeze (a file-access gate enforces it). The subject is pitch quality/shape x swing shape; handedness and player quality are controls only (owner, 2026-10-16). Quality x pitch shape (A - Q+) is large but is **not** to be promoted to the primary hypothesis. Nulls are recorded as facts; do not write "ceiling"/"impossible".
4. **Test code against Kaggle's library versions** (e.g. pandas 2.3.3 on Kaggle broke the command study, which a pandas 3 test missed). Notebooks run with `enable_internet: false`.
5. Code that produces a result gets an independent review (opus subagent) before it runs; verify every review claim against the code/data before accepting it.
6. **Kaggle limits learned the hard way:** tags max 5 (more = version created but CLI errors and later steps stop); subtitle 20-80 and title 6-50 characters (checked by `update-dataset.yml` before the build); `kaggle datasets version` does not change the title; a new version shifts old column descriptions **by position**, so all of them must be re-entered from `settings.json`; Parquet files with ~115 columns cannot get column descriptions even in the UI (preview times out).
7. **Column descriptions, cover images, title, tags: browser only.** README: "Column descriptions are not accepted through the API". Cloud has no browser on Kaggle, so cloud prepares `settings.json` / metadata and tells the owner what to enter; it must not report these as done.
8. Never name the club whose job posting the owner uses as a career yardstick, in commits, files or PRs (data-source credits are fine).

## Next tasks (in order)
1. **Make GitHub Actions reachable from cloud** (prerequisite for everything below). Add a push-triggered path, e.g. a `requests/*.json` file on `main` that `push-notebook.yml` / `update-dataset.yml` read (keep `workflow_dispatch`), and a step that downloads a kernel's output (`kaggle kernels output`) and commits the log to a results branch, because nothing in the repo fetches kernel output today. Unverified: whether cloud's GitHub token may push changes under `.github/workflows/`; if refused, report it to the owner instead of working around it.
2. **Matchup study: rebuild and run D3b** (branch `matchup-diag`). The D3b code is not in the repo (see "Not in the repo yet"); rebuild it from `pitch-swing-matchup-d3/matchup_d3.py` (md5 `c92acc87`) with the D3b changes described in `pitch-swing-matchup/README.md`: donors matched in event time (receiver's k-th tracked swing <-> donor's k-th), strata = prior-whiff tercile x career-swing tercile x first season, gate that donor shapes are finite on every row and come from >= 0.9 x NB swings, quality score rebuilt in each donor world, quality score cross-fitted over 3 batter folds, residual model cross-fitted over date halves, no-shape plant = per-batter height slope, 19-world plant floor with p <= 0.05. Re-run the planted-defect suite on fake data (8 defects must stop at their gates; large attack-angle x VAA plant PASS, large no-shape plant FAIL) before pushing the kernel (`pitch-swing-matchup-d3b/`, private, dataset `yasunorim/mlb-statcast-bat-tracking-2024-2025`). Read only: the donor floor (19 worlds) for Q+ - S and the default-size plant (power).
3. **After D3b, decide before any freeze** whether "shape x shape" stays the primary hypothesis. If power is too low, write PLAN v3 that changes how shape is measured (per-swing distribution, shape by location) or the target (launch angle, spray direction) — no sweeping. Then: the two-stage final run (fit on 2024+2025 without opening 2026 -> md5 -> predict on 2026), FREEZE, full planted-defect suite, opus code audit, run once.
4. **Dataset upkeep (needs the owner's browser):** Statcast + Bat Tracking cover image still reads 2024-2025 (owner's notes; not visible in the repo). Any new dataset version needs all column descriptions re-entered from `settings.json` in the browser.
5. **Scheduled, not before the MLB 2027 regular season ends:** score `abs-challenges-carryover/frozen/abs_frozen_2027.csv` with the rule stated in the ABS carry-over notebook (do not edit the frozen file).

## How to run
All Kaggle work runs in GitHub Actions (secrets `KAGGLE_USERNAME`, `KAGGLE_KEY`) or on Kaggle itself; nothing needs a local Kaggle download.
- `Push Notebook to Kaggle` (`push-notebook.yml`): inputs `notebook_dir`, optional `kernel_id`. Converts `<dir>/<name>.py` to `.ipynb` with jupytext if needed, `kaggle kernels push`, polls status up to 320 min. Does **not** fetch the output.
- `Update Kaggle Dataset` (`update-dataset.yml`): inputs `dataset_dir`, `version_notes`, `dry_run`, `create`, `public`. Checks title/subtitle length, runs `<dir>/build.sh` into `<dir>/_build` (stops on any gate), then `kaggle datasets version` (or `create`) and fails on CLI error text. Always `dry_run=true` first.
- `Update Kaggle Metadata` (`update-metadata.yml` + `scripts/update_metadata.py`): sends file/column descriptions, update frequency and sources from `settings.json` without a new version. The owner's notes say column descriptions sent through the API do not show on the dataset page; treat its result as unverified until checked in the browser.
- `Update WBC 2026 Kaggle Dataset`: rebuilds the WBC dataset from the `wbc-scouting` repo.
- All four are `workflow_dispatch` only (403 from cloud) — see Next task 1.
- Local checks in cloud: build scripts and study code can be run on fake data with `python`; `dataset4_statcast_bat_tracking/build.sh` pins `pandas==3.0.6`, `pyarrow==25.0.1`; dataset builds pin `savant-extras==0.6.0`.

## Data access (cloud network)
`baseballsavant.mlb.com`, `statsapi.mlb.com`, `pypi.org` / `files.pythonhosted.org`, `github.com`, `www.kaggle.com`. A Kaggle credential for `www.kaggle.com` is configured in the cloud environment: authenticated `GET https://www.kaggle.com/api/v1/...` with curl works, but the **`kaggle` CLI does not work in cloud** (measured 2026-11-01, see "Cloud reachability"). So cloud can read Kaggle through `curl https://www.kaggle.com/api/v1/...`; pushing kernels / dataset versions stays on GitHub Actions (Next task 1 still needed). Do not commit raw pitch data; datasets are rebuilt by their `build.sh` in Actions.

## Not in the repo yet (owner to decide)
- Matchup harness used up to 2026-10-18: `matchup_d3b.py` (md5 `e7e41d91`), the part files it was concatenated from (`matchup_core.py`, `_part2*.py`, `_part3*.py`, `_part4.py`, `_part5.py`, `_part6b.py`, `d3b_head.txt`), fake-data generator `mk_fake2.py`, planted-defect drivers `mut_d1/d2/d3/d3b.py`, `fetch_kernel.sh`, and the kernel logs of D0-D3. They lived on the Raspberry Pi (`~/claude-scratch/matchup/`) and a local scratchpad; as of 2026-11-01 neither location exists, so they must be rebuilt (the committed `matchup_d*.py` files are the concatenated kernels). The D0-D3 outputs may still be downloadable from the private Kaggle kernels `yasunorim/pitch-shape-x-swing-shape-development-*`.

## Credits (keep in README)
Data from Baseball Savant and the MLB Stats API (MLBAM); built with pybaseball and savant-extras.
