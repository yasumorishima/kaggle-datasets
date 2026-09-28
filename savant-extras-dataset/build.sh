#!/usr/bin/env bash
# Called by .github/workflows/update-dataset.yml before `kaggle datasets version`.
# Builds the upload folder $DIR/_build (20 CSVs and dataset-metadata.json)
# and fails without writing anything if build.py's gates fail.
# KAGGLE_USERNAME / KAGGLE_KEY are needed only for the download of the current version, so they
# are removed from the environment of everything else (pip and the build script).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
NOCREDS=(env -u KAGGLE_USERNAME -u KAGGLE_KEY)

"${NOCREDS[@]}" pip install -q "savant-extras==0.6.0" pandas requests

# The current Kaggle version: source of the FanGraphs pitcher_quality table, and the list of
# CSVs the new version must have a successor for.
CARRY="$(mktemp -d)"
kaggle datasets download -d yasunorim/baseball-savant-leaderboards-2024 -p "$CARRY" --unzip
ls -l "$CARRY"

rm -rf "$DIR/_build"
"${NOCREDS[@]}" python "$DIR/build.py" --carry "$CARRY" --meta "$DIR/dataset-metadata.json" --out "$DIR/_build"
