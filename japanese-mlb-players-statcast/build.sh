#!/usr/bin/env bash
# Called by .github/workflows/update-dataset.yml before the new version is uploaded.
# Builds the upload folder $DIR/_build (the four CSVs and
# dataset-metadata.json) and fails without writing anything if build.py's gates fail.
# KAGGLE_USERNAME / KAGGLE_KEY are needed only to download the current version's players.csv
# (to check that no earlier player is dropped), so they are removed from everything else.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
NOCREDS=(env -u KAGGLE_USERNAME -u KAGGLE_KEY)

"${NOCREDS[@]}" pip install -q pandas requests

# Only players.csv of the current version is needed (the pitch files are about 100 MB).
CARRY="$(mktemp -d)"
trap 'rm -rf "$CARRY"' EXIT
kaggle datasets download -d yasunorim/japan-mlb-pitchers-batters-statcast -f players.csv -p "$CARRY"
if [ -f "$CARRY/players.csv.zip" ]; then
  (cd "$CARRY" && unzip -q players.csv.zip && rm players.csv.zip)
fi
ls -l "$CARRY"
test -s "$CARRY/players.csv"

rm -rf "$DIR/_build"
"${NOCREDS[@]}" python "$DIR/build.py" --carry "$CARRY" --meta "$DIR/dataset-metadata.json" --out "$DIR/_build"
