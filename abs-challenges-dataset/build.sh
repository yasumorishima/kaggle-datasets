#!/usr/bin/env bash
# Called by .github/workflows/update-dataset.yml before `kaggle datasets create/version`.
# Builds the upload folder $DIR/_build (the CSVs, file_descriptions.txt and dataset-metadata.json)
# and fails without writing anything if build.py's gates fail. Nothing here needs Kaggle
# credentials, so they are removed from the environment.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
NOCREDS=(env -u KAGGLE_USERNAME -u KAGGLE_KEY)

"${NOCREDS[@]}" pip install -q "savant-extras==0.6.0" pandas numpy

rm -rf "$DIR/_build"
"${NOCREDS[@]}" python "$DIR/build.py" --meta "$DIR/dataset-metadata.json" --out "$DIR/_build"
