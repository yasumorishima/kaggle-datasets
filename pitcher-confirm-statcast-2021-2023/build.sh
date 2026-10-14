#!/usr/bin/env bash
# Called by .github/workflows/update-dataset.yml before `kaggle datasets create` / `version`.
# Builds the upload folder $DIR/_build (statcast_2021/2022/2023.parquet, tracking_gaps.csv and
# dataset-metadata.json) and fails without writing anything if build.py's gates fail.
# build.py imports the fetch/parse functions of ../dataset4_statcast_bat_tracking/build.py.
# Nothing here needs the Kaggle credentials, so they are removed from the environment.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
NOCREDS=(env -u KAGGLE_USERNAME -u KAGGLE_KEY)

"${NOCREDS[@]}" pip install -q "pandas==3.0.6" "pyarrow==25.0.1" requests

rm -rf "$DIR/_build"
"${NOCREDS[@]}" python "$DIR/build.py" --meta "$DIR/dataset-metadata.json" --out "$DIR/_build"
