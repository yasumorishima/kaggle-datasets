#!/usr/bin/env bash
# Called by .github/workflows/update-dataset.yml before `kaggle datasets version`.
# Builds the upload folder $DIR/_build (3 season Parquet files, 3 CSVs and dataset-metadata.json)
# and fails without writing anything if build.py's gates fail.
# Nothing here needs the Kaggle credentials (the previous version's column list is
# dataset4_columns.md in this folder), so they are removed from the environment.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
NOCREDS=(env -u KAGGLE_USERNAME -u KAGGLE_KEY)

"${NOCREDS[@]}" pip install -q "pandas==3.0.6" "pyarrow==25.0.1" requests

rm -rf "$DIR/_build"
"${NOCREDS[@]}" python "$DIR/build.py" --meta "$DIR/dataset-metadata.json" --settings "$DIR/settings.json" \
  --prev-columns "$DIR/dataset4_columns.md" --out "$DIR/_build"
