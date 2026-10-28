#!/usr/bin/env bash
# Download the data from PhysioNet's open-data copy on AWS (no account needed).
#   bash scripts/download_data.sh [cinc|external|all] [folder]      (defaults: all, data)
#   cinc:     PhysioNet/CinC Challenge 2017 training set (8,528 recordings) -> folder/cinc2017
#   external: MIT-BIH Arrhythmia Database (about 100 MB) -> folder/mitdb,
#             MIT-BIH Atrial Fibrillation Database (about 0.6 GB) -> folder/afdb
set -e
WHAT="${1:-all}"
OUT="${2:-data}"
if ! command -v aws >/dev/null 2>&1; then pip install -q awscli; fi
get() { aws s3 sync --no-sign-request --only-show-errors "s3://physionet-open/$1" "$2" "${@:3}"; }

if [[ "$WHAT" == "cinc" || "$WHAT" == "all" ]]; then
  get challenge-2017/1.0.0/ "$OUT/cinc2017" --exclude "*" --include "training/*" --include "REFERENCE*.csv"
  echo "CinC 2017 recordings in $OUT/cinc2017: $(find "$OUT/cinc2017/training" -name '*.hea' | wc -l)"
fi
if [[ "$WHAT" == "external" || "$WHAT" == "all" ]]; then
  get mitdb/1.0.0/ "$OUT/mitdb" || python -c "import sys, wfdb; wfdb.dl_database('mitdb', sys.argv[1])" "$OUT/mitdb"
  get afdb/1.0.0/ "$OUT/afdb" || python -c "import sys, wfdb; wfdb.dl_database('afdb', sys.argv[1])" "$OUT/afdb"
  echo "MIT-BIH records: $(ls "$OUT"/mitdb/*.dat | wc -l) | AF Database records with signals: $(ls "$OUT"/afdb/*.dat | wc -l)"
fi
