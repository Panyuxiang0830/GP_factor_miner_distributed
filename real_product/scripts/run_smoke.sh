#!/usr/bin/env bash
# Wiring check on synthetic panels: depths 1-5, tiny population, screening.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/scripts/env.sh"
WORK="${1:-$HERE/smoke_run}"
python3 -m gpp.synthetic --out "$WORK/evolution" --dates 300 --stocks 200 --seed 1 --period evolution
python3 -m gpp.synthetic --out "$WORK/screening" --dates 120 --stocks 200 --seed 2 --period screening
python3 -m gpp.cli --evolution-panels "$WORK/evolution" --screening-panels "$WORK/screening" \
  --output-dir "$WORK/out" --smoke --threads "${OMP_NUM_THREADS}" "${@:2}"
