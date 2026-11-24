#!/usr/bin/env bash
# Full run: 100k population, depths 1-5, then 2022 screening.
#   scripts/run_full.sh DATA_DIR OUT_DIR [CONFIG] [extra gpp.cli args...]
# DATA_DIR holds evolution/ and screening/ from scripts/export_data.sh.
# CONFIG defaults to configs/host_512g.json; pick the preset for the host RAM.
# The run resumes from OUT_DIR/eval/run_*/run_state.pkl if it exists.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/scripts/env.sh"
DATA="$1"; OUT="$2"; CONFIG="${3:-$HERE/configs/host_512g.json}"
mkdir -p "$OUT"
python3 -m gpp.cli --evolution-panels "$DATA/evolution" --screening-panels "$DATA/screening" \
  --output-dir "$OUT" --config "$CONFIG" "${@:4}" 2>&1 | tee -a "$OUT/run.log"
