#!/usr/bin/env bash
# Export the evolution (2016-2021) and screening (2022) panels once.
#   scripts/export_data.sh EOD_FILE LABEL_FILE OUT_DIR [WARMUP_DAYS]
# WARMUP_DAYS = 0 (default) keeps the course semantics: 2022 windows restart
# on the first 2022 day. A positive value prepends that many 2021 rows that
# are used as history only (meta.json score_begin); see README.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/scripts/env.sh"
EOD="$1"; LABEL="$2"; OUT="$3"; WARMUP="${4:-0}"
python3 -m gpp.export --data-file "$EOD" --label-file "$LABEL" --out "$OUT/evolution" --period evolution
python3 -m gpp.export --data-file "$EOD" --label-file "$LABEL" --out "$OUT/screening" --period screening \
  --warmup-days "$WARMUP"
