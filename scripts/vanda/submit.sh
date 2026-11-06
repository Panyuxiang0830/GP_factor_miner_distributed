#!/bin/bash
# Submits all experiment jobs (after build_test.pbs has passed), from the
# repository root:  bash scripts/vanda/submit.sh
set -e
for j in x1_serial scaling_1node scaling_2node scaling_4node; do qsub scripts/vanda/$j.pbs; done
