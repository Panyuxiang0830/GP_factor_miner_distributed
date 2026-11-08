#!/bin/bash
# Submits the 10,000-expression experiments, from the repository root:
#   bash scripts/vanda/submit_10k.sh
set -e
for j in x1_10k_a x1_10k_b strong_10k omp_10k h0_10k multi2_10k multi4_10k; do qsub scripts/vanda/$j.pbs; done
