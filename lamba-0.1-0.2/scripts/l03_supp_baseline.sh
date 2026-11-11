#!/usr/bin/env bash
# supplemental 0.2 baseline for the two new identity conditions (heldout split only); resumable, existing rows are skipped by key
set -u; cd /home/ubuntu/antidoom_data
while systemctl --user is-active --quiet lamba-l03after2; do sleep 30; done
python3 -u l03_identity_eval.py dpov2 /home/ubuntu/eval_results/l03_identity_dpov2.jsonl --split heldout
echo "SUPP_BASELINE_DONE $(date -u +%H:%M)"
