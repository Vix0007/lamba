#!/usr/bin/env bash
# usage: l03_baseline_chain.sh <served_model> <tag>   (HELD-OUT split only: that is all we can compare after training)
set -u; cd /home/ubuntu/antidoom_data
M=$1; T=$2
python3 l03_tool_eval.py $M /home/ubuntu/eval_results/l03_tool_$T.jsonl --split heldout
python3 l03_identity_eval.py $M /home/ubuntu/eval_results/l03_identity_$T.jsonl --split heldout
echo "L03_BASELINE_DONE $T"
