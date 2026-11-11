#!/usr/bin/env bash
# on-policy native-format mining for 0.3 (MINING split only). Order = value: tool (user's real failure) first.
set -u; cd /home/ubuntu/antidoom_data; PY=/home/ubuntu/antidoom/.venv/bin/python
M=$1; T=$2; O=/home/ubuntu/eval_results
$PY l03_mine_native.py $M l03_tool_states_mine_v2.jsonl $O/l03_mined_tool_$T.jsonl --k 8 --temp 1.0
$PY l03_mine_native.py $M l03_identity_states_weak.jsonl $O/l03_mined_idweak_$T.jsonl --k 2 --temp 1.0
$PY l03_mine_native.py $M l03_identity_states_prefill.jsonl $O/l03_mined_idpref_$T.jsonl --k 1 --temp 0.8
$PY l03_mine_native.py $M l03_identity_states_replay.jsonl $O/l03_mined_idreplay_$T.jsonl --k 1 --temp 1.0
echo "L03_MINE_DONE $T"
