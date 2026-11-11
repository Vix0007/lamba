#!/usr/bin/env bash
# Round-2 chain: waits for lamba-ruminconfab-r2 (540-prompt generation) to
# finish, then builds and queues blind DS-labelling batches for it. Separate
# key/prefix from round 1 so they don't collide.
set -uo pipefail

echo "[chain-r2] waiting for lamba-ruminconfab-r2 to finish..."
while systemctl --user is-active --quiet lamba-ruminconfab-r2; do
  sleep 30
done

echo "[chain-r2] generator service ended, checking output..."
N=$(wc -l < ~/eval_results/ruminconfab_round2_rollouts.jsonl 2>/dev/null || echo 0)
echo "[chain-r2] $N rollouts recorded (expected 540)"

if [ "$N" -lt 450 ]; then
  echo "[chain-r2] WARNING: only $N/540 rollouts -- generator may have crashed or been killed. Building labelling batches anyway from what exists; check the service log before trusting the decision."
fi

echo "[chain-r2] building DS labelling batches..."
RC_ROLLOUTS=~/eval_results/ruminconfab_round2_rollouts.jsonl \
RC_KEY=~/antidoom_data/ruminconfab_round2_blind_key.json \
RC_PREFIX=101_ruminconfab_r2_label \
python3 ~/antidoom_data/build_ruminconfab_labelling.py

echo "[chain-r2] RUMINCONFAB_R2_LABELLING_QUEUED"
