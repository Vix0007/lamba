#!/usr/bin/env bash
# Waits for the ruminconfab rollout generator to finish, then builds and
# queues the DS blind-labelling batches automatically. No Claude turns
# needed for either stage -- ds-worker answers batches autonomously.
set -uo pipefail

echo "[chain] waiting for lamba-ruminconfab to finish..."
while systemctl --user is-active --quiet lamba-ruminconfab; do
  sleep 30
done

echo "[chain] generator service ended, checking output..."
N=$(wc -l < ~/eval_results/ruminconfab_calib120_rollouts.jsonl 2>/dev/null || echo 0)
echo "[chain] $N rollouts recorded"

if [ "$N" -lt 100 ]; then
  echo "[chain] WARNING: only $N/120 rollouts -- generator may have crashed or been killed. Building labelling batches anyway from what exists; check the service log before trusting the calibration decision."
fi

echo "[chain] building DS labelling batches..."
python3 ~/antidoom_data/build_ruminconfab_labelling.py

echo "[chain] RUMINCONFAB_LABELLING_QUEUED"
