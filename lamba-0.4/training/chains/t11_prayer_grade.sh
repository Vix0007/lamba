#!/usr/bin/env bash
exec >> ~/t11_prayer.log 2>&1
cd ~/antidoom_data
until grep -q "WARMTH_GEN_DONE" ~/t11_prayho.log 2>/dev/null; do sleep 30; done
python3 prayer_judge2.py build t11 t11=/home/ubuntu/eval_results/v40t11/prayer_heldout.jsonl --cond none
N=$(ls ~/ds_queue/pj2_t11_*.txt 2>/dev/null | wc -l); echo "batches $N"
for i in $(seq 1 120); do [ "$(ls ~/ds_queue/out/pj2_t11_*.md 2>/dev/null | wc -l)" -ge "$N" ] && break; sleep 30; done
python3 prayer_judge2.py score t11; echo T11_PRAYER_SCORED
