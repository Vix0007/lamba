#!/usr/bin/env bash
exec >> ~/t11p_prayer.log 2>&1
cd ~/antidoom_data
until grep -q "WARMTH_GEN_DONE" ~/t11p_prayho.log 2>/dev/null; do sleep 30; done
python3 prayer_judge2.py build t11p t11p=/home/ubuntu/eval_results/v40t11p/prayer_heldout.jsonl --cond none
N=$(ls ~/ds_queue/pj2_t11p_*.txt 2>/dev/null | wc -l); echo "batches $N"
for i in $(seq 1 120); do [ "$(ls ~/ds_queue/out/pj2_t11p_*.md 2>/dev/null | wc -l)" -ge "$N" ] && break; sleep 30; done
python3 prayer_judge2.py score t11p; echo T11_PRAYER_SCORED
