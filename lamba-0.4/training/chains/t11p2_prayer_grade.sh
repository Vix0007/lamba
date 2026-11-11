#!/usr/bin/env bash
exec >> ~/t11p2_prayer.log 2>&1
cd ~/antidoom_data
until grep -q "DONE prayer_ho" ~/evalq_v40t11p2_box1.log 2>/dev/null; do sleep 30; done
python3 prayer_judge2.py build t11p2 t11p2=/home/ubuntu/eval_results/v40t11p2/prayer_heldout.jsonl --cond none
N=$(ls ~/ds_queue/pj2_t11p2_*.txt 2>/dev/null | wc -l); echo "batches $N"
for i in $(seq 1 120); do [ "$(ls ~/ds_queue/out/pj2_t11p2_*.md 2>/dev/null | wc -l)" -ge "$N" ] && break; sleep 30; done
python3 prayer_judge2.py score t11p2; echo T11_PRAYER_SCORED
