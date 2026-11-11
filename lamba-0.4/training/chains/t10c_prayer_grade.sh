#!/usr/bin/env bash
# grade v40t10 checkpoint-100 held-out prayer as soon as evalq v40t10c finishes prayer_ho. log ~/t10c_prayer.log
exec >> ~/t10c_prayer.log 2>&1
cd ~/antidoom_data
until grep -q "T10C_PRAYHO_DONE" ~/t10c_prayho.log 2>/dev/null; do sleep 30; done
python3 prayer_judge2.py build t10c t10c=/home/ubuntu/eval_results/v40t10c/prayer_heldout.jsonl --cond none
N=$(ls ~/ds_queue/pj2_t10c_*.txt 2>/dev/null | wc -l); echo "batches $N"
for i in $(seq 1 120); do [ "$(ls ~/ds_queue/out/pj2_t10c_*.md 2>/dev/null | wc -l)" -ge "$N" ] && break; sleep 30; done
python3 prayer_judge2.py score t10c; echo T10C_PRAYER_SCORED
