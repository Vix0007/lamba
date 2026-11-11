#!/bin/bash
# run the three remaining gate evals for rc5 NOW (the chain would run them after the whole queue): names battery, empty-reply replay, tone battery (+ DS tone grading)
exec >> ~/early_gates_z.log 2>&1
cd ~/antidoom_data; G=/home/ubuntu/eval_results/diag05; D=/home/ubuntu/eval_results/l1810z
echo "EG_START $(TZ=Asia/Singapore date +%H:%M)"
( python3 diag05_battery.py l1810z $G/l1810z.jsonl --n 4 && python3 diag05_score.py l1810p2=$G/l1810p2.jsonl l1810z=$G/l1810z.jsonl > $D/name_battery.txt 2>&1; echo "EG_NAMES_DONE" ) &
( python3 diag05_emptyreplay.py l1810z $G/b2_v40t11p3.jsonl $G/emptyreplay_l1810z.jsonl --n 10; echo "EG_EMPTY_DONE" ) &
( python3 tone_battery.py l1810z /home/ubuntu/eval_results/tone1810/l1810z.jsonl; echo "EG_TONE_GEN_DONE"; python3 tone_grade.py build toneZ z=/home/ubuntu/eval_results/tone1810/l1810z.jsonl y=/home/ubuntu/eval_results/tone1810/l1810y.jsonl > /dev/null 2>&1; echo "EG_TONE_GRADING_QUEUED" ) &
wait; echo "EG_DONE $(TZ=Asia/Singapore date +%H:%M)"
