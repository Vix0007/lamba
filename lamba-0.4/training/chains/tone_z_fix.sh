#!/bin/bash
# rebuild the DS tone grading for rc5 against rc1 (l1810t) once the tone generation is done (rc4 tone file does not exist)
until grep -q EG_TONE_GEN_DONE ~/early_gates_z.log; do sleep 20; done
cd ~/antidoom_data; python3 tone_grade.py build toneZ z=/home/ubuntu/eval_results/tone1810/l1810z.jsonl t=/home/ubuntu/eval_results/tone1810/l1810t.jsonl >> ~/early_gates_z.log 2>&1; echo "EG_TONE_REBUILT_VS_RC1" >> ~/early_gates_z.log
