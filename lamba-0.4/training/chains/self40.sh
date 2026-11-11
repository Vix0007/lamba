#!/usr/bin/env bash
# precise self-attribution: 5 Self items x 40 reps, no system prompt. p3 (when served) then v40t2 (after p3 evalq). log ~/self40.log
exec >> ~/self40.log 2>&1
cd ~/antidoom_data; ts(){ TZ=Asia/Singapore date +%T; }
until grep -q "T11P3_SERVED" ~/v40t12.log; do sleep 30; done
CONSC_ONLY=self_ ~/antidoom/.venv/bin/python -u consc_server.py lamba04v40t11p3 v40t11p3_self40 --reps 40 --conds none
~/antidoom/.venv/bin/python consc_server.py --summary ~/eval_results/consc/v40t11p3_self40_none.json | grep unparsed; echo "SELF40_P3_DONE $(ts)"
