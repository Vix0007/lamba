#!/usr/bin/env bash
until grep -q MINE_A_DONE ~/v40t7.log; do sleep 30; done
cd ~/antidoom_data; ~/venv-vllm/bin/python -u warmth_gen.py lamba04v40t2 v40t7_faith_prompts.json ~/eval_results/v40t7_mine/faith_none_v40t2.jsonl --n 2 --conds none > ~/v40t7_faithnone.log 2>&1 && echo "MINE_FAITHNONE_DONE $(TZ=Asia/Singapore date +%T)" >> ~/v40t7.log
