#!/usr/bin/env bash
exec >> ~/rumin1810p2.log 2>&1
export PATH=/home/ubuntu/venv-vllm/bin:$PATH
cd /home/ubuntu/antidoom_data
curl -s -m 5 127.0.0.1:8001/v1/models | grep -q l1810p2 || { echo "RUMIN_NO_SERVER"; exit 1; }
python3 -u gen_a3_08_family_conc.py l1810p2 /home/ubuntu/eval_results/a3_08_family_l1810p2.jsonl 10 8; echo "RUMIN_A3_DONE $(date -u +%H:%M)"
python3 -u gen_matched_iterative.py l1810p2 /home/ubuntu/eval_results/matched_iterative_l1810p2.jsonl; echo "RUMIN_MATCHED_DONE $(date -u +%H:%M)"
