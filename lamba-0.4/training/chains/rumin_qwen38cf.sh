#!/usr/bin/env bash
# stock Qwen3.8-27B via CF proxy on the CURRENT (safe-import) sandbox, same families as 1810p2
exec >> ~/rumin_qwen38cf.log 2>&1
export EVAL_API=http://127.0.0.1:8011 PATH=/home/ubuntu/venv-vllm/bin:$PATH
cd /home/ubuntu/antidoom_data
python3 -u gen_a3_08_family_conc.py qwen38cf /home/ubuntu/eval_results/a3_08_family_qwen38cf_sbB.jsonl 10 8; echo "RUMINQ_A3_DONE $(date -u +%H:%M)"
python3 -u gen_matched_iterative.py qwen38cf /home/ubuntu/eval_results/matched_iterative_qwen38cf_sbB.jsonl; echo "RUMINQ_MATCHED_DONE $(date -u +%H:%M)"
