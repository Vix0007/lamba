#!/bin/bash
# after the v6 gate: base greedy 4000-tok generations on the 586 clean eval prompts (DS 65 protocol). log ~/base_eval586.log ; output eval_results/loop_base_ev586.jsonl (resumable)
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0

~/venv-vllm/bin/python ~/antidoom_data/loop_test_g.py --model /home/ubuntu/models/soup-linear-l1.0 --label base --tag ev586 --prompts ~/antidoom_data/evalpool_clean.jsonl --max-tokens 4000 --chunk 128 --mtp >> ~/base_eval586.log 2>&1
echo "BASE EVAL586 DONE" >> ~/base_eval586.log
