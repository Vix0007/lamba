#!/bin/bash
# robustness: sampled decoding (T=0.7) for malay-final on the same 100 prompts as v9c's check (base arm reused). log ~/sampling_eval_malay.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; VPY=~/venv-vllm/bin/python
$VPY $A/loop_test_gt.py --model /home/ubuntu/models/lmh_malay_final --label malayfinal --tag T07s1 --prompts $A/eval_keep100s_prompts.jsonl --max-tokens 4000 --chunk 100 --lora ~/adapters_malay_final --temp 0.7 --seed 1234 >> ~/sampling_eval_malay.log 2>&1
echo "$(date -u +%FT%TZ) SAMPLING EVAL MALAY DONE" >> ~/sampling_eval_malay.log
