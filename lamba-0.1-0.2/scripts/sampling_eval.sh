#!/bin/bash
# robustness: sampled decoding (T=0.7, top_p 0.95, seed 1234) on 100 held-out prompts: base (MTP) then v9b (LoRA). log ~/sampling_eval.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; VPY=~/venv-vllm/bin/python
echo "$(date -u +%FT%TZ) START base_T07" >> ~/sampling_eval.log
$VPY $A/loop_test_gt.py --model /home/ubuntu/models/soup-linear-l1.0 --label base --tag T07s1 --prompts $A/eval_keep100s_prompts.jsonl --max-tokens 4000 --chunk 100 --mtp --temp 0.7 --seed 1234 >> ~/sampling_eval.log 2>&1
echo "$(date -u +%FT%TZ) START v9b_T07" >> ~/sampling_eval.log
$VPY $A/loop_test_gt.py --model /home/ubuntu/models/lmh_v9b_final --label v9bfinal --tag T07s1 --prompts $A/eval_keep100s_prompts.jsonl --max-tokens 4000 --chunk 100 --lora ~/adapters_v9b_final --temp 0.7 --seed 1234 >> ~/sampling_eval.log 2>&1
echo "$(date -u +%FT%TZ) SAMPLING EVAL DONE" >> ~/sampling_eval.log
