#!/bin/bash
# after malaymine (1000): (a) base + v9c on 220 extra held-out Malay eval prompts (eval400 = eval180 + these; never mined), (b) v9c on ~2,800 more Malay mining prompts. log ~/malay_chain2.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; VPY=~/venv-vllm/bin/python
until grep -q "MALAYMINE DONE" ~/malay_mine_gen.log 2>/dev/null; do sleep 30; done
$VPY $A/loop_test_g.py --model /home/ubuntu/models/soup-linear-l1.0 --label base --tag malayex --prompts $A/malay_eval_extra220.jsonl --max-tokens 4000 --chunk 110 --mtp >> ~/malay_chain2.log 2>&1
$VPY $A/loop_test_g.py --model /home/ubuntu/models/lmh_v9c_final --label v9cfinal --tag malayex --prompts $A/malay_eval_extra220.jsonl --max-tokens 4000 --chunk 110 --lora ~/adapters_v9c_final >> ~/malay_chain2.log 2>&1
echo "MALAYEX DONE" >> ~/malay_chain2.log
$VPY $A/loop_test_g.py --model /home/ubuntu/models/lmh_v9c_final --label v9cfinal --tag malaymine2 --prompts $A/malay_mine_prompts2.jsonl --max-tokens 4000 --chunk 125 --lora ~/adapters_v9c_final >> ~/malay_chain2.log 2>&1
echo "MALAYMINE2 DONE" >> ~/malay_chain2.log
