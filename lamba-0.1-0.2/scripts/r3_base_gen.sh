#!/bin/bash
# base greedy 4000 tok on round3_prompts (paired comparison vs v9c r3gen). log ~/r3_base_gen.log ; output eval_results/loop_base_r3gen.jsonl
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
~/venv-vllm/bin/python ~/antidoom_data/loop_test_g.py --model /home/ubuntu/models/soup-linear-l1.0 --label base --tag r3gen --prompts ~/antidoom_data/round3_prompts.jsonl --max-tokens 4000 --chunk 128 --mtp >> ~/r3_base_gen.log 2>&1
echo "R3BASE DONE" >> ~/r3_base_gen.log
