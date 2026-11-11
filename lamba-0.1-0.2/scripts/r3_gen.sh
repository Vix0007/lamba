#!/bin/bash
# round 3 generation: v9c (LoRA) greedy 4000 tok on ~700 disjoint mcp+science prompts, after the v9c robustness bundle. log ~/r3_gen.log ; output eval_results/loop_v9cfinal_r3gen.jsonl
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
until grep -q "GPQA4000 DONE" ~/gpqa_b4000.log 2>/dev/null; do sleep 30; done
~/venv-vllm/bin/python ~/antidoom_data/loop_test_g.py --model /home/ubuntu/models/lmh_v9c_final --label v9cfinal --tag r3gen --prompts ~/antidoom_data/round3_prompts.jsonl --max-tokens 4000 --chunk 128 --lora ~/adapters_v9c_final >> ~/r3_gen.log 2>&1
echo "R3GEN DONE" >> ~/r3_gen.log
