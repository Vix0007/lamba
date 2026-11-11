#!/bin/bash
# round 2b: v8 greedy on the REMAINING disjoint mcp+science prompts, after r2gen. output eval_results/loop_v8final_r2gen2.jsonl, log ~/r2_gen2.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0

~/venv-vllm/bin/python ~/antidoom_data/loop_test_g.py --model /home/ubuntu/models/lmh_v8_final --label v8final --tag r2gen2 --prompts ~/antidoom_data/round2b_prompts.jsonl --max-tokens 4000 --chunk 128 --lora ~/adapters_v8_final >> ~/r2_gen2.log 2>&1
echo "R2GEN2 DONE" >> ~/r2_gen2.log
