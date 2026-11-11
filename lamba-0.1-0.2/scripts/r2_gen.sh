#!/bin/bash
# round 2 mining generation: v8 adapter, greedy 4000 tok on 500 fresh disjoint mcp+science prompts (after the 586 adapter run). log ~/r2_gen.log ; output eval_results/loop_v8final_r2gen.jsonl
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
until grep -q "V8 VALIDATE2 DONE" ~/v8_validate.log 2>/dev/null; do sleep 30; done
~/venv-vllm/bin/python ~/antidoom_data/loop_test_g.py --model /home/ubuntu/models/lmh_v8_final --label v8final --tag r2gen --prompts ~/antidoom_data/round2_prompts.jsonl --max-tokens 4000 --chunk 128 --lora ~/adapters_v8_final >> ~/r2_gen.log 2>&1
echo "R2GEN DONE" >> ~/r2_gen.log
