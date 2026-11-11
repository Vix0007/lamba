#!/bin/bash
set -e
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data

echo "=== B3 CHECK: frozen c10 ===" >> ~/ruminconfab_b3check.log
~/venv-vllm/bin/python $A/loop_test_g.py --model ~/models/lmh_letterfix2_c10 \
  --lora ~/adapters_letterfix2_c10 --label c10 --tag b3aug \
  --prompts $A/b3_augment_prompts.jsonl \
  --max-tokens 1200 --chunk 20 --repetition-penalty 1.1 \
  >> ~/ruminconfab_b3check.log 2>&1

echo "=== B3 CHECK: ruminconfab s1337 ===" >> ~/ruminconfab_b3check.log
~/venv-vllm/bin/python $A/loop_test_g.py --model ~/models/lmh_ruminconfab_s1337 \
  --lora ~/adapters_ruminconfab_s1337 --label ruminconfabS1337 --tag b3aug \
  --prompts $A/b3_augment_prompts.jsonl \
  --max-tokens 1200 --chunk 20 --repetition-penalty 1.1 \
  >> ~/ruminconfab_b3check.log 2>&1

echo "B3_CHECK_GEN_DONE" >> ~/ruminconfab_b3check.log
