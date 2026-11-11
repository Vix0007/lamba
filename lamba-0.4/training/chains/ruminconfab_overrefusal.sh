#!/bin/bash
set -e
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data
cat $A/overrefusal_stratumA.jsonl $A/overrefusal_stratumB.jsonl > $A/overrefusal_all.jsonl

echo "=== OVERREFUSAL: frozen c10 ===" >> ~/ruminconfab_overrefusal.log
~/venv-vllm/bin/python $A/loop_test_g.py --model ~/models/lmh_letterfix2_c10 \
  --lora ~/adapters_letterfix2_c10 --label c10 --tag overrefusal \
  --prompts $A/overrefusal_all.jsonl \
  --max-tokens 800 --chunk 40 --repetition-penalty 1.1 \
  >> ~/ruminconfab_overrefusal.log 2>&1

echo "=== OVERREFUSAL: ruminconfab s1337 ===" >> ~/ruminconfab_overrefusal.log
~/venv-vllm/bin/python $A/loop_test_g.py --model ~/models/lmh_ruminconfab_s1337 \
  --lora ~/adapters_ruminconfab_s1337 --label ruminconfabS1337 --tag overrefusal \
  --prompts $A/overrefusal_all.jsonl \
  --max-tokens 800 --chunk 40 --repetition-penalty 1.1 \
  >> ~/ruminconfab_overrefusal.log 2>&1

echo "OVERREFUSAL_GEN_DONE" >> ~/ruminconfab_overrefusal.log
