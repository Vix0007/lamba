#!/bin/bash
set -e
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data
~/venv-vllm/bin/python $A/loop_test_g.py --model ~/models/lmh_ruminconfab_s1337 \
  --lora ~/adapters_ruminconfab_s1337 --label ruminconfabS1337 --tag holdout385 \
  --prompts $A/concise_sft/holdout_confirm_prompts.jsonl \
  --max-tokens 4000 --chunk 96 --repetition-penalty 1.1 \
  >> ~/ruminconfab_loop_eval.log 2>&1

~/antidoom/.venv/bin/python $A/tail_stats_cmp.py \
  ~/eval_results/loop_letterfix2c10_holdout385rp11.jsonl \
  ~/eval_results/loop_ruminconfabS1337_holdout385.jsonl \
  $A/concise_sft/holdout_confirm_prompts.jsonl \
  | tee ~/eval_results/ruminconfabS1337_vs_c10_holdout385.txt

echo "RUMINCONFAB_LOOP_EVAL_DONE" >> ~/ruminconfab_loop_eval.log
