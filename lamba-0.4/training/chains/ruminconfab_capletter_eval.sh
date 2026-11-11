#!/bin/bash
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
cd ~/antidoom_data
~/venv-vllm/bin/python eval_combined.py --model ~/models/lmh_ruminconfab_s1337 --lora ~/adapters_ruminconfab_s1337 \
  --tag ruminconfabS1337 --jobs gpqa,A --repetition-penalty 1.1 \
  --compare-to ~/eval_results/baseb4000_gpqa_full198_samples.jsonl \
  >> ~/ruminconfab_capletter.log 2>&1
echo "RUMINCONFAB_CAPLETTER_DONE" >> ~/ruminconfab_capletter.log
