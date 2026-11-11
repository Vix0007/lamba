#!/bin/bash
# Malay rollouts with v9c (LoRA) on 1000 unseen Malay prompts (candidate mining pool for a Malay-targeted round / larger Malay eval). log ~/malay_mine_gen.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
~/venv-vllm/bin/python ~/antidoom_data/loop_test_g.py --model /home/ubuntu/models/lmh_v9c_final --label v9cfinal --tag malaymine --prompts ~/antidoom_data/malay_mine_prompts1000.jsonl --max-tokens 4000 --chunk 125 --lora ~/adapters_v9c_final >> ~/malay_mine_gen.log 2>&1
echo "MALAYMINE DONE" >> ~/malay_mine_gen.log
