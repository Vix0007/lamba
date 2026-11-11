#!/bin/bash
# malay-final on the same 700 fresh mcp/science prompts already tested on base+v9c (bigger-n confirmation on the current best adapter). log ~/malayfinal_700.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
~/venv-vllm/bin/python ~/antidoom_data/loop_test_g.py --model /home/ubuntu/models/lmh_malay_final --label malayfinal --tag r3gen --prompts ~/antidoom_data/round3_prompts.jsonl --max-tokens 4000 --chunk 128 --lora ~/adapters_malay_final >> ~/malayfinal_700.log 2>&1
echo "MALAYFINAL700 DONE" >> ~/malayfinal_700.log
