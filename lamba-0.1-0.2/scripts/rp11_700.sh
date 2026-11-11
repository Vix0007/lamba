#!/bin/bash
# largest-n confirmation: malay-final + repetition_penalty=1.1 on the 700 fresh mcp/science prompts. log ~/rp11_700.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
~/venv-vllm/bin/python ~/antidoom_data/loop_test_g.py --model /home/ubuntu/models/lmh_malay_final --label malayfinal --tag rp11r3gen --prompts ~/antidoom_data/round3_prompts.jsonl --max-tokens 4000 --chunk 128 --lora ~/adapters_malay_final --repetition-penalty 1.1 >> ~/rp11_700.log 2>&1
echo "RP11_700 DONE" >> ~/rp11_700.log
