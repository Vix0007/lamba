#!/bin/bash
# loop regression check: letterfix2-c30 (letter fix on top of malay-final) on the 150-test set. log ~/letterfix2_c30_loop.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data
~/venv-vllm/bin/python $A/loop_test_g.py --model /home/ubuntu/models/lmh_letterfix2_c30 --label letterfix2c30 --tag g150 --prompts $A/loop_testset150.jsonl --max-tokens 4000 --chunk 75 --lora ~/adapters_letterfix2_c30 --repetition-penalty 1.1 >> ~/letterfix2_c30_loop.log 2>&1
echo "C30_LOOP_DONE" >> ~/letterfix2_c30_loop.log
