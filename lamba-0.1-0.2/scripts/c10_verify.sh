#!/bin/bash
until grep -q "LETTERFIX2_MORE_DONE" ~/letterfix2_more.log 2>/dev/null; do sleep 20; done
# verify checkpoint-10: shift-consistency (job B) + loop regression check (150-test, rp=1.1). log ~/c10_verify.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; VPY=~/venv-vllm/bin/python
$VPY $A/eval_letter_adapter.py --model /home/ubuntu/models/lmh_letterfix2_c10 --lora ~/adapters_letterfix2_c10 --tag letterfix2c10 --jobs B >> ~/c10_verify.log 2>&1
$VPY $A/loop_test_g.py --model /home/ubuntu/models/lmh_letterfix2_c10 --label letterfix2c10 --tag g150 --prompts $A/loop_testset150.jsonl --max-tokens 4000 --chunk 75 --lora ~/adapters_letterfix2_c10 --repetition-penalty 1.1 >> ~/c10_verify.log 2>&1
echo "C10_VERIFY_DONE" >> ~/c10_verify.log
