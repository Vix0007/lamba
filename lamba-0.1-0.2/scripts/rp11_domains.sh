#!/bin/bash
# validate repetition_penalty=1.1 doesn't hurt the Malay/science domains where malay-final already performs well at rp=1.0. log ~/rp11_domains.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; VPY=~/venv-vllm/bin/python
$VPY $A/loop_test_g.py --model /home/ubuntu/models/lmh_malay_final --label malayfinal --tag rp11malay400 --prompts $A/malay_eval400_prompts.jsonl --max-tokens 4000 --chunk 100 --lora ~/adapters_malay_final --repetition-penalty 1.1 >> ~/rp11_domains.log 2>&1
$VPY $A/loop_test_g.py --model /home/ubuntu/models/lmh_malay_final --label malayfinal --tag rp11sci --prompts $A/science_eval_prep_prompts.jsonl --max-tokens 4000 --chunk 100 --lora ~/adapters_malay_final --repetition-penalty 1.1 >> ~/rp11_domains.log 2>&1
echo "RP11DOMAINS DONE" >> ~/rp11_domains.log
