#!/bin/bash
# 5k shift-consistency check (job B only) for the effreason pilot adapter -- faster than the
# full 23,958-item job A, per user's call given time already lost today.
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; VPY=~/venv-vllm/bin/python
$VPY $A/eval_letter_adapter.py --model /home/ubuntu/models/lmh_effreason_final --lora ~/adapters_effreason_final --tag effreasonfinal --jobs B >> ~/effreason_shift5k.log 2>&1
echo "EFFREASON_SHIFT5K_DONE" >> ~/effreason_shift5k.log
