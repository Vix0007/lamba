#!/bin/bash
# MalayMMLU full 23,903-item letter-bias regression check for the effreason pilot adapter.
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; VPY=~/venv-vllm/bin/python
$VPY $A/eval_letter_adapter.py --model /home/ubuntu/models/lmh_effreason_final --lora ~/adapters_effreason_final --tag effreasonfinal --jobs A >> ~/effreason_malaymmlu.log 2>&1
echo "EFFREASON_MALAYMMLU_DONE" >> ~/effreason_malaymmlu.log
