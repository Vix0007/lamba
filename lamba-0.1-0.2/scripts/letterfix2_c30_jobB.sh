#!/bin/bash
# shift-consistency test (job B) on the selected letterfix2 checkpoint-30. log ~/letterfix2_c30_jobB.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data
~/venv-vllm/bin/python $A/eval_letter_adapter.py --model /home/ubuntu/models/lmh_letterfix2_c30 --lora ~/adapters_letterfix2_c30 --tag letterfix2c30 --jobs B >> ~/letterfix2_c30_jobB.log 2>&1
echo "C30_JOBB_DONE" >> ~/letterfix2_c30_jobB.log
