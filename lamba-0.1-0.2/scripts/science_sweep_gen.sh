#!/bin/bash
# science domain sweep: base then malay-final on 300 held-out science prompts (parallels the Malay generalisation check). log ~/science_sweep.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; VPY=~/venv-vllm/bin/python
$VPY $A/loop_test_g.py --model /home/ubuntu/models/soup-linear-l1.0 --label base --tag scisweep --prompts $A/science_eval_prep_prompts.jsonl --max-tokens 4000 --chunk 100 --mtp >> ~/science_sweep.log 2>&1
$VPY $A/loop_test_g.py --model /home/ubuntu/models/lmh_malay_final --label malayfinal --tag scisweep --prompts $A/science_eval_prep_prompts.jsonl --max-tokens 4000 --chunk 100 --lora ~/adapters_malay_final >> ~/science_sweep.log 2>&1
echo "SCISWEEP DONE" >> ~/science_sweep.log
