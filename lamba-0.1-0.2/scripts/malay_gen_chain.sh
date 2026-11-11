#!/bin/bash
# generalisation eval (DS 87): 180 unseen Malay MCQ prompts, greedy 4000 tok, base(MTP) vs v9c(LoRA). log ~/malay_gen.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; VPY=~/venv-vllm/bin/python
$VPY $A/loop_test_g.py --model /home/ubuntu/models/soup-linear-l1.0 --label base --tag malaygen --prompts $A/malay_gen_eval180.jsonl --max-tokens 4000 --chunk 90 --mtp >> ~/malay_gen.log 2>&1
$VPY $A/loop_test_g.py --model /home/ubuntu/models/lmh_v9c_final --label v9cfinal --tag malaygen --prompts $A/malay_gen_eval180.jsonl --max-tokens 4000 --chunk 90 --lora ~/adapters_v9c_final >> ~/malay_gen.log 2>&1
echo "MALAYGEN DONE" >> ~/malay_gen.log
