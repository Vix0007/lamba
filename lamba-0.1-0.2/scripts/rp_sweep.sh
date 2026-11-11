#!/bin/bash
until grep -q "MALAYFINAL700 DONE" ~/malayfinal_700.log 2>/dev/null; do sleep 20; done
# decode-time repetition-penalty sweep on malay-final, greedy, exact 150-test protocol (same prompts/max-tokens as the pre-registered primary). log ~/rp_sweep.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; VPY=~/venv-vllm/bin/python
for rp in 1.05 1.1 1.15 1.2; do
  tag="g150rp${rp//./}"
  echo "=== rp=$rp tag=$tag ===" >> ~/rp_sweep.log
  $VPY $A/loop_test_g.py --model /home/ubuntu/models/lmh_malay_final --label malayfinal --tag "$tag" --prompts $A/loop_testset150.jsonl --max-tokens 4000 --chunk 75 --lora ~/adapters_malay_final --repetition-penalty $rp >> ~/rp_sweep.log 2>&1
done
echo "RPSWEEP DONE" >> ~/rp_sweep.log
