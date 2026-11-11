#!/bin/bash
until grep -q "SCISWEEP DONE" ~/science_sweep.log 2>/dev/null; do sleep 20; done
# structural-loop calibration round 2: v9c/malay-final rollouts on redesigned final-only-answer prompts (no restate confound). log ~/structgen2.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
~/venv-vllm/bin/python ~/antidoom_data/loop_test_g.py --model /home/ubuntu/models/lmh_malay_final --label malayfinal --tag structcalib --prompts ~/antidoom_data/structsyn2_calib_prompts.jsonl --max-tokens 4000 --chunk 107 --lora ~/adapters_malay_final >> ~/structgen2.log 2>&1
echo "STRUCTGEN2 DONE" >> ~/structgen2.log
