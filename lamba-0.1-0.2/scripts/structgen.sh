#!/bin/bash
# structural-loop pilot: v9c LoRA (malay-final) rollouts on synthetic long-reasoning-trap prompts (resumable, re-run as more DS batches land). log ~/structgen.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
~/venv-vllm/bin/python ~/antidoom_data/loop_test_g.py --model /home/ubuntu/models/lmh_malay_final --label malayfinal --tag structpilot --prompts ~/antidoom_data/structsyn_pilot_prompts.jsonl --max-tokens 4000 --chunk 70 --lora ~/adapters_malay_final >> ~/structgen.log 2>&1
echo "STRUCTGEN_PASS DONE $(wc -l < ~/antidoom_data/structsyn_pilot_prompts.jsonl)" >> ~/structgen.log
