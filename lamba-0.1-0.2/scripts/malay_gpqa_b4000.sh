#!/bin/bash
# seed-2 GPQA-D b4000 gate (paired vs base b4000). log ~/malay_gpqa.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
E=~/eval_results; VPY=~/venv-vllm/bin/python
$VPY ~/ref_eval_scripts/gpqa_ab_g.py --model /home/ubuntu/models/lmh_malay_final --label malayfinalb4000 --budget 4000 --lora ~/adapters_malay_final --compare-to $E/baseb4000_gpqa_full198_samples.jsonl >> ~/malay_gpqa.log 2>&1
echo "$(date -u +%FT%TZ) MALAYGPQA DONE" >> ~/malay_gpqa.log
