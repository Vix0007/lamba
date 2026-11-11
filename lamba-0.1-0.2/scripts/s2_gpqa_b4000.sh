#!/bin/bash
# seed-2 GPQA-D b4000 gate (paired vs base b4000). log ~/s2_gpqa.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
E=~/eval_results; VPY=~/venv-vllm/bin/python
$VPY ~/ref_eval_scripts/gpqa_ab_g.py --model /home/ubuntu/models/lmh_v9c_s2_final --label v9cs2finalb4000 --budget 4000 --lora ~/adapters_v9c_s2_final --compare-to $E/baseb4000_gpqa_full198_samples.jsonl >> ~/s2_gpqa.log 2>&1
echo "$(date -u +%FT%TZ) S2GPQA DONE" >> ~/s2_gpqa.log
