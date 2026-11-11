#!/bin/bash
# DS 84: resolve the GPQA-D b2000 dip: budget 4000, greedy, base (MTP, graph) then v9c (LoRA) with paired comparison. log ~/gpqa_b4000.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
E=~/eval_results; VPY=~/venv-vllm/bin/python
until grep -q "V9C ROBUST DONE" ~/v9c_robust.log 2>/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) START base_b4000" >> ~/gpqa_b4000.log
$VPY ~/ref_eval_scripts/gpqa_ab_g.py --model /home/ubuntu/models/soup-linear-l1.0 --label baseb4000 --budget 4000 --mtp >> ~/gpqa_b4000.log 2>&1
echo "$(date -u +%FT%TZ) START v9c_b4000" >> ~/gpqa_b4000.log
$VPY ~/ref_eval_scripts/gpqa_ab_g.py --model /home/ubuntu/models/lmh_v9c_final --label v9cfinalb4000 --budget 4000 --lora ~/adapters_v9c_final --compare-to $E/baseb4000_gpqa_full198_samples.jsonl >> ~/gpqa_b4000.log 2>&1
echo "$(date -u +%FT%TZ) GPQA4000 DONE" >> ~/gpqa_b4000.log
