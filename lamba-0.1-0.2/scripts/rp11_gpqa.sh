#!/bin/bash
# capability check: GPQA-D b4000 for malay-final + repetition_penalty=1.1 (the rp sweep winner: 39.6% relative on the primary endpoint). log ~/rp11_gpqa.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
E=~/eval_results; VPY=~/venv-vllm/bin/python
$VPY ~/ref_eval_scripts/gpqa_ab_g.py --model /home/ubuntu/models/lmh_malay_final --label malayfinalrp11 --budget 4000 --lora ~/adapters_malay_final --repetition-penalty 1.1 --compare-to $E/baseb4000_gpqa_full198_samples.jsonl >> ~/rp11_gpqa.log 2>&1
echo "RP11GPQA DONE" >> ~/rp11_gpqa.log
