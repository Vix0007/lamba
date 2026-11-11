#!/bin/bash
# capability check: GPQA-D b4000 for letterfix2-c30 + repetition_penalty=1.1 (final adapter). log ~/letterfix2_c30_gpqa.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
E=~/eval_results; VPY=~/venv-vllm/bin/python
$VPY ~/ref_eval_scripts/gpqa_ab_g.py --model /home/ubuntu/models/lmh_letterfix2_c30 --label letterfix2c30 --budget 4000 --lora ~/adapters_letterfix2_c30 --repetition-penalty 1.1 --compare-to $E/baseb4000_gpqa_full198_samples.jsonl >> ~/letterfix2_c30_gpqa.log 2>&1
echo "C30GPQA DONE" >> ~/letterfix2_c30_gpqa.log
