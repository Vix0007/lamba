#!/bin/bash
# final capability check: GPQA-D b4000 for letterfix2-c10 + repetition_penalty=1.1 (new best candidate). log ~/letterfix2_c10_gpqa.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
E=~/eval_results; VPY=~/venv-vllm/bin/python
$VPY ~/ref_eval_scripts/gpqa_ab_g.py --model /home/ubuntu/models/lmh_letterfix2_c10 --label letterfix2c10 --budget 4000 --lora ~/adapters_letterfix2_c10 --repetition-penalty 1.1 --compare-to $E/baseb4000_gpqa_full198_samples.jsonl >> ~/letterfix2_c10_gpqa.log 2>&1
echo "C10GPQA DONE" >> ~/letterfix2_c10_gpqa.log
