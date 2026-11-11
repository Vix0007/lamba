#!/bin/bash
# capability check: GPQA-D b4000 for the effreason pilot adapter. log ~/effreason_gpqa.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
E=~/eval_results; VPY=~/venv-vllm/bin/python
$VPY ~/ref_eval_scripts/gpqa_ab_g.py --model /home/ubuntu/models/lmh_effreason_final --label effreasonfinal --budget 4000 --lora ~/adapters_effreason_final --repetition-penalty 1.1 --compare-to $E/baseb4000_gpqa_full198_samples.jsonl >> ~/effreason_gpqa.log 2>&1
echo "EFFREASON_GPQA DONE" >> ~/effreason_gpqa.log
