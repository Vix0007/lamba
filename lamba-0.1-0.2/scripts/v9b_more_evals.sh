#!/bin/bash
# v9b: 150 test (once) -> GPQA-D (capability regression vs base 145/198) -> MalayMMLU letter eval. log ~/v9b_more.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python; LMH=/home/ubuntu/models/lmh_v9b_final; AD=~/adapters_v9b_final
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/v9b_more.log; }
log "START g150"; $VPY $A/loop_test_g.py --model $LMH --label v9bfinal --tag g150 --prompts $A/loop_testset150.jsonl --max-tokens 4000 --chunk 75 --lora $AD >> ~/v9b_more.log 2>&1; log "END g150"
log "START gpqa"; $VPY ~/ref_eval_scripts/gpqa_ab_g.py --model $LMH --label v9bfinal --lora $AD --compare-to $E/ab_base_lamba_gpqa_full198_samples.jsonl >> ~/v9b_more.log 2>&1; log "END gpqa"
log "START letter"; $VPY $A/eval_letter_adapter.py --tag v9bfinal --model $LMH --lora $AD >> ~/v9b_more.log 2>&1; log "END letter"
log "V9B MORE DONE"
