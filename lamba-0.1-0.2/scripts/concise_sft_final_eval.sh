#!/bin/bash
# Beat-ThinkingCap track: FINAL, one-shot 150-test + GPQA-D b4000 + full MalayMMLU job A.
# Per DS's protocol: run once, no tuning after seeing this.
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python
LMH=/home/ubuntu/models/lmh_concise_sft; AD=~/adapters_concise_sft
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/concise_sft_final_eval.log; }

log "START final 150-test + GPQA-D + MalayMMLU (one engine)"
$VPY $A/eval_combined.py --model $LMH --lora $AD --tag concisesft --jobs gpqa,A,g150 --budget 4000 >> ~/concise_sft_final_eval.log 2>&1
log "END final eval"

~/antidoom/.venv/bin/python $A/tail_stats_by_domain.py $E/loop_letterfix2c10_g150.jsonl $E/loop_concisesft_g150.jsonl $A/loop_testset150.jsonl > $E/concisesft_vs_letterfix2c10_g150_bydomain.txt 2>&1
cat $E/concisesft_vs_letterfix2c10_g150_bydomain.txt >> ~/concise_sft_final_eval.log
log "CONCISE_SFT_FINAL_EVAL DONE"
