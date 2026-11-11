#!/bin/bash
# v8 validation (DS 70): base586 -> GPQA-D (v8) -> letter eval (v8) -> v8 adapter on the 586 clean prompts. Each step resumable / skippable via markers. log ~/v8_validate.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python; APY=~/antidoom/.venv/bin/python; BASE=/home/ubuntu/models/soup-linear-l1.0; LMH=/home/ubuntu/models/lmh_v8_final; AD=~/adapters_v8_final
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/v8_validate.log; }
st(){ n=$1; shift; m=$E/.v8val_$n; [ -e $m ] && { log "SKIP $n"; return 0; }; log "START $n"; "$@" >> ~/v8_validate.log 2>&1 && touch $m; log "END $n rc=$?"; }
st base586 $VPY $A/loop_test_g.py --model $BASE --label base --tag ev586 --prompts $A/evalpool_clean.jsonl --max-tokens 4000 --chunk 128 --mtp
st gpqa $VPY ~/ref_eval_scripts/gpqa_ab_g.py --model $LMH --label v8final --lora $AD --compare-to $E/ab_base_lamba_gpqa_full198_samples.jsonl
st letter $VPY $A/eval_letter_adapter.py --tag v8final --model $LMH --lora $AD
st adapter586 $VPY $A/loop_test_g.py --model $LMH --label v8final --tag ev586 --prompts $A/evalpool_clean.jsonl --max-tokens 4000 --chunk 128 --lora $AD
log "V8 VALIDATE DONE"
