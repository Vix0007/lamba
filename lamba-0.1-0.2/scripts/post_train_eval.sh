#!/bin/bash
# Post-train evals for Round 2. Run as capped service after prod finishes. Every step skips if its output exists. Log: ~/post_eval.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python; APY=~/antidoom/.venv/bin/python
AD=~/antidoom/runs/lamba2/lora; CONV=~/adapters_r2; LMH=/home/ubuntu/models/soup-linear-l1.0-r2lmh; BASE=/home/ubuntu/models/soup-linear-l1.0
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/post_eval.log; }
step(){ # step <name> <cmd...>  (marker file written only on rc=0; every step resumable/skippable)
  local n=$1 m=$E/.postdone_$1; shift
  if [ -e "$m" ]; then log "SKIP $n (done)"; return 0; fi
  log "START $n"; "$@" >> ~/post_eval.log 2>&1; local rc=$?; log "END $n rc=$rc"; [ $rc = 0 ] && touch "$m"; return $rc; }
# wait until the orchestrator writes the chosen adapter dir (phase B done + best checkpoint picked)
until [ -s ~/final_adapter_dir.txt ]; do sleep 60; done
FIN=$(cat ~/final_adapter_dir.txt)
log "final adapter dir (chosen): $FIN"
step convert $APY $A/convert_adapter_for_vllm.py $FIN $CONV
step make_lmh $APY $A/make_lmh_model.py $CONV/lm_head_lora.safetensors $LMH
step letter_base $VPY $A/eval_letter_adapter.py --tag base --mtp
step letter_r2 $VPY $A/eval_letter_adapter.py --tag r2 --model $LMH --lora $CONV
step loop_base $VPY $A/loop_test_g.py --model $BASE --label base --tag g150 --mtp
step loop_r2 $VPY $A/loop_test_g.py --model $LMH --label r2 --tag g150 --lora $CONV
step loop_cmp bash -c "$APY $A/loop_compare_v2.py g150 g150 --adapter-label r2 --semantic > $E/loop_compare_v2_r2.txt 2>&1"
step gpqa_r2 $VPY ~/ref_eval_scripts/gpqa_ab_g.py --model $LMH --label r2 --lora $CONV --compare-to $E/ab_base_lamba_gpqa_full198_samples.jsonl
step merge_fid bash -c "$APY $A/merge_fidelity.py $CONV/adapter_model.safetensors > $E/merge_fidelity_r2.txt 2>&1"
log "POST EVAL ALL DONE"
