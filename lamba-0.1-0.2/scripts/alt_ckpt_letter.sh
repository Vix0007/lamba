#!/bin/bash
# POST-HOC (disclosed) letter eval of alternative checkpoints on MalayMMLU, to show how the natural-benchmark A-B gap varies. NOT used for selection. Runs after post_train_eval finishes. Resumable via markers.
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python; APY=~/antidoom/.venv/bin/python
log(){ echo "$(date -u +%FT%TZ) ALT $*" >> ~/post_eval.log; }
until grep -q "POST EVAL ALL DONE" ~/post_eval.log 2>/dev/null; do sleep 60; done
for spec in "v2c100:$HOME/antidoom/runs/keep_v2/checkpoint-100" "b100:$HOME/antidoom/runs/lamba2b/lora/checkpoint-100" "b50:$HOME/antidoom/runs/lamba2b/lora/checkpoint-50" "b124:$HOME/antidoom/runs/lamba2b/lora/checkpoint-124"; do
  n=${spec%%:*}; d=${spec#*:}; m=$E/.altdone_$n; [ -e $m ] && { log "skip $n"; continue; }
  log "START $n ($d)"
  $APY $A/convert_adapter_for_vllm.py $d ~/adapters_alt_$n >> ~/post_eval.log 2>&1 || { log "convert $n failed"; continue; }
  $APY $A/make_lmh_model.py ~/adapters_alt_$n/lm_head_lora.safetensors /home/ubuntu/models/lmh_alt_$n >> ~/post_eval.log 2>&1 || { log "lmh $n failed"; continue; }
  $VPY $A/eval_letter_adapter.py --tag alt_$n --model /home/ubuntu/models/lmh_alt_$n --lora ~/adapters_alt_$n >> ~/post_eval.log 2>&1 && touch $m
  log "END $n"
  rm -rf /home/ubuntu/models/lmh_alt_$n/model-*.safetensors 2>/dev/null   # free disk (3.4GB shard); symlinks left are harmless
done
log "ALT DONE"
