#!/bin/bash
# check more already-trained letterfix2 checkpoints (cheap, no retraining) adjacent to the good c30 candidate. log ~/letterfix2_more.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; VPY=~/venv-vllm/bin/python; APY=~/antidoom/.venv/bin/python
L=~/antidoom/runs/lamba_letterfix2/lora
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/letterfix2_more.log; }
for step in 20 40 10 50; do
  ck="$L/checkpoint-$step"
  [ -d "$ck" ] || { log "no checkpoint-$step, skip"; continue; }
  AD=~/adapters_letterfix2_c$step
  LMH=/home/ubuntu/models/lmh_letterfix2_c$step
  log "=== checkpoint-$step: convert"
  $APY $A/convert_adapter_for_vllm.py "$ck" "$AD" >> ~/letterfix2_more.log 2>&1 || continue
  log "=== checkpoint-$step: make_lmh"
  $APY $A/make_lmh_model.py "$AD/lm_head_lora.safetensors" "$LMH" --scale 1.0 >> ~/letterfix2_more.log 2>&1 || continue
  log "=== checkpoint-$step: eval (job A, full 23,903)"
  $VPY $A/eval_letter_adapter.py --model "$LMH" --lora "$AD" --tag "letterfix2c$step" --jobs A >> ~/letterfix2_more.log 2>&1
  echo "CHECKPOINT_DONE $step" >> ~/letterfix2_more.log
done
echo "LETTERFIX2_MORE_DONE" >> ~/letterfix2_more.log
