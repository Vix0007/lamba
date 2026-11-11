#!/bin/bash
# letter-bias fix v2: check 5 checkpoints sequentially on the FULL 23,903-item MalayMMLU, stop early on first pass. log ~/letterfix2_chain.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python; APY=~/antidoom/.venv/bin/python
L=~/antidoom/runs/lamba_letterfix2/lora
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/letterfix2_chain.log; }
for step in 30 60 90 110 130; do
  ck="$L/checkpoint-$step"
  [ -d "$ck" ] || { log "no checkpoint-$step, skip"; continue; }
  AD=~/adapters_letterfix2_c$step
  LMH=/home/ubuntu/models/lmh_letterfix2_c$step
  log "=== checkpoint-$step: convert"
  $APY $A/convert_adapter_for_vllm.py "$ck" "$AD" >> ~/letterfix2_chain.log 2>&1 || { log "convert failed c$step"; continue; }
  log "=== checkpoint-$step: make_lmh"
  $APY $A/make_lmh_model.py "$AD/lm_head_lora.safetensors" "$LMH" --scale 1.0 >> ~/letterfix2_chain.log 2>&1 || { log "lmh failed c$step"; continue; }
  log "=== checkpoint-$step: eval (job A only, full 23,903)"
  $VPY $A/eval_letter_adapter.py --model "$LMH" --lora "$AD" --tag "letterfix2c$step" --jobs A >> ~/letterfix2_chain.log 2>&1
  echo "CHECKPOINT_DONE $step" >> ~/letterfix2_chain.log
done
echo "LETTERFIX2_CHAIN_ALL_DONE" >> ~/letterfix2_chain.log
