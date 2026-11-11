#!/bin/bash
# Beat-ThinkingCap track: gate24 + dev-set (321 prompts) generation for the concise-SFT adapter,
# reusing the one-engine eval_combined.py pattern (gate + g150-style job in one load).
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python
LMH=/home/ubuntu/models/lmh_concise_sft; AD=~/adapters_concise_sft
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/concise_sft_eval.log; }

log "START gate+dev (one engine)"
$VPY $A/eval_combined.py --model $LMH --lora $AD --tag concisesft --jobs gate,g150 \
  --gate-prompts $A/loop_gate24.jsonl \
  --g150-prompts $A/concise_sft/dev_prompts.jsonl >> ~/concise_sft_eval.log 2>&1
log "END gate+dev"
