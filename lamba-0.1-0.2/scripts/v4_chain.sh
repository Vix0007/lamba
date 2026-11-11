#!/bin/bash
# v4: after training -> gate (base model + LoRA, NO lm_head merge) -> if pass: loop75 + compare + letter. log ~/v4_chain.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python; APY=~/antidoom/.venv/bin/python; BASE=/home/ubuntu/models/soup-linear-l1.0
L=~/antidoom/runs/lamba4/lora
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/v4_chain.log; }
until grep -q "V4 FINISHED" ~/orch.log; do sleep 30; done
log "v4 training finished"
$APY $A/convert_adapter_for_vllm.py $L ~/adapters_v4 >> ~/v4_chain.log 2>&1 || { log convert failed; exit 1; }
$VPY $A/loop_test_g.py --model $BASE --label v4 --tag gate --prompts $A/loop_gate24.jsonl --max-tokens 1200 --lora ~/adapters_v4 --chunk 24 >> ~/v4_chain.log 2>&1
$APY $A/gate_check.py v4 > $E/v4_gate.txt 2>&1; cat $E/v4_gate.txt >> ~/v4_chain.log
log "GATE done (note: v4 gate is informational; loop75 always runs next because time is short)"
$VPY $A/loop_test_g.py --model $BASE --label v4 --tag g75 --prompts $A/loop_first75.jsonl --max-tokens 4000 --chunk 75 --lora ~/adapters_v4 >> ~/v4_chain.log 2>&1
$APY $A/loop_compare_v2.py g75 g75 --adapter-label v4 --semantic > $E/loop_compare_v4.txt 2>&1; log "loop compare written"
$VPY $A/eval_letter_adapter.py --tag v4 --model $BASE --lora ~/adapters_v4 >> ~/v4_chain.log 2>&1
log "V4 CHAIN DONE"
