#!/bin/bash
# v5: after training -> convert -> gate24 (informational) -> loop150 (LoRA on BASE model, no lm_head) -> compare+metric -> letter eval. log ~/v5_chain.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python; APY=~/antidoom/.venv/bin/python; BASE=/home/ubuntu/models/soup-linear-l1.0
L=~/antidoom/runs/lamba5/lora
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/v5_chain.log; }
until grep -q "V5 FINISHED" ~/orch.log; do sleep 30; done
log "v5 training finished"
FIN=$L; [ -f ~/v5_adapter_dir.txt ] && FIN=$(cat ~/v5_adapter_dir.txt)
$APY $A/convert_adapter_for_vllm.py $FIN ~/adapters_v5 >> ~/v5_chain.log 2>&1 || { log convert failed; exit 1; }
$VPY $A/loop_test_g.py --model $BASE --label v5 --tag gate --prompts $A/loop_gate24.jsonl --max-tokens 1200 --lora ~/adapters_v5 --chunk 24 >> ~/v5_chain.log 2>&1
$APY $A/gate_check.py v5 > $E/v5_gate.txt 2>&1; cat $E/v5_gate.txt >> ~/v5_chain.log; log "gate done"
$VPY $A/loop_test_g.py --model $BASE --label v5 --tag g150 --prompts $A/loop_testset150.jsonl --max-tokens 4000 --chunk 75 --lora ~/adapters_v5 >> ~/v5_chain.log 2>&1
$APY $A/loop_compare_v2.py g150 g150 --adapter-label v5 > $E/loop_compare_v5.txt 2>&1; log "loop compare written"
$APY $A/loop_metric.py $E/loop_base_g150.jsonl $E/loop_v5_g150.jsonl > $E/loop_metric_v5.txt 2>&1
$VPY $A/eval_letter_adapter.py --tag v5 --model $BASE --lora ~/adapters_v5 >> ~/v5_chain.log 2>&1
log "V5 CHAIN DONE"
