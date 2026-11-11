#!/bin/bash
# v6: after training -> convert checkpoints 20..120 (+final) -> ONE-engine multi-adapter gate (gate_multi.py) -> summary in ~/v6_chain.log. Loop150 for the chosen checkpoint is launched by the orchestrator.
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; VPY=~/venv-vllm/bin/python; APY=~/antidoom/.venv/bin/python; L=~/antidoom/runs/lamba6/lora
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/v6_chain.log; }
until grep -q "V6 FINISHED" ~/orch.log; do sleep 30; done
log "v6 training finished"
ADS=""
for c in $(ls -d $L/checkpoint-* | sed 's/.*checkpoint-//' | sort -n); do
  case $c in 20|40|60|80|100|120) ;; *) continue;; esac
  $APY $A/convert_adapter_for_vllm.py $L/checkpoint-$c ~/adapters_v6_c$c >> ~/v6_chain.log 2>&1 && ADS="$ADS $HOME/adapters_v6_c$c"
done
$APY $A/convert_adapter_for_vllm.py $L ~/adapters_v6_final >> ~/v6_chain.log 2>&1 && ADS="$ADS $HOME/adapters_v6_final"
log "gating: $ADS"
$VPY $A/gate_multi.py v6 $ADS >> ~/v6_chain.log 2>&1
log "V6 GATE DONE"
