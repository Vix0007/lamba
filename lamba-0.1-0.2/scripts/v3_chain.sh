#!/bin/bash
# v3: wait for training -> gate candidates (final, ckpt-100, ckpt-50) -> if a gate passes run loop(75 ids) + letter + gpqa. Markers in ~/eval_results/.v3done_*; log ~/v3_chain.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python; APY=~/antidoom/.venv/bin/python; BASE=/home/ubuntu/models/soup-linear-l1.0
L=~/antidoom/runs/lamba3/lora
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/v3_chain.log; }
until grep -q "V3 FINISHED" ~/orch.log; do sleep 30; done
log "training finished"
CH=""
for cand in final:$L ckpt100:$L/checkpoint-100 ckpt50:$L/checkpoint-50; do
  n=${cand%%:*}; d=${cand#*:}
  [ -e $E/.v3done_gate_$n ] && { grep -q '"pass": true' $E/v3_gate_$n.txt && { CH=$n; break; } || continue; }
  log "GATE $n ($d)"
  $APY $A/convert_adapter_for_vllm.py $d ~/adapters_v3_$n >> ~/v3_chain.log 2>&1 || { log "convert $n failed"; continue; }
  $APY $A/make_lmh_model.py ~/adapters_v3_$n/lm_head_lora.safetensors /home/ubuntu/models/lmh_v3_$n >> ~/v3_chain.log 2>&1 || { log "lmh $n failed"; continue; }
  $VPY $A/loop_test_g.py --model /home/ubuntu/models/lmh_v3_$n --label v3$n --tag gate --prompts $A/loop_gate24.jsonl --max-tokens 1200 --lora ~/adapters_v3_$n --chunk 24 >> ~/v3_chain.log 2>&1
  $APY $A/gate_check.py v3$n > $E/v3_gate_$n.txt 2>&1; cat $E/v3_gate_$n.txt >> ~/v3_chain.log; touch $E/.v3done_gate_$n
  grep -q '"pass": true' $E/v3_gate_$n.txt && { CH=$n; break; }
done
[ -z "$CH" ] && { log "NO CANDIDATE PASSED THE GATE"; exit 0; }
log "CHOSEN $CH"; echo $CH > ~/v3_chosen.txt
LMH=/home/ubuntu/models/lmh_v3_$CH; AD=~/adapters_v3_$CH
st(){ n=$1; shift; m=$E/.v3done_${CH}_$n; [ -e $m ] && return 0; log "START $n"; "$@" >> ~/v3_chain.log 2>&1 && touch $m; log "END $n"; }
st loop75 $VPY $A/loop_test_g.py --model $LMH --label v3$CH --tag g75 --prompts $A/loop_first75.jsonl --max-tokens 4000 --chunk 75 --lora $AD
$APY $A/loop_compare_v2.py g75 g75 --adapter-label v3$CH --semantic > $E/loop_compare_v3_$CH.txt 2>&1; log "loop compare written"
st letter $VPY $A/eval_letter_adapter.py --tag v3$CH --model $LMH --lora $AD
st gpqa $VPY ~/ref_eval_scripts/gpqa_ab_g.py --model $LMH --label v3$CH --lora $AD --compare-to $E/ab_base_lamba_gpqa_full198_samples.jsonl
log "V3 CHAIN DONE"
