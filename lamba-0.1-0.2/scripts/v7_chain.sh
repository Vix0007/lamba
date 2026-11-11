#!/bin/bash
# v7: after training -> for checkpoints [FINAL, 140, 80] (each needs its own lm_head-merged model because the adapter has an lm_head LoRA r16 alpha16 => scale 1.0): convert -> make_lmh -> gate24 (1200 tok) -> gate_check (health + liquid loops). log ~/v7_chain.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python; APY=~/antidoom/.venv/bin/python; L=~/antidoom/runs/lamba7/lora
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/v7_chain.log; }
until grep -q "V7 FINISHED" ~/orch.log; do sleep 30; done
log "v7 training finished; ckpts: $(ls $L | tr '\n' ' ')"
for cand in final:$L $(for c in 140 80; do [ -d $L/checkpoint-$c ] && echo ckpt$c:$L/checkpoint-$c; done); do
  n=${cand%%:*}; d=${cand#*:}
  log "GATE $n"
  $APY $A/convert_adapter_for_vllm.py $d ~/adapters_v7_$n >> ~/v7_chain.log 2>&1 || { log "convert $n failed"; continue; }
  $APY $A/make_lmh_model.py ~/adapters_v7_$n/lm_head_lora.safetensors /home/ubuntu/models/lmh_v7_$n --scale 1.0 >> ~/v7_chain.log 2>&1 || { log "lmh $n failed"; continue; }
  $VPY $A/loop_test_g.py --model /home/ubuntu/models/lmh_v7_$n --label v7$n --tag gate --prompts $A/loop_gate24.jsonl --max-tokens 1200 --lora ~/adapters_v7_$n --chunk 24 >> ~/v7_chain.log 2>&1
  $APY $A/gate_check.py v7$n > $E/v7_gate_$n.txt 2>&1; cat $E/v7_gate_$n.txt >> ~/v7_chain.log
done
log "V7 GATES DONE"
