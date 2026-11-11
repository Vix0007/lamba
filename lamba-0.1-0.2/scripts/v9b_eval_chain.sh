#!/bin/bash
# v9b: after training -> convert + lmh merge (FINAL) -> gate24 (health) -> 380-prompt eval -> paired stats vs v8. log ~/v9b_eval.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python; APY=~/antidoom/.venv/bin/python; L=~/antidoom/runs/lamba9b/lora
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/v9b_eval.log; }
until grep -q "V9B FINISHED" ~/orch.log; do sleep 30; done
log "v9b finished; ckpts: $(ls $L | tr '\n' ' ')"
$APY $A/convert_adapter_for_vllm.py $L ~/adapters_v9b_final >> ~/v9b_eval.log 2>&1 || { log convert failed; exit 1; }
$APY $A/make_lmh_model.py ~/adapters_v9b_final/lm_head_lora.safetensors /home/ubuntu/models/lmh_v9b_final --scale 1.0 >> ~/v9b_eval.log 2>&1 || { log lmh failed; exit 1; }
$VPY $A/loop_test_g.py --model /home/ubuntu/models/lmh_v9b_final --label v9bfinal --tag gate --prompts $A/loop_gate24.jsonl --max-tokens 1200 --lora ~/adapters_v9b_final --chunk 24 >> ~/v9b_eval.log 2>&1
$APY $A/gate_check.py v9bfinal > $E/v9b_gate_final.txt 2>&1; cat $E/v9b_gate_final.txt >> ~/v9b_eval.log
if ! grep -q '"pass": true' $E/v9b_gate_final.txt; then log "GATE FAILED -> stopping chain (consult DS)"; exit 0; fi
log "gate passed; running 380-prompt eval"
$VPY $A/loop_test_g.py --model /home/ubuntu/models/lmh_v9b_final --label v9bfinal --tag ev380 --prompts $A/eval380_prompts.jsonl --max-tokens 4000 --chunk 128 --lora ~/adapters_v9b_final >> ~/v9b_eval.log 2>&1
$APY $A/tail_stats_cmp.py $E/loop_v8final_ev380.jsonl $E/loop_v9bfinal_ev380.jsonl $A/eval380_prompts.jsonl > $E/v9b_vs_v8_ev380.txt 2>&1; cat $E/v9b_vs_v8_ev380.txt >> ~/v9b_eval.log
log "V9B EVAL DONE"
