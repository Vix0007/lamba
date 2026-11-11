#!/bin/bash
# v9c: after training -> convert+lmh -> gate24 -> MalayMMLU letter eval -> 150 test -> paired stats vs base and v9b. log ~/v9c_s2_eval.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python; APY=~/antidoom/.venv/bin/python; L=~/antidoom/runs/lamba9c_s2/lora; LMH=/home/ubuntu/models/lmh_v9c_s2_final; AD=~/adapters_v9c_s2_final
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/v9c_s2_eval.log; }
until grep -q "V9CS2 FINISHED" ~/orch.log; do sleep 30; done
log "v9c finished"
$APY $A/convert_adapter_for_vllm.py $L $AD >> ~/v9c_s2_eval.log 2>&1 || { log convert failed; exit 1; }
$APY $A/make_lmh_model.py $AD/lm_head_lora.safetensors $LMH --scale 1.0 >> ~/v9c_s2_eval.log 2>&1 || { log lmh failed; exit 1; }
$VPY $A/loop_test_g.py --model $LMH --label v9cs2final --tag gate --prompts $A/loop_gate24.jsonl --max-tokens 1200 --lora $AD --chunk 24 >> ~/v9c_s2_eval.log 2>&1
$APY $A/gate_check.py v9cs2final > $E/v9cs2_gate_final.txt 2>&1; cat $E/v9cs2_gate_final.txt >> ~/v9c_s2_eval.log
grep -q '"pass": true' $E/v9cs2_gate_final.txt || { log "GATE FAILED -> stop (consult DS)"; exit 0; }
log "START letter"; $VPY $A/eval_letter_adapter.py --tag v9cs2final --model $LMH --lora $AD >> ~/v9c_s2_eval.log 2>&1; log "END letter"
log "START g150"; $VPY $A/loop_test_g.py --model $LMH --label v9cs2final --tag g150 --prompts $A/loop_testset150.jsonl --max-tokens 4000 --chunk 75 --lora $AD >> ~/v9c_s2_eval.log 2>&1; log "END g150"
$APY $A/tail_stats_cmp.py $E/loop_base_g150.jsonl $E/loop_v9cs2final_g150.jsonl $A/loop_testset150.jsonl > $E/v9cs2_vs_base_g150.txt 2>&1
$APY $A/tail_stats_cmp.py $E/loop_v9cfinal_g150.jsonl $E/loop_v9cs2final_g150.jsonl $A/loop_testset150.jsonl > $E/v9cs2_vs_v9c_g150.txt 2>&1
log "V9CS2 EVAL DONE"
