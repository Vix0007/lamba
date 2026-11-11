#!/bin/bash
# v9c: after training -> convert+lmh -> gate24 -> MalayMMLU letter eval -> 150 test -> paired stats vs base and v9b. log ~/malay_eval.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python; APY=~/antidoom/.venv/bin/python; L=~/antidoom/runs/lamba_malay/lora; LMH=/home/ubuntu/models/lmh_malay_final; AD=~/adapters_malay_final
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/malay_eval.log; }
until grep -q "MALAY FINISHED" ~/orch.log; do sleep 30; done
log "v9c finished"
$APY $A/convert_adapter_for_vllm.py $L $AD >> ~/malay_eval.log 2>&1 || { log convert failed; exit 1; }
$APY $A/make_lmh_model.py $AD/lm_head_lora.safetensors $LMH --scale 1.0 >> ~/malay_eval.log 2>&1 || { log lmh failed; exit 1; }
$VPY $A/loop_test_g.py --model $LMH --label malayfinal --tag gate --prompts $A/loop_gate24.jsonl --max-tokens 1200 --lora $AD --chunk 24 >> ~/malay_eval.log 2>&1
$APY $A/gate_check.py malayfinal > $E/malay_gate_final.txt 2>&1; cat $E/malay_gate_final.txt >> ~/malay_eval.log
grep -q '"pass": true' $E/malay_gate_final.txt || { log "GATE FAILED -> stop (consult DS)"; exit 0; }
log "START letter"; $VPY $A/eval_letter_adapter.py --tag malayfinal --model $LMH --lora $AD >> ~/malay_eval.log 2>&1; log "END letter"
log "START g150"; $VPY $A/loop_test_g.py --model $LMH --label malayfinal --tag g150 --prompts $A/loop_testset150.jsonl --max-tokens 4000 --chunk 75 --lora $AD >> ~/malay_eval.log 2>&1; log "END g150"
$APY $A/tail_stats_cmp.py $E/loop_base_g150.jsonl $E/loop_malayfinal_g150.jsonl $A/loop_testset150.jsonl > $E/malay_vs_base_g150.txt 2>&1
$VPY $A/loop_test_g.py --model $LMH --label malayfinal --tag malayex400 --prompts $A/malay_eval400_prompts.jsonl --max-tokens 4000 --chunk 100 --lora $AD >> ~/malay_eval.log 2>&1
cat $E/loop_base_malaygen.jsonl $E/loop_base_malayex.jsonl > /tmp/malaybase400.jsonl
$APY $A/tail_stats_cmp.py /tmp/malaybase400.jsonl $E/loop_malayfinal_malayex400.jsonl $A/malay_eval400_prompts.jsonl > $E/malay_eval400_vs_base.txt 2>&1
$APY $A/tail_stats_cmp.py $E/loop_v9cfinal_g150.jsonl $E/loop_malayfinal_g150.jsonl $A/loop_testset150.jsonl > $E/malay_vs_v9c_g150.txt 2>&1
log "MALAY EVAL DONE"
