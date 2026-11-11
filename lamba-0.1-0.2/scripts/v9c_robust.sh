#!/bin/bash
# v9c robustness bundle (DS 80): sampled decoding (T=0.7, seed 1234, 100 prompts) base vs v9c -> GPQA-D v9c -> 380-prompt eval v9c. log ~/v9c_robust.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python; APY=~/antidoom/.venv/bin/python; LMH=/home/ubuntu/models/lmh_v9c_final; AD=~/adapters_v9c_final
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/v9c_robust.log; }
log "START base_T07"; $VPY $A/loop_test_gt.py --model /home/ubuntu/models/soup-linear-l1.0 --label base --tag T07s1 --prompts $A/eval_keep100s_prompts.jsonl --max-tokens 4000 --chunk 100 --mtp --temp 0.7 --seed 1234 >> ~/v9c_robust.log 2>&1
log "START v9c_T07"; $VPY $A/loop_test_gt.py --model $LMH --label v9cfinal --tag T07s1 --prompts $A/eval_keep100s_prompts.jsonl --max-tokens 4000 --chunk 100 --lora $AD --temp 0.7 --seed 1234 >> ~/v9c_robust.log 2>&1
$APY $A/tail_stats_cmp.py $E/loop_base_T07s1.jsonl $E/loop_v9cfinal_T07s1.jsonl $A/eval_keep100s_prompts.jsonl > $E/v9c_vs_base_T07.txt 2>&1; cat $E/v9c_vs_base_T07.txt >> ~/v9c_robust.log
log "START gpqa"; $VPY ~/ref_eval_scripts/gpqa_ab_g.py --model $LMH --label v9cfinal --lora $AD --compare-to $E/ab_base_lamba_gpqa_full198_samples.jsonl >> ~/v9c_robust.log 2>&1; log "END gpqa"
log "START ev380"; $VPY $A/loop_test_g.py --model $LMH --label v9cfinal --tag ev380 --prompts $A/eval380_prompts.jsonl --max-tokens 4000 --chunk 128 --lora $AD >> ~/v9c_robust.log 2>&1
$APY $A/tail_stats_cmp.py $E/loop_v8final_ev380.jsonl $E/loop_v9cfinal_ev380.jsonl $A/eval380_prompts.jsonl > $E/v9c_vs_v8_ev380.txt 2>&1
$APY $A/tail_stats_cmp.py $E/loop_v9bfinal_ev380.jsonl $E/loop_v9cfinal_ev380.jsonl $A/eval380_prompts.jsonl > $E/v9c_vs_v9b_ev380.txt 2>&1
log "V9C ROBUST DONE"
