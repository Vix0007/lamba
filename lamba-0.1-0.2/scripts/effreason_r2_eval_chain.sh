#!/bin/bash
# effreason round 2 (v2, fixed onset): gate -> full GPQA-D b4000 + full MalayMMLU job A (one
# engine) -> g150 (per-domain paired stats). STOPS HERE deliberately -- the expensive fresh-1082
# confirmation run is launched separately, manually, only if g150 looks clean (v1 regressed
# badly on loops; no point burning ~8h confirming a bad result again).
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; E=~/eval_results; VPY=~/venv-vllm/bin/python; APY=~/antidoom/.venv/bin/python
LMH=/home/ubuntu/models/lmh_effreason_r2_final; AD=~/adapters_effreason_r2_final
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/effreason_r2_eval.log; }

log "START gate24"
$VPY $A/loop_test_g.py --model $LMH --label effreasonr2 --tag gate --prompts $A/loop_gate24.jsonl --max-tokens 1200 --lora $AD --chunk 24 >> ~/effreason_r2_eval.log 2>&1
$APY $A/gate_check.py effreasonr2 > $E/effreasonr2_gate.txt 2>&1; cat $E/effreasonr2_gate.txt >> ~/effreason_r2_eval.log
grep -q '"pass": true' $E/effreasonr2_gate.txt || { log "GATE FAILED -- stopping, consult before proceeding"; exit 0; }
log "gate PASSED"

log "START combined GPQA-D b4000 + full MalayMMLU job A"
$VPY $A/eval_combined.py --model $LMH --lora $AD --tag effreasonr2 --jobs gpqa,A --budget 4000 >> ~/effreason_r2_eval.log 2>&1
log "END combined eval"

log "START effreason_r2 g150"
$VPY $A/loop_test_g.py --model $LMH --label effreasonr2 --tag g150 --prompts $A/loop_testset150.jsonl --max-tokens 4000 --chunk 75 --lora $AD >> ~/effreason_r2_eval.log 2>&1
log "END effreason_r2 g150"

$APY $A/tail_stats_by_domain.py $E/loop_letterfix2c10_g150.jsonl $E/loop_effreasonr2_g150.jsonl $A/loop_testset150.jsonl > $E/effreasonr2_vs_base_g150_bydomain.txt 2>&1
cat $E/effreasonr2_vs_base_g150_bydomain.txt >> ~/effreason_r2_eval.log
log "EFFREASON_R2_EVAL_CHAIN_STAGE1 DONE -- review before launching the fresh-1082 confirmation"
