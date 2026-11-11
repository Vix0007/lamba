#!/bin/bash
# ROUND 2 pipeline after r2gen2 (DS 76/77): mine r2gen2 -> concat candidates -> alts (GPU) -> assemble (relaxed filters) + reg rows -> choose epochs/lr by row count -> launch v9. log ~/round2_after_gen.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0 NONALNUM_OK=1
A=~/antidoom_data; APY=~/antidoom/.venv/bin/python; VPY=~/venv-vllm/bin/python
log(){ echo "$(date -u +%FT%TZ) $*" >> ~/round2_after_gen.log; }
until grep -q "R2GEN2 DONE" ~/r2_gen2.log 2>/dev/null; do sleep 30; done
log "r2gen2 done; mining"
$APY $A/mine_round2.py ~/eval_results/loop_v8final_r2gen2.jsonl $A/round2b_prompts.jsonl $A/round2_cands_r2gen2.jsonl >> ~/round2_after_gen.log 2>&1
cat $A/round2_cands_586half_v2.jsonl $A/round2_cands_r2gen_v2.jsonl $A/round2_cands_r2gen2.jsonl $A/round2_cands_old_sel.jsonl | $APY -c "
import sys,json
seen=set(); out=[]
for l in sys.stdin:
    r=json.loads(l)
    if r['source_sample_id'] in seen: continue
    seen.add(r['source_sample_id']); out.append(l)
open('$A/round2_cands_all2.jsonl','w').writelines(out); print('candidates total',len(out))" >> ~/round2_after_gen.log 2>&1
rm -f $A/round2_alts2.jsonl
$VPY $A/mine_alts.py --cands $A/round2_cands_all2.jsonl --out $A/round2_alts2.jsonl >> ~/round2_after_gen.log 2>&1
log "alts done"
$APY $A/assemble_round2.py $A/round2_cands_all2.jsonl $A/round2_alts2.jsonl $A/all_ftpo_pairs_round2_core.jsonl >> ~/round2_after_gen.log 2>&1
$APY $A/build_round2_reg.py ~/eval_results/loop_v8final_r2gen.jsonl $A/round2_prompts.jsonl $A/round2_dev_ids.txt $A/round2_reg_rows.jsonl 120 >> ~/round2_after_gen.log 2>&1
cat $A/all_ftpo_pairs_round2_core.jsonl $A/round2_reg_rows.jsonl > $A/all_ftpo_pairs_round2.jsonl
N=$(wc -l < $A/all_ftpo_pairs_round2_core.jsonl); log "core rows=$N ; total file rows=$(wc -l < $A/all_ftpo_pairs_round2.jsonl)"
cd ~/antidoom
if [ "$N" -ge 350 ]; then $APY $A/setcfg.py configs/lamba_r2_v9.yaml num_epochs=4 learning_rate=0.00001 lambda_mse=0.1; else $APY $A/setcfg.py configs/lamba_r2_v9.yaml num_epochs=3 learning_rate=0.000007 lambda_mse=0.1; fi
log "config set (rows=$N): $(grep -E '^\s+(num_epochs|learning_rate|lambda_mse):' configs/lamba_r2_v9.yaml | tr -s ' ' | tr '\n' ' ')"
log "launching v9"
bash ~/launch_v9.sh
log "ROUND2 PIPELINE DONE"
