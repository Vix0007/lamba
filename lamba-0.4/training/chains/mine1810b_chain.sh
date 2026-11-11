#!/usr/bin/env bash
# 1810 part 2: base loop rollouts -> restored constitution prompts -> stop server -> loop cands -> alts -> loop pairs. log ~/mine1810b.log
exec >> ~/mine1810b.log 2>&1
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/home/ubuntu/venv-vllm/bin:$PATH
A=/home/ubuntu/antidoom_data; E=/home/ubuntu/eval_results/id1810; P=/home/ubuntu/antidoom/.venv/bin/python
until grep -q MINE1810_ALL_DONE ~/mine1810.log && ! systemctl --user is-active -q lamba-mine1810; do sleep 30; done
echo "LOOPGEN_START $(date -u +%H:%M)"
python3 $A/gen_loop_mine_model.py $A/loop_mine_1810_prompts.jsonl $E/loop_base_rollouts.jsonl 1 base; echo "LOOPGEN_ENDED $(date -u +%H:%M)"
echo "RESTORE_SKIPPED (time)"
systemctl --user stop lamba-infer; sleep 10
python3 $A/mine_round2.py $E/loop_base_rollouts.jsonl $A/loop_mine_1810_prompts.jsonl $E/loop_cands_1810.jsonl; echo "LOOPCANDS $(wc -l < $E/loop_cands_1810.jsonl)"
$P $A/mine_alts_1810.py --cands $E/loop_cands_1810.jsonl --out $E/loop_alts_1810.jsonl; echo "ALTS_ENDED $(date -u +%H:%M)"
python3 $A/assemble_round2.py $E/loop_cands_1810.jsonl $E/loop_alts_1810.jsonl $E/loop_pairs_1810.jsonl; echo "LOOPPAIRS $(wc -l < $E/loop_pairs_1810.jsonl)"
echo MINE1810B_ALL_DONE $(date -u +%H:%M)
