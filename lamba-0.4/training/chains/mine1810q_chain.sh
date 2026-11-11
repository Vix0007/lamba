#!/usr/bin/env bash
# 1810q mining: base server -> humour (n=4) -> more loop rollouts (time-capped) -> stop server -> loop cands/alts/pairs. log ~/mine1810q.log
exec >> ~/mine1810q.log 2>&1
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/home/ubuntu/venv-vllm/bin:$PATH
A=/home/ubuntu/antidoom_data; E=/home/ubuntu/eval_results/id1810q; O=/home/ubuntu/eval_results/id1810; P=/home/ubuntu/antidoom/.venv/bin/python
until grep -q FIX2_ALL_DONE ~/run03fix2.log && ! systemctl --user is-active -q lamba-run03fix2; do sleep 20; done
echo "Q_START $(date -u +%H:%M)"
systemd-run --user --unit=lamba-infer --collect -p MemoryMax=27G -p MemorySwapMax=0 --working-directory=/home/ubuntu bash /home/ubuntu/launch_infer_base.sh
for i in $(seq 1 120); do curl -s http://127.0.0.1:8001/v1/models | grep -q '"base"' && break; sleep 10; done
curl -s http://127.0.0.1:8001/v1/models | grep -q '"base"' || { echo Q_SERVE_FAILED; exit 1; }
echo "Q_SERVED $(date -u +%H:%M)"
python3 $A/mine_humour_1810q.py base $E/humour_base.jsonl; echo "Q_HUMOUR_DONE $(date -u +%H:%M)"
[ -f $E/loop_base_rollouts.jsonl ] || cp $O/loop_base_rollouts.jsonl $E/loop_base_rollouts.jsonl
timeout 6000 python3 $A/gen_loop_mine_model.py $A/loop_mine_1810_prompts.jsonl $E/loop_base_rollouts.jsonl 1 base; echo "Q_LOOPGEN_ENDED $(date -u +%H:%M) rows $(wc -l < $E/loop_base_rollouts.jsonl)"
systemctl --user stop lamba-infer; sleep 10
python3 $A/mine_round2.py $E/loop_base_rollouts.jsonl $A/loop_mine_1810_prompts.jsonl $E/loop_cands.jsonl; echo "Q_LOOPCANDS $(wc -l < $E/loop_cands.jsonl)"
$P $A/mine_alts_1810.py --cands $E/loop_cands.jsonl --out $E/loop_alts.jsonl; echo "Q_ALTS_DONE $(date -u +%H:%M)"
python3 $A/assemble_round2.py $E/loop_cands.jsonl $E/loop_alts.jsonl $E/loop_pairs.jsonl; echo "Q_LOOPPAIRS $(wc -l < $E/loop_pairs.jsonl)"
echo Q_MINE_ALL_DONE $(date -u +%H:%M)
