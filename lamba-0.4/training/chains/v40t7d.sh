#!/usr/bin/env bash
# after v40t7s evalq ends: rebuild merged v40t7, run diagnostic queue v40t7d, then serve v40t2 and mine prayer (rulesys3). log ~/v40t7d.log markers T7D_*
exec >> ~/v40t7d.log 2>&1
A=/home/ubuntu/antidoom_data; P=/home/ubuntu/antidoom/.venv/bin/python; ts(){ TZ=Asia/Singapore date +%T; }
cd $A
until [ "$(timeout 90 python3 evalq.py v40t7s status 2>/dev/null | grep -c DONE)" -ge 21 ] || ! systemctl --user is-active --quiet lamba-v40t7s; do sleep 60; done
echo "T7D_7S_QUEUE_ENDED $(ts) done=$(timeout 90 python3 evalq.py v40t7s status | grep -c DONE)"
systemctl --user stop lamba-v40t7s lamba-infer 2>/dev/null; sleep 20
rm -r /home/ubuntu/models/lamba04v40t7s_merged_fp8b128
$P -u $A/merge_quant_block128.py /home/ubuntu/adapters_lamba_04_v40t7 /home/ubuntu/models/lmh_lamba_04_v40t7 /home/ubuntu/models/lamba04v40t7_merged_fp8b128 --base /home/ubuntu/models/cand04_bf16_sh || { echo T7D_MERGE_FAILED; exit 1; }
echo "T7D_MERGED $(ts)"
systemd-run --user --unit=lamba-infer --collect -p MemoryMax=27G -p MemorySwapMax=0 ~/launch_vllm_v40t7_merged.sh >/dev/null
until curl -s -m 5 http://127.0.0.1:8001/v1/models | grep -q lamba04v40t7; do systemctl --user is-active --quiet lamba-infer || { echo T7D_SERVE_FAILED; exit 1; }; sleep 15; done
echo "T7D_SERVED $(ts)"
python3 evalq.py v40t7d init; python3 -u evalq.py v40t7d run box1:2 > ~/evalq_v40t7d_box1.log 2>&1
echo "T7D_EVALS_DONE $(ts) done=$(timeout 90 python3 evalq.py v40t7d status | grep -c DONE)"
systemctl --user stop lamba-infer; sleep 20
systemd-run --user --unit=lamba-infer --collect -p MemoryMax=27G -p MemorySwapMax=0 ~/launch_vllm_v40t2_merged.sh >/dev/null
until curl -s -m 5 http://127.0.0.1:8001/v1/models | grep -q lamba04v40t2; do systemctl --user is-active --quiet lamba-infer || { echo T7D_SERVE2_FAILED; exit 1; }; sleep 15; done
echo "T7D_V40T2_SERVED $(ts)"
mkdir -p ~/eval_results/v40t8_mine
~/venv-vllm/bin/python -u prayer_rulesys3.py lamba04v40t2 ~/eval_results/v40t8_mine/prayer_rulesys3_v40t2.jsonl 4 && echo "T7D_PRAYER_MINED $(ts)"
