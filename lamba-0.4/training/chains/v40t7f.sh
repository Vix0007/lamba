#!/usr/bin/env bash
# full scorecard of v40t7 (remaining 14 jobs; 7 already done in v40t7d/t7). log ~/v40t7f.log markers T7F_*
exec >> ~/v40t7f.log 2>&1
A=/home/ubuntu/antidoom_data; P=/home/ubuntu/antidoom/.venv/bin/python; ts(){ TZ=Asia/Singapore date +%T; }
cd $A
echo "T7F_START $(ts)"
$P -u $A/merge_quant_block128.py /home/ubuntu/adapters_lamba_04_v40t7 /home/ubuntu/models/lmh_lamba_04_v40t7 /home/ubuntu/models/lamba04v40t7_merged_fp8b128 --base /home/ubuntu/models/cand04_bf16_sh || { echo T7F_MERGE_FAILED; exit 1; }
echo "T7F_MERGED $(ts)"
systemd-run --user --unit=lamba-infer --collect -p MemoryMax=27G -p MemorySwapMax=0 ~/launch_vllm_v40t7_merged.sh >/dev/null
until curl -s -m 5 http://127.0.0.1:8001/v1/models | grep -q lamba04v40t7; do systemctl --user is-active --quiet lamba-infer || { echo T7F_SERVE_FAILED; exit 1; }; sleep 15; done
echo "T7F_SERVED $(ts)"
python3 evalq.py v40t7f init; python3 -u evalq.py v40t7f run box1:2 > ~/evalq_v40t7f_box1.log 2>&1
echo "T7F_EVALS_DONE $(ts) done=$(timeout 90 python3 evalq.py v40t7f status | grep -c DONE)"
