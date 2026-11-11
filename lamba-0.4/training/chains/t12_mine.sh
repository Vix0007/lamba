#!/usr/bin/env bash
exec >> ~/t12_mine.log 2>&1
cd ~/antidoom_data; ts(){ TZ=Asia/Singapore date +%T; }
systemctl --user stop lamba-infer; sleep 15
systemd-run --user --unit=lamba-infer --collect -p MemoryMax=27G -p MemorySwapMax=0 ~/launch_vllm_v40t2_merged.sh >/dev/null
until curl -s -m 5 http://127.0.0.1:8001/v1/models | grep -q lamba04v40t2; do systemctl --user is-active --quiet lamba-infer || { echo "T12_SERVE_FAILED"; exit 1; }; sleep 15; done
echo "T12_V40T2_SERVED $(ts)"
mkdir -p ~/eval_results/v40t12_mine
MINE_NMAX=4 ~/venv-vllm/bin/python -u mine_constv12c.py lamba04v40t2 /home/ubuntu/eval_results/v40t12_mine/nature_const.jsonl /home/ubuntu/web/lamba_constitution.md 4 t12_nature_prompts.json && echo "T12_NATURE_MINE_DONE $(ts)"
