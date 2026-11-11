#!/usr/bin/env bash
# rented box: setup, serve v40t2, mine half B under constitution v1.2c, pull results back. Terminate BY HAND afterwards.
exec >> ~/v40t7_boxB.log 2>&1
cd ~; bash ~/box14_setup.sh launch || { echo BOXB_LAUNCH_FAILED; exit 1; }
bash ~/box14_setup.sh serve lamba04v40t2_merged_fp8b128 lamba04v40t2 launch_vllm_v40t2_merged.sh || { echo BOXB_SERVE_FAILED; exit 1; }
IP=$(cat ~/.lamba-box14a.ip); S="ssh -i $HOME/.ssh/vix_fleet.pem -o ServerAliveInterval=30 ubuntu@$IP"
echo "BOXB_SERVED $(TZ=Asia/Singapore date +%T)" >> ~/v40t7.log
$S 'mkdir -p ~/eval_results/v40t7_mine; cd ~/antidoom_data; ~/venv-vllm/bin/python -u mine_constv12c.py lamba04v40t2 ~/eval_results/v40t7_mine/const_B.jsonl ~/web/drafts/lamba_constitution_v1.2c_draft.md 4 v40t7_mine_B.json' || { echo BOXB_MINE_FAILED >> ~/v40t7.log; exit 1; }
rsync -a -e "ssh -i $HOME/.ssh/vix_fleet.pem" ubuntu@$IP:~/eval_results/v40t7_mine/const_B.jsonl ~/eval_results/v40t7_mine/ && echo "MINE_B_DONE_PULLED $(wc -l < ~/eval_results/v40t7_mine/const_B.jsonl) rows $(TZ=Asia/Singapore date +%T)" >> ~/v40t7.log
