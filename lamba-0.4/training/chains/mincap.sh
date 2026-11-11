#!/bin/bash
# stop the extra rc5 miner once there are >=90 raw fresh pairs (comply+refalt) or at 17:15 SGT, whichever first; rc5d then proceeds with the partial file.
exec >> ~/mincap.log 2>&1
cd ~/antidoom_data; P=~/antidoom/.venv/bin/python; E=~/eval_results/id1810t
while systemctl --user is-active -q lamba-minerc5x; do
  sleep 480
  cat $E/safe3_y.jsonl $E/safe3x_y.jsonl > $E/yt_cap_tmp.jsonl
  c=$(REF_FILE=yt_cap_tmp.jsonl $P sel_pairs_1810t.py comply /tmp/cap_c.jsonl 2>&1 | tail -1 | grep -oE "'pairs': [0-9]+" | grep -oE "[0-9]+"); r=$(REF_FILE=yt_cap_tmp.jsonl $P sel_pairs_1810t.py refalt /tmp/cap_r.jsonl 2>&1 | tail -1 | grep -oE "'pairs': [0-9]+" | grep -oE "[0-9]+")
  rm -f $E/yt_cap_tmp.jsonl; n=$(( ${c:-0} + ${r:-0} )); now=$(TZ=Asia/Singapore date +%H%M); echo "$(TZ=Asia/Singapore date +%H:%M) raw pairs comply=$c refalt=$r total=$n rows=$(wc -l < $E/safe3x_y.jsonl)"
  if [ "$n" -ge 90 ] || [ "$now" -ge 1715 ]; then systemctl --user stop lamba-minerc5x; echo "MINCAP_STOPPED_EXTRA_MINER n=$n"; fi
done
