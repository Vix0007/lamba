#!/usr/bin/env bash
# user 14:30: terminate the rented box once everything is pulled. Verify the pulled file is complete first.
until grep -q "MINE_B_DONE_PULLED\|BOXB_MINE_FAILED" ~/v40t7.log; do sleep 30; done
n=$(wc -l < ~/eval_results/v40t7_mine/const_B.jsonl 2>/dev/null || echo 0); want=$(python3 -c "import json;print(len(json.load(open('/home/ubuntu/antidoom_data/v40t7_mine_B.json'))))")
if [ "$n" -ge "$want" ]; then r=$(bash ~/box14_setup.sh terminate); echo "BOXB_TERMINATED rows=$n/$want state=$r $(TZ=Asia/Singapore date +%T)" >> ~/v40t7.log
else echo "BOXB_NOT_TERMINATED pull incomplete rows=$n/$want (check by hand) $(TZ=Asia/Singapore date +%T)" >> ~/v40t7.log; fi
