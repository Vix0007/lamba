#!/usr/bin/env bash
exec >> ~/train1810p2.log 2>&1
until ! systemctl --user is-active -q lamba-train1810p2; do sleep 30; done
curl -s -m 5 127.0.0.1:8001/v1/models | grep -q l1810p2 || { echo "P2SAFETY_NO_SERVER"; exit 1; }
cd /home/ubuntu/antidoom_data; E=/home/ubuntu/eval_results
python3 -u strongreject_probe.py l1810p2 $E/matrix/l1810p2_merged_strongreject.jsonl > $E/l1810p2/sr.log 2>&1; echo "P2_SR_DONE"
python3 -u xstest_eval.py l1810p2 $E/matrix/l1810p2_merged_xstest.jsonl > $E/l1810p2/xstest.log 2>&1; echo "P2_XSTEST_DONE $(date -u +%H:%M)"
