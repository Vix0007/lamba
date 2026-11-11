#!/bin/bash
# after rc5_suite: sampled T=0.7 loops on keep100 (same as v9c/malay-final robustness check), then critique pilot on full hard+guard repeats? no: more g150 seeds not needed. log ~/rc5_suite.log SUITE2_*
exec >> ~/rc5_suite.log 2>&1
M=l1810z; E=$HOME/eval_results/l1810z; A=$HOME/antidoom_data; P=$HOME/antidoom/.venv/bin/python; cd $A
until ! systemctl --user is-active --quiet lamba-rc5suite; do sleep 30; done
$P -u loop_test_server_t07.py $M $E/T07s1.jsonl --rp 1.05 --conc 16 --prompts $A/eval_keep100s_prompts.jsonl; echo SUITE2_T07_DONE
echo "SUITE2_DONE $(TZ=Asia/Singapore date +%H:%M)"
