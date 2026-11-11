#!/bin/bash
# after suite6: sampled-decoding seeds 2 and 3 on keep100, then sampled science300 (seed 1234). log ~/rc5_suite.log SUITE7_*
exec >> ~/rc5_suite.log 2>&1
cd $HOME/antidoom_data; P=$HOME/antidoom/.venv/bin/python; E=$HOME/eval_results/l1810z
until ! systemctl --user is-active --quiet lamba-rc5suite6; do sleep 30; done
for s in 2 3; do $P -u loop_test_server_t07s$s.py l1810z $E/T07s$s.jsonl --rp 1.05 --conc 16 --prompts $HOME/antidoom_data/eval_keep100s_prompts.jsonl; echo SUITE7_T07S${s}_DONE; done
$P -u loop_test_server_t07.py l1810z $E/science300_T07.jsonl --rp 1.05 --conc 16 --prompts $HOME/antidoom_data/science_eval_prep_prompts.jsonl; echo SUITE7_SCI_T07_DONE
echo "SUITE7_DONE $(TZ=Asia/Singapore date +%H:%M)"
