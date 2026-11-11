#!/bin/bash
# after suite7: second greedy runs of the larger loop sets for run-to-run noise bars: malay400, science300, r3_700. log ~/rc5_suite.log SUITE8_*
exec >> ~/rc5_suite.log 2>&1
cd $HOME/antidoom_data; P=$HOME/antidoom/.venv/bin/python; E=$HOME/eval_results/l1810z
until ! systemctl --user is-active --quiet lamba-rc5suite7; do sleep 30; done
$P -u loop_test_server.py l1810z $E/malay400_run2.jsonl --rp 1.05 --conc 16 --prompts $E/malay400_prompts.jsonl; echo SUITE8_MALAY400_R2_DONE
$P -u loop_test_server.py l1810z $E/science300_run2.jsonl --rp 1.05 --conc 16 --prompts $HOME/antidoom_data/science_eval_prep_prompts.jsonl; echo SUITE8_SCI_R2_DONE
$P -u loop_test_server.py l1810z $E/r3_700_run2.jsonl --rp 1.05 --conc 16 --prompts $HOME/antidoom_data/round3_prompts.jsonl; echo SUITE8_R3_R2_DONE
echo "SUITE8_DONE $(TZ=Asia/Singapore date +%H:%M)"
