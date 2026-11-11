#!/bin/bash
# after suite2: critique-only (answer->critique->final) on a random 1500-item MalayMMLU sample (not in subset) to settle the +2.4pp question. log ~/rc5_suite.log SUITE3_*
exec >> ~/rc5_suite.log 2>&1
cd $HOME/antidoom_data
until ! systemctl --user is-active --quiet lamba-rc5suite2; do sleep 30; done
PILOT_ITEMS=$HOME/eval_results/pilot/crit_sample1500.json $HOME/antidoom/.venv/bin/python -u pilot_tricks.py run l1810z $HOME/eval_results/pilot/crit1500_l1810z.jsonl --conc 40 2>&1 | grep -E "PILOT_DONE"; echo SUITE3_CRIT1500_DONE
