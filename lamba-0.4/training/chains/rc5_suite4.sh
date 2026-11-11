#!/bin/bash
# after critique run: fill the XSTest constitution condition for all prompts (was skipped for time) -> matrix file l1810z_merged_xstest_full.jsonl. log ~/rc5_suite.log SUITE4_*
exec >> ~/rc5_suite.log 2>&1
cd $HOME/antidoom_data
until ! systemctl --user is-active --quiet lamba-rc5suite3; do sleep 30; done
XS_PARTIAL=$HOME/eval_results/matrix/l1810z_merged_xstest.jsonl $HOME/antidoom/.venv/bin/python -u xstest_eval_const.py l1810z $HOME/eval_results/matrix/l1810z_merged_xstest_full.jsonl; echo SUITE4_XSTEST_CONST_DONE
