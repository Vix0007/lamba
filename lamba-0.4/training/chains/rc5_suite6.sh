#!/bin/bash
# after suite5: tool-augmented rumination families from the 0.1-era investigation, on rc5 (integral family x10 reps, matched iterative-symbolic x10). log ~/rc5_suite.log SUITE6_*
exec >> ~/rc5_suite.log 2>&1
cd $HOME/antidoom_data; P=$HOME/antidoom/.venv/bin/python; E=$HOME/eval_results/l1810z
until ! systemctl --user is-active --quiet lamba-rc5suite5; do sleep 30; done
$P -u gen_a3_08_family_conc.py l1810z $E/a3_08_family.jsonl 10 8 2>&1 | grep -E "^DONE|Traceback" ; echo SUITE6_A3_08_DONE
$P -u gen_matched_iterative.py l1810z $E/matched_iterative.jsonl 2>&1 | grep -E "Traceback"; echo SUITE6_MATCHED_DONE
echo "SUITE6_DONE $(TZ=Asia/Singapore date +%H:%M)"
