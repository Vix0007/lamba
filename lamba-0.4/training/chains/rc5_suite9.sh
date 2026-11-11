#!/bin/bash
# noise bars for the 4 contested metrics (user request): g150 R_any/L, identity says-Lamba, persona following. repeat runs on rc5. log ~/rc5_suite.log SUITE9_*
exec >> ~/rc5_suite.log 2>&1
cd $HOME/antidoom_data; P=$HOME/antidoom/.venv/bin/python; E=$HOME/eval_results/l1810z; M=l1810z
echo "SUITE9_START $(TZ=Asia/Singapore date +%H:%M)"
$P -u loop_test_server.py $M $E/g150_run4.jsonl --rp 1.05 --conc 16; echo SUITE9_G4_DONE
python3 -u l03_identity_eval.py $M $E/identity_run2.jsonl --split heldout; echo SUITE9_ID2_DONE
python3 -u l03_r3_defect_eval.py $M $E/defect_run2.json; echo SUITE9_DEF2_DONE
$P -u loop_test_server.py $M $E/g150_run5.jsonl --rp 1.05 --conc 16; echo SUITE9_G5_DONE
python3 -u l03_identity_eval.py $M $E/identity_run3.jsonl --split heldout; echo SUITE9_ID3_DONE
python3 -u l03_r3_defect_eval.py $M $E/defect_run3.json; echo SUITE9_DEF3_DONE
$P -u loop_test_server.py $M $E/g150_run6.jsonl --rp 1.05 --conc 16; echo SUITE9_G6_DONE
echo "SUITE9_DONE $(TZ=Asia/Singapore date +%H:%M)"
