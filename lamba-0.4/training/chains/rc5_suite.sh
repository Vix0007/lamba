#!/bin/bash
# rc5 (l1810z) full 0.3-style eval suite on the served rc5 server; fully automatic, resumable. log ~/rc5_suite.log markers SUITE_*
exec >> ~/rc5_suite.log 2>&1
M=l1810z; E=$HOME/eval_results/l1810z; A=$HOME/antidoom_data; P=$HOME/antidoom/.venv/bin/python; cd $A; mkdir -p $E
echo "SUITE_START $(TZ=Asia/Singapore date +%H:%M)"
until ! systemctl --user is-active --quiet lamba-mmlu5k3; do sleep 30; done; echo SUITE_MMLU5K_DONE
$P -u gen_sandboxb_eval.py $M $E/eval30.jsonl rumination_eval30.jsonl 8 8; echo SUITE_E30_DONE
$P -u loop_test_server.py $M $E/g150_run2.jsonl --rp 1.05 --conc 16; echo SUITE_G2_DONE
$P -u loop_test_server.py $M $E/g150_run3.jsonl --rp 1.05 --conc 16; echo SUITE_G3_DONE
$P -u loop_test_server.py $M $E/r3_700.jsonl --rp 1.05 --conc 16 --prompts $A/round3_prompts.jsonl; echo SUITE_R3_DONE
cat $A/malay_gen_eval180.jsonl $A/malay_eval_extra220.jsonl > $E/malay400_prompts.jsonl
$P -u loop_test_server.py $M $E/malay400.jsonl --rp 1.05 --conc 16 --prompts $E/malay400_prompts.jsonl; echo SUITE_MALAY400_DONE
$P -u loop_test_server.py $M $E/science300.jsonl --rp 1.05 --conc 16 --prompts $A/science_eval_prep_prompts.jsonl; echo SUITE_SCI_DONE
$P -u gpqa_server.py --model-name $M --tokenizer /home/ubuntu/models/cand04_bf16_sh --tag l1810z_run2 --budget 4000 --rp 1.05 --conc 12; echo SUITE_GPQA2_DONE
echo "SUITE_ALL_DONE $(TZ=Asia/Singapore date +%H:%M)"
