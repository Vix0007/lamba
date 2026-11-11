#!/bin/bash
# after suite4: noise estimates for the report: GPQA run3 + plain MalayMMLU 5k run2 (different file). log ~/rc5_suite.log SUITE5_*
exec >> ~/rc5_suite.log 2>&1
cd $HOME/antidoom_data; P=$HOME/antidoom/.venv/bin/python
until ! systemctl --user is-active --quiet lamba-rc5suite4; do sleep 30; done
$P -u gpqa_server.py --model-name l1810z --tokenizer /home/ubuntu/models/cand04_bf16_sh --tag l1810z_run3 --budget 4000 --rp 1.05 --conc 12; echo SUITE5_GPQA3_DONE
$P -u malaymmlu_gen.py l1810z $HOME/eval_results/malaygen_l1810z_run2.jsonl --conc 44 --rp 1.05 | tail -2; echo SUITE5_MMLU5K_RUN2_DONE
echo "SUITE5_DONE $(TZ=Asia/Singapore date +%H:%M)"
