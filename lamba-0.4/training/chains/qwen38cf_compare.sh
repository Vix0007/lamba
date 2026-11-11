#!/usr/bin/env bash
# Stock Qwen3.8-27B (Cloudflare) through the SAME harnesses as Lamba0.4-1810p2, via local proxy :8011. All outputs jsonl in eval_results/qwen38cf/.
exec >> ~/qwen38cf_compare.log 2>&1
export EVAL_API=http://127.0.0.1:8011 PATH=/home/ubuntu/venv-vllm/bin:$PATH
cd /home/ubuntu/antidoom_data; O=/home/ubuntu/eval_results/qwen38cf; G=/home/ubuntu/eval_results/diag05; M=qwen38cf
curl -s -m 5 127.0.0.1:8011/v1/models | grep -q qwen38cf || { echo "QCF_NO_PROXY"; exit 1; }
echo "QCF_START $(date -u +%H:%M)"
python3 diag05_battery.py $M $G/qwen38cf.jsonl --n 4 --port 8011 > $O/name_battery_gen.log 2>&1 && python3 diag05_score.py qwen38cf=$G/qwen38cf.jsonl l1810p2=$G/l1810p2.jsonl > $O/name_battery.txt 2>&1; echo "QCF_NAME_DONE"
python3 l03_identity_eval.py $M $O/identity.jsonl --split heldout > $O/identity.log 2>&1; echo "QCF_IDENTITY_DONE"
python3 l03_r3_defect_eval.py $M $O/defect.json > $O/defect.log 2>&1; echo "QCF_DEFECT_DONE"
python3 tone_battery.py $M /home/ubuntu/eval_results/tone1810/qwen38cf.jsonl > $O/tone.log 2>&1; echo "QCF_TONE_DONE"
python3 diag05_emptyreplay.py $M $G/b2_v40t11p3.jsonl $G/emptyreplay_qwen38cf.jsonl --n 10 > $O/empty.log 2>&1; echo "QCF_EMPTY_DONE"
python3 xstest_eval.py $M /home/ubuntu/eval_results/matrix/qwen38cf_xstest.jsonl --port 8011 > $O/xstest.log 2>&1; echo "QCF_XSTEST_DONE"
python3 -u l03_tool_eval.py $M $O/toolv2.jsonl --split heldout --prompts l03_tool_prompts_v2.jsonl --temp 1.0 --top-p 0.95 > $O/tool.log 2>&1; echo "QCF_TOOL_DONE"
python3 -u warmth_gen.py $M bm/prayer_heldout.json $O/prayer_heldout.jsonl --n 4 --conds none,constitution --port 8011 > $O/prayer.log 2>&1; echo "QCF_PRAYER_DONE"
python3 -u gpqa_server.py --model-name $M --tokenizer /home/ubuntu/models/cand04_bf16_sh --tag qwen38cf --budget 4000 --rp 1.0 --conc 12 --port 8011 > $O/gpqa.log 2>&1; echo "QCF_GPQA_DONE"
python3 -u malaymmlu_gen.py $M $O/malaysub.jsonl --ids-file malaymmlu_sub_ids.json --conc 12 --rp 1.05 --port 8011 > $O/malaysub.log 2>&1; echo "QCF_MALAYSUB_DONE"
python3 -u malaymmlu_gen.py $M $O/malaygen5k_rp11.jsonl --conc 12 --rp 1.1 --port 8011 > $O/malay5k.log 2>&1; echo "QCF_MALAY5K_DONE"
echo "QCF_ALL_DONE $(date -u +%H:%M)"
