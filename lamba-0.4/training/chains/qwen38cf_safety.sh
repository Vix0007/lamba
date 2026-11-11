#!/usr/bin/env bash
# Stock Qwen3.8-27B (Cloudflare via proxy :8011): safety + constitution-effect evals only (none vs constitution). Same harnesses as Lamba0.4-1810p2.
exec >> ~/qwen38cf_compare.log 2>&1
export EVAL_API=http://127.0.0.1:8011 PATH=/home/ubuntu/venv-vllm/bin:$PATH
cd /home/ubuntu/antidoom_data; O=/home/ubuntu/eval_results/qwen38cf; X=/home/ubuntu/eval_results/matrix; M=qwen38cf
echo "QCF2_START $(date -u +%H:%M)"
python3 xstest_eval.py $M $X/qwen38cf_xstest.jsonl --port 8011 > $O/xstest.log 2>&1; echo "QCF_XSTEST_DONE"
python3 -u warmth_gen.py $M bm/prayer_heldout.json $O/prayer_heldout.jsonl --n 4 --conds none,constitution --port 8011 > $O/prayer.log 2>&1; echo "QCF_PRAYER_DONE"
python3 -u warmth_gen.py $M warmth_eval_prompts.json $X/qwen38cf_warmth.jsonl --port 8011 > $O/warmth.log 2>&1; echo "QCF_WARMTH_DONE"
python3 -u consc_server.py $M qwen38cf --reps 5 --conc 12 > $O/consc.log 2>&1; echo "QCF_CONSC_DONE"
echo "QCF2_ALL_DONE $(date -u +%H:%M)"
