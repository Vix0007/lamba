#!/usr/bin/env bash
# waits for the loop chain, rebuilds final 1810 data, sanity gates, then touches ~/.go1810 (train chain starts). log ~/train1810.log
exec >> ~/train1810.log 2>&1
until grep -q "^MINE1810B_ALL_DONE" ~/mine1810b.log && ! systemctl --user is-active -q lamba-mine1810b; do sleep 30; done
cd ~/antidoom_data
~/antidoom/.venv/bin/python build_1810.py cands > ~/build1810_cands.txt 2>&1
~/antidoom/.venv/bin/python build_1810.py final > ~/build1810_final.txt 2>&1 || { echo "AUTOGO_BUILD_FAIL"; exit 1; }
cat ~/build1810_final.txt | grep -E "TOTAL|LOOP|hyg_|TOOL"
python3 - <<'PY' || { echo "AUTOGO_GATE_FAIL"; exit 1; }
import json,re,sys,collections
rows=[json.loads(l) for l in open('/home/ubuntu/antidoom_data/l03_training_data_1810.jsonl')]
k=collections.Counter(r.get('kind') for r in rows); print('kinds',dict(k))
loops=k.get(None,0)+k.get('ftpo',0)+k.get('ftpo_soft',0)
bad=sum(1 for r in rows if r.get('kind')=='sft_ce' and re.search(r'\bLambda\b|\b(nggak|banget|gimana)\b', r['response_text'].split('</think>')[-1]))
print('loops',loops,'bad',bad,'total',len(rows))
sys.exit(0 if (6500<=len(rows)<=9000 and 1<=loops<=4000 and bad==0) else 1)
PY
touch ~/.go1810; echo "AUTOGO_TOUCHED $(date -u +%H:%M)"
