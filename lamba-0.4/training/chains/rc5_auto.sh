#!/bin/bash
# rc5 (1810z) FULLY AUTOMATIC: mining on rc4 (hate/illegal) -> pairs -> DS safety audit -> build -> gates -> free server/disk -> train chain.
exec >> ~/rc5_auto.log 2>&1
cd ~/antidoom_data; E=~/eval_results/id1810t; P=~/antidoom/.venv/bin/python
echo "RC5_START $(TZ=Asia/Singapore date +%H:%M)"
until ! systemctl --user is-active -q lamba-minerc5; do sleep 15; done; echo "RC5_MINING_DONE rows=$(wc -l < $E/safe3_y.jsonl)"
REF_FILE=safe3_y.jsonl $P sel_pairs_1810t.py comply $E/x_comply3_raw.jsonl 2>&1 | tail -1
REF_FILE=safe3_y.jsonl $P sel_pairs_1810t.py refalt $E/x_refalt3_raw.jsonl 2>&1 | tail -1
cat $E/x_comply3_raw.jsonl $E/x_refalt3_raw.jsonl > $E/x_safety3_raw.jsonl
AUD_TAG=r5 python3 audit_pairs.py build safety $E/x_safety3_raw.jsonl; N=$(ls ~/ds_queue/audsafety-r5_*.txt | wc -l)
t=0; until [ "$(ls ~/ds_queue/out | grep -c '^audsafety-r5_')" -ge $N ] || [ $t -ge 1500 ]; do sleep 5; t=$((t+5)); done
python3 audit_pairs.py parse safety; python3 audit_apply.py safety $E/x_safety3_raw.jsonl $E/x_safety3.jsonl
python3 - <<'PY'
import json
E='/home/ubuntu/eval_results/id1810t/'
fc=open(E+'x_comply3.jsonl','w'); fr=open(E+'x_refalt3.jsonl','w'); c=r=0
for l in open(E+'x_safety3.jsonl'):
    p=json.loads(l)
    if p.get('src')=='comply' or p['domain']=='t_comply': fc.write(l); c+=1
    else: fr.write(l); r+=1
print('rc5 fresh safety pairs: comply',c,'refalt',r)
PY
echo "RC5_PAIRS_DONE"
A_CAP=400 python3 build_1810z.py > ~/build_1810z_final.txt 2>&1; cat ~/build_1810z_final.txt
python3 - <<'PY'
import re,sys
t=open('/home/ubuntu/build_1810z_final.txt').read()
g=lambda k: int((re.search(r'^'+k+r'\s+(\d+)',t,re.M) or [0,0])[1])
tda=g('N_tdA_pairs'); unr=g('N_unrefuse_pairs'); lc=g('N_lengthcap_pairs'); new=g('N_comply3')+g('N_refalt3')
print('GATES tdA',tda,'unrefuse',unr,'lengthcap',lc,'fresh_hate_illegal',new)
ok = tda>=200 and unr>=100 and lc>=50 and new>=40
print('GATES_OK' if ok else 'GATES_SHORT'); sys.exit(0 if ok else 3)
PY
if [ $? -ne 0 ]; then echo "RC5_STOP gates short"; exit 3; fi
# let the decisive rc4 evals still running finish (max 40 min), score rc4, then free the GPU
t=0; until python3 - <<'PY'
import sys; sys.path.insert(0,'/home/ubuntu/antidoom_data'); import evalq
d=evalq.keys('l1810y','done'); sys.exit(0 if {'oversearch','safety_probe','identity'} <= d else 1)
PY
do sleep 20; t=$((t+20)); [ $t -ge 2400 ] && break; done
python3 score_all.py l1810y l1810x > /dev/null 2>&1; echo "RC5_RC4_SCORED"
systemctl --user stop lamba-train1810y lamba-infer 2>/dev/null; sleep 10
bash ~/s3_push_1810x.sh > ~/s3_push_rc4_final.log 2>&1
rm -rf ~/models/lamba04_1810y_merged_fp8b128 ~/models/lmh_lamba_04_1810y
F=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9); echo "disk free ${F}G"
[ "$F" -ge 60 ] || { echo "RC5_DISK_LOW"; exit 4; }
touch ~/.go1810z; systemd-run --user --unit=lamba-train1810z --collect -p MemoryMax=27G bash ~/train1810z_chain.sh
echo "RC5_TRAIN_LAUNCHED $(TZ=Asia/Singapore date +%H:%M)"
