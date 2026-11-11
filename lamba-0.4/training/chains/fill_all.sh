#!/usr/bin/env bash
# Complete every partial scorecard (tech report). Sequential per model: rebuild merged -> serve -> evalq fill_<T> -> DS grading (bg) -> delete merged.
# log ~/fill_all.log markers FILL_*
exec >> ~/fill_all.log 2>&1
P=/home/ubuntu/antidoom/.venv/bin/python; A=/home/ubuntu/antidoom_data; BASE=/home/ubuntu/models/cand04_bf16_sh; E=/home/ubuntu/eval_results
ts(){ TZ=Asia/Singapore date +%T; }; cd $A
until grep -q "ALL WORKERS EXITED" ~/evalq_v40t11p3_box1.log 2>/dev/null; do sleep 30; done
echo "FILL_P3_QUEUE_DONE $(ts)"
( bash ~/lamba04_report_data/collect.sh > ~/collect_interim.log 2>&1; bash ~/s3_push_04.sh > ~/s3_push_interim.log 2>&1; echo "FILL_S3_INTERIM_DONE $(ts)" >> ~/fill_all.log ) &
serve(){ # $1 tag  $2 merged dir
  systemctl --user stop lamba-infer 2>/dev/null; sleep 15; until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 10; done
  sed "s#lamba04v40t2_merged_fp8b128#$2#; s#--served-model-name lamba04v40t2#--served-model-name lamba04$1#" ~/launch_vllm_v40t2_merged.sh > ~/launch_vllm_fill_$1.sh; chmod +x ~/launch_vllm_fill_$1.sh
  systemd-run --user --unit=lamba-infer --collect -p MemoryMax=27G -p MemorySwapMax=0 ~/launch_vllm_fill_$1.sh >/dev/null
  until curl -s -m 5 http://127.0.0.1:8001/v1/models | grep -q "lamba04$1\""; do systemctl --user is-active --quiet lamba-infer || { echo "FILL_SERVE_FAILED $1"; return 1; }; sleep 15; done
  echo "FILL_SERVED $1 $(ts)"; }
dsgrade(){ # $1 tag: prayer (none+const), warmth, pushback via DS; background
  T=$1
  ( cd $A
    if [ -s $E/$T/prayer_heldout.jsonl ]; then
      for c in none constitution; do tg=f${T//v40/}${c:0:1}; python3 prayer_judge2.py build $tg $tg=$E/$T/prayer_heldout.jsonl --cond $c >/dev/null
        N=$(ls ~/ds_queue/pj2_${tg}_*.txt | wc -l); for i in $(seq 1 240); do [ "$(ls ~/ds_queue/out/pj2_${tg}_*.md 2>/dev/null | wc -l)" -ge "$N" ] && break; sleep 30; done
        echo "FILL_PRAYER $T $c $(python3 prayer_judge2.py score $tg | tail -1)"; done
    fi
    if [ -s $E/matrix/${T}_merged_warmth.jsonl ] && [ ! -f ~/fillwg_$T.log ]; then tg=fw${T//v40/}; python3 warmth_grade.py build $tg $E/matrix/${T}_merged_warmth.jsonl >/dev/null
      N=$(ls ~/ds_queue/wg_${tg}_*.txt | wc -l); for i in $(seq 1 240); do [ "$(ls ~/ds_queue/out/wg_${tg}_*.md 2>/dev/null | wc -l)" -ge "$N" ] && break; sleep 30; done
      python3 warmth_grade.py score $tg > ~/fillwg_$T.log 2>&1; echo "FILL_WARMTH $T $(grep "/none" ~/fillwg_$T.log | head -1)"; fi
    if [ -s $E/matrix/${T}_merged_pushback.jsonl ] && [ ! -f ~/fillpb_$T.log ]; then tg=fp${T//v40/}; python3 pushback_eval.py build-grade $tg $E/matrix/${T}_merged_pushback.jsonl >/dev/null
      N=$(ls ~/ds_queue/*${tg}*.txt | wc -l); for i in $(seq 1 240); do [ "$(ls ~/ds_queue/out/*${tg}*.md 2>/dev/null | wc -l)" -ge "$N" ] && break; sleep 30; done
      python3 pushback_eval.py score $tg > ~/fillpb_$T.log 2>&1; echo "FILL_PUSHBACK $T $(grep /none ~/fillpb_$T.log | head -1)"; fi
  ) >> ~/fill_all.log 2>&1 & }
self40(){ CONSC_ONLY=self_ $P -u consc_server.py lamba04$1 ${1}_self40 --reps 40 --conds none > ~/self40_$1.log 2>&1; echo "FILL_SELF40 $1 $($P consc_server.py --summary $E/consc/${1}_self40_none.json | grep -o 'SelfAttribution=[0-9.]*')"; }
fill(){ # $1 tag  $2 adapter dir  $3 lmh dir
  T=$1; M=lamba04${T}_merged_fp8b128; mkdir -p $E/$T
  if [ ! -d /home/ubuntu/models/$M ]; then
    [ -f $3/lm_head.weight.safetensors ] || [ -d $3 ] || { $P $A/make_lmh_model.py $2/lm_head_lora.safetensors $3 --base $BASE --scale 1.0 || { echo "FILL_LMH_FAILED $T"; return; }; rm -f $3/generation_config.json; python3 -c "
import json; g=json.load(open('$BASE/generation_config.json')); g['repetition_penalty']=1.05; json.dump(g,open('$3/generation_config.json','w'),indent=2)"; }
    $P -u $A/merge_quant_block128.py $2 $3 /home/ubuntu/models/$M --base $BASE > ~/fill_merge_$T.log 2>&1 || { echo "FILL_MERGE_FAILED $T"; return; }
  fi
  echo "FILL_MERGED $T $(ts)"
  serve $T $M || return
  FILL_ONLY=$4 python3 evalq.py fill_$T init; FILL_ONLY=$4 python3 -u evalq.py fill_$T run box1:2 > ~/evalq_fill_$T.log 2>&1
  [ -f $E/consc/${T}_self40_none.json ] || self40 $T
  echo "FILL_DONE $T $(ts) failed=$(grep -c FAILED ~/evalq_fill_$T.log)"
  dsgrade $T
  systemctl --user stop lamba-infer; sleep 15
  [ "$T" != "v40t2" ] && rm -r /home/ubuntu/models/$M
}
# p3 finished: delete its merged to free disk (adapter kept), grade p3 extras
rm -r /home/ubuntu/models/lamba04v40t11p3_merged_fp8b128 2>/dev/null
for T in v40t11p3 v40t11; do dsgrade $T; done
fill v40t2 /home/ubuntu/adapters_lamba_04_v40t2 /home/ubuntu/models/lmh_lamba_04_v40t2
fill v40t11 /home/ubuntu/adapters_lamba_04_v40t11 /home/ubuntu/models/lmh_lamba_04_v40t11
fill v40t7 /home/ubuntu/adapters_lamba_04_v40t7 /home/ubuntu/models/lmh_lamba_04_v40t7
fill v40t9 /home/ubuntu/adapters_lamba_04_v40t9 /home/ubuntu/models/lmh_lamba_04_v40t9 consc,warmth,strongreject,msfact_ho,ibu_ho,xtuapa_ho,prayer_ho
fill v40t10c /home/ubuntu/adapters_lamba_04_v40t10_c100 /home/ubuntu/models/lmh_lamba_04_v40t10_c100 consc,warmth,strongreject,msfact_ho,ibu_ho,xtuapa_ho,prayer_ho
$P $A/ms_drift_eval.py $E/v40q/identity.jsonl $E/v40r/identity.jsonl $E/v40t5/identity.jsonl > $E/v40t5/msdrift.txt 2>&1
wait
bash ~/lamba04_report_data/collect.sh > ~/collect_final.log 2>&1; bash ~/s3_push_04.sh > ~/s3_push_final.log 2>&1
echo "FILL_ALL_DONE $(ts)"
