#!/usr/bin/env bash
S=~/.v40t7_mon_seen; touch $S
emit(){ grep -qxF "$1" $S || { echo "$1" >> $S; echo "$(TZ=Asia/Singapore date +%H:%M) $1"; }; }
while true; do
  for f in ~/v40t7.log ~/s3_push_04.log; do [ -f $f ] && grep -aoE "^(PF_|BOX1_|BOXB_|S3_PUSH|V40T7S?|MINE)[A-Za-z0-9_.]*.*" $f | cut -c1-80 | while read -r l; do emit "$l"; done; done
  if [ -f ~/lamba04_v40t7_train.log ]; then
    st=$(tr '\r' '\n' < ~/lamba04_v40t7_train.log | grep -a "s/it" | grep -av "Loading" | grep -aoE "[0-9]+/[0-9]{2,3} \[" | tail -1 | cut -d/ -f1)
    [ -n "$st" ] && [ $((st / 40 * 40)) -gt 0 ] && emit "TRAIN step $((st / 40 * 40)) reached"
    grep -aqE "Traceback|CUDA out of memory|REFUSING" ~/lamba04_v40t7_train.log && emit "TRAIN_ERROR in train log"
  fi
  [ -f ~/s3_push_04.log ] && grep -qiE "error|fatal" ~/s3_push_04.log && emit "S3_PUSH_ERROR (see ~/s3_push_04.log)"
  for u in lamba-v40t7 lamba-v40t7s; do r=$(systemctl --user show $u -p Result --value 2>/dev/null); [ -n "$r" ] && [ "$r" != success ] && emit "UNIT_FAILED $u $r"; done
  free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$free" -lt 25 ] && emit "DISK_LOW ${free}G"
  sleep 30
done
