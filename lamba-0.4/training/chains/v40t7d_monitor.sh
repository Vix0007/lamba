#!/usr/bin/env bash
# events: evalq v40t7s job done/failed, V40T7S markers, unit end, tracebacks, disk low
cd ~/antidoom_data; S=~/.v40t7d_mon_seen; touch $S
while true; do
  st=$( (timeout 90 python3 evalq.py v40t7s status; timeout 90 python3 evalq.py v40t7d status | sed "s/^/d:/") 2>/dev/null)
  echo "$st" | awk 'tolower($2)=="done"||tolower($2)=="failed"{print $1, $2}' | while read j s; do
    grep -qx "Q $j $s" $S || { echo "Q $j $s" >> $S; echo "$(TZ=Asia/Singapore date +%H:%M) EVALQ $j $s ($(echo "$st" | awk 'tolower($2)=="done"' | wc -l) done)"; }; done
  grep -hE "^(V40T7S_|T7D_|V40T8)" ~/v40t7.log ~/v40t7d.log ~/v40t8.log ~/v40t8_prep.log 2>/dev/null | while read l; do grep -qxF "M $l" $S || { echo "M $l" >> $S; echo "$(TZ=Asia/Singapore date +%H:%M) $l"; }; done
  if ! systemctl --user is-active --quiet lamba-v40t7d && ! grep -qx "UNIT_ENDED" $S; then echo UNIT_ENDED >> $S; echo "$(TZ=Asia/Singapore date +%H:%M) lamba-v40t7d chain ENDED"; fi
  n=$(cat ~/evalq_v40t7s_box1.log ~/evalq_v40t7d_box1.log ~/v40t7d.log 2>/dev/null | grep -c Traceback); grep -qx "TB $n" $S || { [ "${n:-0}" -gt 0 ] && { sed -i '/^TB /d' $S; echo "TB $n" >> $S; echo "$(TZ=Asia/Singapore date +%H:%M) TRACEBACK in evalq log (count $n)"; }; }
  d=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); [ "$d" -lt 25 ] && echo "$(TZ=Asia/Singapore date +%H:%M) DISK_LOW ${d}G"
  sleep 120
done
