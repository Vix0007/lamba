#!/usr/bin/env bash
# events: evalq v40t7s job done/failed, V40T7S markers, unit end, tracebacks, disk low
cd ~/antidoom_data; S=~/.v40t7s_mon_seen; touch $S
while true; do
  st=$(timeout 90 python3 evalq.py v40t7s status 2>/dev/null)
  echo "$st" | awk 'tolower($2)=="done"||tolower($2)=="failed"{print $1, $2}' | while read j s; do
    grep -qx "Q $j $s" $S || { echo "Q $j $s" >> $S; echo "$(TZ=Asia/Singapore date +%H:%M) EVALQ $j $s ($(echo "$st" | awk 'tolower($2)=="done"' | wc -l) done)"; }; done
  grep -E "^V40T7S_" ~/v40t7.log | while read l; do grep -qxF "M $l" $S || { echo "M $l" >> $S; echo "$(TZ=Asia/Singapore date +%H:%M) $l"; }; done
  if ! systemctl --user is-active --quiet lamba-v40t7s && ! grep -qx "UNIT_ENDED" $S; then echo UNIT_ENDED >> $S; echo "$(TZ=Asia/Singapore date +%H:%M) lamba-v40t7s chain ENDED"; fi
  n=$(grep -c Traceback ~/evalq_v40t7s_box1.log 2>/dev/null); grep -qx "TB $n" $S || { [ "${n:-0}" -gt 0 ] && { sed -i '/^TB /d' $S; echo "TB $n" >> $S; echo "$(TZ=Asia/Singapore date +%H:%M) TRACEBACK in evalq log (count $n)"; }; }
  d=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); [ "$d" -lt 25 ] && echo "$(TZ=Asia/Singapore date +%H:%M) DISK_LOW ${d}G"
  sleep 120
done
