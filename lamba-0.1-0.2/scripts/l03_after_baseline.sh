#!/usr/bin/env bash
set -u; cd /home/ubuntu/antidoom_data
while systemctl --user is-active --quiet lamba-l03base-dpov2b || systemctl --user is-active --quiet lamba-l03toolv2; do sleep 20; done
echo "baselines finished $(date -u +%H:%M)"
./l03_mine_chain.sh dpov2 dpov2
echo "MINING_DONE $(date -u +%H:%M)"
