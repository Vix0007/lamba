#!/usr/bin/env bash
# ordered: identity (running) -> distill -> tool states. log ~/mine1810.log
exec >> ~/mine1810.log 2>&1
up() { until curl -s -m 2 127.0.0.1:8001/v1/models | grep -q '"base"'; do sleep 5; done; }
echo "RESTART $(date -u +%H:%M)"
up; python3 ~/antidoom_data/mine_distill_1810.py base ~/eval_results/id1810/distill_base.jsonl; echo "DISTILL_ENDED $(date -u +%H:%M)"
up; ~/antidoom/.venv/bin/python ~/antidoom_data/l03_mine_native_1810.py base ~/antidoom_data/td/td_states_1810.jsonl ~/eval_results/id1810/td_base.jsonl --k 4 --temp 0.8; echo "TDMINE_ENDED $(date -u +%H:%M)"
echo MINE1810_ALL_DONE
