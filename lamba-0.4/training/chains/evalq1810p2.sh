#!/usr/bin/env bash
exec >> ~/evalq_l1810p2.log 2>&1
export PATH=/home/ubuntu/venv-vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
until ! systemctl --user is-active -q lamba-train1810p2 && ! systemctl --user is-active -q lamba-p2safety; do sleep 30; done
curl -s -m 5 127.0.0.1:8001/v1/models | grep -q l1810p2 || { echo "EVALQ_NO_SERVER"; exit 1; }
cd /home/ubuntu/antidoom_data; echo "EVALQ_START $(date -u +%H:%M)"
python3 -u evalq.py l1810p2 run box1:2; echo "EVALQ_L1810P2_DONE $(date -u +%H:%M)"
python3 evalq.py l1810p2 status
