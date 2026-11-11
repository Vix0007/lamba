#!/usr/bin/env bash
exec >> ~/evalq_l1810p2.log 2>&1
export PATH=/home/ubuntu/venv-vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
until ! systemctl --user is-active -q lamba-evalq1810p2; do sleep 30; done
cd /home/ubuntu/antidoom_data; echo "EVALQ_RETRY_START $(date -u +%H:%M)"
python3 evalq.py l1810p2 release; python3 -u evalq.py l1810p2 run box1:2; echo "EVALQ_RETRY_DONE $(date -u +%H:%M)"; python3 evalq.py l1810p2 status
