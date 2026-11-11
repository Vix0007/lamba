#!/usr/bin/env bash
exec >> ~/train1810p.log 2>&1
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/home/ubuntu/venv-vllm/bin:$PATH
until ! systemctl --user is-active -q lamba-train1810p; do sleep 30; done
systemctl --user stop lamba-infer 2>/dev/null; sleep 5
systemd-run --user --unit=lamba-infer --collect -p MemoryMax=27G -p MemorySwapMax=0 -E PATH=$PATH -E VLLM_USE_FLASHINFER_SAMPLER=0 \
  /home/ubuntu/venv-vllm/bin/vllm serve /home/ubuntu/models/lamba04_1810_merged_fp8b128 --served-model-name l1810 --generation-config auto --trust-remote-code --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 32 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3 --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --host 127.0.0.1 --port 8001 >/dev/null
until curl -s -m 2 127.0.0.1:8001/v1/models >/dev/null || ! systemctl --user is-active -q lamba-infer; do sleep 5; done
python3 /home/ubuntu/antidoom_data/loop_test_server.py l1810 /home/ubuntu/eval_results/loop_l1810_g150_rp105.jsonl --rp 1.05; echo "L1810_LOOP_DONE $(date -u +%H:%M)"
systemctl --user stop lamba-infer
