#!/usr/bin/env bash
# Lamba 0.4 v40s MERGED (LoRA folded into candidate BF16, static FP8 block128): no LoRA, no spec (vLLM MTP broken on block-FP8).
set -euo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0
exec /home/ubuntu/venv-vllm/bin/vllm serve /home/ubuntu/models/lamba04v40t11p_merged_fp8b128 \
  --served-model-name lamba04v40t11p --trust-remote-code --dtype bfloat16 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 64 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64]}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice --enable-prefix-caching --mamba-cache-mode align \
  --host 127.0.0.1 --port 8001
