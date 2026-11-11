#!/usr/bin/env bash
# Eval-only serving for the effreason pilot adapter (loop_span efficient-reasoning round on
# top of letterfix2-c10). Port 8002 so it doesn't collide with the main letterfix2-c10 server.
set -euo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0
cd /home/ubuntu

exec /home/ubuntu/venv-vllm/bin/vllm serve /home/ubuntu/models/lmh_effreason_final \
  --served-model-name effreason-final \
  --quantization fp8 \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --enable-lora \
  --max-lora-rank 256 \
  --max-loras 1 \
  --lora-modules effreason-final=/home/ubuntu/adapters_effreason_final \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
  --reasoning-parser qwen3 \
  --host 127.0.0.1 \
  --port 8002
