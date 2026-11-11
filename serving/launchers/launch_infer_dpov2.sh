#!/usr/bin/env bash
# Persistent OpenAI-compatible inference backend for the FINAL deliverable:
# dpov2 (DPO on top of c10 recipe, non-inferior to c10 at rp1.1), served with its bundled
# repetition_penalty=1.1 generation_config default. Internal-only (127.0.0.1).
set -euo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0
cd /home/ubuntu

exec /home/ubuntu/venv-vllm/bin/vllm serve /home/ubuntu/models/lmh_ruminconfab_dpo_v2 \
  --served-model-name dpov2 \
  --generation-config auto \
  --quantization fp8 \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 55296 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --enable-lora \
  --max-lora-rank 256 \
  --max-loras 1 \
  --lora-modules dpov2=/home/ubuntu/adapters_ruminconfab_dpo_v2 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --host 127.0.0.1 \
  --port 8001
