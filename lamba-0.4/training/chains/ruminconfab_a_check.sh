#!/bin/bash
set -e
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=$HOME/antidoom_data
LOG=$HOME/ruminconfab_a_check.log

wait_healthy() {
  for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8001/v1/models > /dev/null 2>&1; then
      echo "backend healthy after ${i}0s" >> $LOG
      return 0
    fi
    sleep 10
  done
  echo "BACKEND NEVER CAME UP" >> $LOG
  return 1
}

echo "=== A-CHECK: launching c10 backend ===" >> $LOG
systemd-run --user --unit=lamba-infer --collect -p MemoryMax=30G -p MemorySwapMax=0 \
  bash -c "export VLLM_USE_FLASHINFER_SAMPLER=0; exec $HOME/venv-vllm/bin/vllm serve $HOME/models/lmh_letterfix2_c10 \
    --served-model-name letterfix2-c10 --generation-config auto --quantization fp8 --dtype bfloat16 \
    --trust-remote-code --max-model-len 55296 --gpu-memory-utilization 0.90 --max-num-seqs 8 \
    --enable-lora --max-lora-rank 256 --max-loras 1 \
    --lora-modules letterfix2-c10=$HOME/adapters_letterfix2_c10 \
    --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[1,2,4,8,16,32]}' \
    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
    --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
    --host 127.0.0.1 --port 8001" >> $LOG 2>&1
wait_healthy
echo "=== A-CHECK: c10 tool-loop gen ===" >> $LOG
$HOME/venv-vllm/bin/python $A/gen_a_fresh_toolloop.py letterfix2-c10 c10 $HOME/eval_results/a_fresh_c10.jsonl >> $LOG 2>&1
systemctl --user stop lamba-infer >> $LOG 2>&1
sleep 5

echo "=== A-CHECK: launching ruminconfab s1337 backend ===" >> $LOG
systemd-run --user --unit=lamba-infer --collect -p MemoryMax=30G -p MemorySwapMax=0 \
  bash -c "export VLLM_USE_FLASHINFER_SAMPLER=0; exec $HOME/venv-vllm/bin/vllm serve $HOME/models/lmh_ruminconfab_s1337 \
    --served-model-name ruminconfab-s1337 --generation-config auto --quantization fp8 --dtype bfloat16 \
    --trust-remote-code --max-model-len 55296 --gpu-memory-utilization 0.90 --max-num-seqs 8 \
    --enable-lora --max-lora-rank 256 --max-loras 1 \
    --lora-modules ruminconfab-s1337=$HOME/adapters_ruminconfab_s1337 \
    --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[1,2,4,8,16,32]}' \
    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
    --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
    --host 127.0.0.1 --port 8001" >> $LOG 2>&1
wait_healthy
echo "=== A-CHECK: ruminconfab s1337 tool-loop gen ===" >> $LOG
$HOME/venv-vllm/bin/python $A/gen_a_fresh_toolloop.py ruminconfab-s1337 ruminconfab $HOME/eval_results/a_fresh_ruminconfab.jsonl >> $LOG 2>&1
systemctl --user stop lamba-infer >> $LOG 2>&1

echo "A_CHECK_ALL_DONE" >> $LOG
