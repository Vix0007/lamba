#!/bin/bash
set -e
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$HOME/s42_eval_chain.log
A=$HOME/antidoom_data
E=$HOME/eval_results

wait_flag() {
  until grep -q "$1" $HOME/ruminconfab_train.log 2>/dev/null; do sleep 15; done
}
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
serve_and_wait() {
  # $1 = model dir, $2 = served-model-name, $3 = lora dir
  systemd-run --user --unit=lamba-infer --collect -p MemoryMax=30G -p MemorySwapMax=0 \
    bash -c "export VLLM_USE_FLASHINFER_SAMPLER=0; exec $HOME/venv-vllm/bin/vllm serve $1 \
      --served-model-name $2 --generation-config auto --quantization fp8 --dtype bfloat16 \
      --trust-remote-code --max-model-len 55296 --gpu-memory-utilization 0.90 --max-num-seqs 8 \
      --enable-lora --max-lora-rank 256 --max-loras 1 \
      --lora-modules $2=$3 \
      --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[1,2,4,8,16,32]}' \
      --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
      --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
      --host 127.0.0.1 --port 8001" >> $LOG 2>&1
  wait_healthy
}

echo "=== S42 EVAL CHAIN: waiting for RUMINCONFAB_S42 FINISHED OK ===" >> $LOG
wait_flag "RUMINCONFAB_S42 FINISHED OK"

if [ ! -f $HOME/models/lmh_ruminconfab_s42/model-00018-of-00018.safetensors ]; then
  echo "=== S42: converting adapter ===" >> $LOG
  cd $HOME/antidoom
  $HOME/antidoom/.venv/bin/python $A/convert_adapter_for_vllm.py \
    runs/lamba_ruminconfab_s42/lora $HOME/adapters_ruminconfab_s42 >> $LOG 2>&1
  $HOME/antidoom/.venv/bin/python $A/make_lmh_model.py \
    $HOME/adapters_ruminconfab_s42/lm_head_lora.safetensors $HOME/models/lmh_ruminconfab_s42 \
    --scale 1.0 >> $LOG 2>&1
else
  echo "=== S42: adapter already converted+merged, skipping ===" >> $LOG
fi

echo "=== S42: fresh c10 re-run on 47-prompt pool (collapse is stochastic, DS wants a fresh run) ===" >> $LOG
rm -f $E/a_pool_c10_fresh.jsonl
serve_and_wait $HOME/models/lmh_letterfix2_c10 letterfix2-c10 $HOME/adapters_letterfix2_c10
$HOME/venv-vllm/bin/python $A/gen_a_fresh_toolloop.py letterfix2-c10 c10_fresh \
  $E/a_pool_c10_fresh.jsonl $A/a_check_pool.jsonl >> $LOG 2>&1
systemctl --user stop lamba-infer >> $LOG 2>&1
echo "S42_C10_FRESH_DONE" >> $LOG
sleep 5

echo "=== S42: seed42 adapter run on 47-prompt pool ===" >> $LOG
rm -f $E/a_pool_s42.jsonl
serve_and_wait $HOME/models/lmh_ruminconfab_s42 ruminconfab-s42 $HOME/adapters_ruminconfab_s42
$HOME/venv-vllm/bin/python $A/gen_a_fresh_toolloop.py ruminconfab-s42 s42 \
  $E/a_pool_s42.jsonl $A/a_check_pool.jsonl >> $LOG 2>&1
systemctl --user stop lamba-infer >> $LOG 2>&1
echo "S42_POOL_DONE" >> $LOG
sleep 5

echo "=== S42: reproducibility check (5 known collapse/discordant prompts x 5 reps) -- c10 arm ===" >> $LOG
rm -f $E/a_repro_c10.jsonl
serve_and_wait $HOME/models/lmh_letterfix2_c10 letterfix2-c10 $HOME/adapters_letterfix2_c10
$HOME/venv-vllm/bin/python $A/gen_a_repro_check.py letterfix2-c10 $E/a_repro_c10.jsonl >> $LOG 2>&1
systemctl --user stop lamba-infer >> $LOG 2>&1
sleep 5

echo "=== S42: reproducibility check -- s42 arm ===" >> $LOG
rm -f $E/a_repro_s42.jsonl
serve_and_wait $HOME/models/lmh_ruminconfab_s42 ruminconfab-s42 $HOME/adapters_ruminconfab_s42
$HOME/venv-vllm/bin/python $A/gen_a_repro_check.py ruminconfab-s42 $E/a_repro_s42.jsonl >> $LOG 2>&1
systemctl --user stop lamba-infer >> $LOG 2>&1

echo "S42_EVAL_CHAIN_ALL_DONE" >> $LOG
