#!/usr/bin/env bash
exec >> ~/train1810p.log 2>&1
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/home/ubuntu/venv-vllm/bin:$PATH
until ! systemctl --user is-active -q lamba-loop1810; do sleep 30; done
systemctl --user stop lamba-infer 2>/dev/null; sleep 5
systemd-run --user --unit=lamba-infer --collect -p MemoryMax=27G -p MemorySwapMax=0 -E PATH=$PATH -E VLLM_USE_FLASHINFER_SAMPLER=0 \
  /home/ubuntu/venv-vllm/bin/vllm serve /home/ubuntu/models/lamba04_1810p_merged_fp8b128 --served-model-name l1810p --generation-config auto --trust-remote-code --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 32 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --host 127.0.0.1 --port 8001 >/dev/null
until curl -s -m 2 127.0.0.1:8001/v1/models >/dev/null || ! systemctl --user is-active -q lamba-infer; do sleep 5; done
echo "SHIP_SERVED $(date -u +%H:%M)"; cd /home/ubuntu/antidoom_data; E=/home/ubuntu/eval_results
python3 -u l03_tool_eval.py l1810p $E/l1810p/toolv2.jsonl --split heldout --prompts l03_tool_prompts_v2.jsonl --temp 1.0 --top-p 0.95 > $E/l1810p/tool.log 2>&1; echo "SHIP_TOOL_DONE"
python3 -u strongreject_probe.py l1810p $E/matrix/l1810p_merged_strongreject.jsonl > $E/l1810p/sr.log 2>&1; echo "SHIP_SR_DONE"
python3 -u xstest_eval.py l1810p $E/matrix/l1810p_merged_xstest.jsonl > $E/l1810p/xstest.log 2>&1; echo "SHIP_XSTEST_DONE $(date -u +%H:%M)"
