#!/bin/bash
# DS 72: after base586 -> adapter on the 586 clean prompts ONLY (GPQA/letters cut from this cycle). log ~/v8_validate.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
A=~/antidoom_data; VPY=~/venv-vllm/bin/python; LMH=/home/ubuntu/models/lmh_v8_final; AD=~/adapters_v8_final
echo "$(date -u +%FT%TZ) START adapter586 (validate2)" >> ~/v8_validate.log
$VPY $A/loop_test_g.py --model $LMH --label v8final --tag ev586 --prompts $A/evalpool_clean.jsonl --max-tokens 4000 --chunk 128 --lora $AD >> ~/v8_validate.log 2>&1
echo "$(date -u +%FT%TZ) END adapter586 rc=$? ; V8 VALIDATE2 DONE" >> ~/v8_validate.log
