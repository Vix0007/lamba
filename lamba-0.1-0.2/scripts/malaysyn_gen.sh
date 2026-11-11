#!/bin/bash
# v9c rollouts on synthetic Malay MCQ prompts (resumable; rerun after re-building malay_synth_prompts.jsonl with more batches to pick up new ids). log ~/malaysyn_gen.log
export PATH="$HOME/venv-vllm/bin:$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
~/venv-vllm/bin/python ~/antidoom_data/loop_test_g.py --model /home/ubuntu/models/lmh_v9c_final --label v9cfinal --tag malaysyn --prompts ~/antidoom_data/malay_synth_prompts.jsonl --max-tokens 4000 --chunk 119 --lora ~/adapters_v9c_final >> ~/malaysyn_gen.log 2>&1
echo "MALAYSYN_PASS DONE $(wc -l < ~/antidoom_data/malay_synth_prompts.jsonl)" >> ~/malaysyn_gen.log
