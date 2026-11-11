#!/usr/bin/env bash
# Lamba0.4-1810 chain: waits ~/.go1810 (set after data audit) -> smoke -> full train -> keep mid ckpt adapter -> convert/merge FP8 b128 (rp1.05) -> serve merged+MTP -> evals. log ~/train1810.log markers T1810P_*
exec >> ~/train1810p.log 2>&1
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/home/ubuntu/venv-vllm/bin:$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/home/ubuntu/antidoom/.venv/bin/python; A=/home/ubuntu/antidoom_data; BASE=/home/ubuntu/models/cand04_bf16_sh; RUN=lamba_04_1810p; TL=~/lamba04_1810p_train.log
DATA=$A/l03_training_data_1810p.jsonl; D=/home/ubuntu/eval_results/l1810p; mkdir -p $D
until ! systemctl --user is-active -q lamba-train1810; do sleep 30; done
echo "T1810P_GO $(date -u +%H:%M) rows $(wc -l < $DATA)"
systemctl --user stop lamba-infer 2>/dev/null; sleep 5
F=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9); [ "$F" -ge 45 ] || { echo "T1810P_DISK_LOW ${F}G"; exit 1; }
cd ~/antidoom; rm -rf runs/${RUN}_smoke; echo "T1810P_SMOKE_OK (24 steps of the first smoke attempt ran clean; smoke stage removed)"
rm -rf runs/$RUN; echo "T1810P_TRAIN_START $(date -u +%H:%M)"
uv run antidoom -c configs/lamba_04_1810p.yaml -r runs/$RUN --model-name $BASE train --dataset-jsonl $DATA > $TL 2>&1
grep -q 'REFUSING this checkpoint' $TL && { echo T1810P_DISKGUARD; exit 1; }
EP=$(grep -o "'train_runtime'[^}]*'epoch': '[0-9.]*'" $TL | tail -1 | grep -o "'epoch': '[0-9.]*'" | grep -oE '[0-9.]+' | tail -1)
python3 -c "import sys; sys.exit(0 if float('${EP:-0}') >= 0.95 else 1)" || { echo "T1810P_TRAIN_INCOMPLETE epoch=$EP"; tail -5 $TL; exit 1; }
echo "T1810P_TRAIN_OK epoch=$EP $(date -u +%H:%M)"
# keep the checkpoint closest to mid-run (adapter only) for comparison; drop the rest
TOT=$(ls -d runs/$RUN/lora/checkpoint-* | sed 's/.*-//' | sort -n | tail -1); MID=$(ls -d runs/$RUN/lora/checkpoint-* | sed 's/.*-//' | sort -n | awk -v t=$TOT 'BEGIN{b=1e9}{d=$1-t/2; if(d<0)d=-d; if(d<b){b=d;m=$1}}END{print m}')
mkdir -p runs/$RUN/mid_ckpt; cp runs/$RUN/lora/checkpoint-$MID/adapter_* runs/$RUN/mid_ckpt/ && echo "T1810P_MID_KEPT step $MID of ~$TOT"
for c in runs/$RUN/lora/checkpoint-*; do [ -d "$c" ] && rm -r "$c"; done
$P $A/convert_adapter_for_vllm.py runs/$RUN/lora /home/ubuntu/adapters_$RUN || { echo T1810P_CONVERT_FAIL; exit 1; }
$P $A/make_lmh_model.py /home/ubuntu/adapters_$RUN/lm_head_lora.safetensors /home/ubuntu/models/lmh_$RUN --base $BASE --scale 1.0 || { echo T1810P_LMH_FAIL; exit 1; }
M=/home/ubuntu/models/lamba04_1810p_merged_fp8b128
$P $A/merge_quant_block128.py /home/ubuntu/adapters_$RUN /home/ubuntu/models/lmh_$RUN $M --base $BASE || { echo T1810P_MERGE_FAIL; exit 1; }
$P - <<PY
import json,os; f="$M/generation_config.json"
g=json.load(open(f)) if os.path.exists(f) and not os.path.islink(f) else (json.load(open(f)) if os.path.exists(f) else {})
if os.path.islink(f): os.unlink(f)
g["repetition_penalty"]=1.05; json.dump(g,open(f,"w"),indent=1); print("gen_config",g)
PY
echo "T1810P_MERGED $(date -u +%H:%M)"
systemd-run --user --unit=lamba-infer --collect -p MemoryMax=27G -p MemorySwapMax=0 -E PATH=$PATH -E VLLM_USE_FLASHINFER_SAMPLER=0 \
  /home/ubuntu/venv-vllm/bin/vllm serve $M --served-model-name l1810p --generation-config auto --trust-remote-code --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 32 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --host 127.0.0.1 --port 8001 >/dev/null
until curl -s -m 2 127.0.0.1:8001/v1/models >/dev/null || ! systemctl --user is-active -q lamba-infer; do sleep 5; done
systemctl --user is-active -q lamba-infer || { echo T1810P_SERVE_FAIL; exit 1; }
echo "T1810P_SERVED $(date -u +%H:%M)"
cd $A; G=/home/ubuntu/eval_results/diag05
python3 diag05_battery.py l1810p $G/l1810p.jsonl --n 4 && python3 diag05_score.py p3=$G/v40t11p3.jsonl v40sc=$G/v40sc.jsonl l1810=$G/l1810.jsonl l1810p=$G/l1810p.jsonl > $D/name_battery.txt 2>&1; echo "T1810P_NAME_DONE"
python3 diag05_emptyreplay.py l1810p $G/b2_v40t11p3.jsonl $G/emptyreplay_l1810p.jsonl --n 10; echo "T1810P_EMPTY_DONE"
python3 l03_identity_eval.py l1810p $D/identity.jsonl > $D/identity.log 2>&1; echo "T1810P_IDENTITY_DONE"
python3 l03_r3_defect_eval.py l1810p $D/defect.json > $D/defect.log 2>&1; echo "T1810P_DEFECT_DONE"
python3 tone_battery.py l1810p /home/ubuntu/eval_results/tone1810/l1810p.jsonl; echo "T1810P_TONE_DONE"
python3 loop_test_server.py l1810p /home/ubuntu/eval_results/loop_l1810p_g150_rp105.jsonl --rp 1.05; echo "T1810P_LOOP_DONE"
echo "T1810P_ALL_DONE $(date -u +%H:%M)"
