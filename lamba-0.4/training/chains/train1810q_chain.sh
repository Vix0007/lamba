#!/usr/bin/env bash
# Lamba0.4-1810q ONE-SHOT chain: waits ~/.go1810q -> full train -> verify -> mid ckpt -> convert/merge FP8 b128 (rp1.05) -> serve merged+MTP -> evalq l1810q (no GPQA/MalayMMLU) -> batteries. log ~/train1810q.log markers T1810Q_*
exec >> ~/train1810q.log 2>&1
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/home/ubuntu/venv-vllm/bin:$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/home/ubuntu/antidoom/.venv/bin/python; A=/home/ubuntu/antidoom_data; BASE=/home/ubuntu/models/cand04_bf16_sh; RUN=lamba_04_1810q; TL=~/lamba04_1810q_train.log
DATA=$A/l03_training_data_1810q.jsonl; D=/home/ubuntu/eval_results/l1810q; mkdir -p $D
until [ -f ~/.go1810q ]; do sleep 30; done
echo "T1810Q_GO $(date -u +%H:%M) rows $(wc -l < $DATA)"
systemctl --user stop lamba-infer 2>/dev/null; sleep 5
F=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9); [ "$F" -ge 60 ] || { echo "T1810Q_DISK_LOW ${F}G"; exit 1; }
cd ~/antidoom; rm -rf runs/$RUN; rm -f $TL; echo "T1810Q_TRAIN_START $(date -u +%H:%M)"
for ATT in 1 2 3; do
  echo "T1810Q_ATTEMPT $ATT $(date -u +%H:%M)"
  uv run antidoom -c configs/lamba_04_1810q.yaml -r runs/$RUN --model-name $BASE train --dataset-jsonl $DATA >> $TL 2>&1 && break
  grep -q "REFUSING this checkpoint" $TL && break
  echo "T1810Q_ATTEMPT_FAILED $ATT (resuming from last complete checkpoint)"; sleep 20
done
grep -q 'REFUSING this checkpoint' $TL && { echo T1810Q_DISKGUARD; exit 1; }
EP=$(grep -o "'train_runtime'[^}]*'epoch': '[0-9.]*'" $TL | tail -1 | grep -o "'epoch': '[0-9.]*'" | grep -oE '[0-9.]+' | tail -1)
python3 -c "import sys; sys.exit(0 if float('${EP:-0}') >= 0.95 else 1)" || { echo "T1810Q_TRAIN_INCOMPLETE epoch=$EP"; tail -5 $TL; exit 1; }
echo "T1810Q_TRAIN_OK epoch=$EP $(date -u +%H:%M)"
TOT=$(ls -d runs/$RUN/lora/checkpoint-* | sed 's/.*-//' | sort -n | tail -1); MID=$(ls -d runs/$RUN/lora/checkpoint-* | sed 's/.*-//' | sort -n | awk -v t=$TOT 'BEGIN{b=1e9}{d=$1-t/2; if(d<0)d=-d; if(d<b){b=d;m=$1}}END{print m}')
mkdir -p runs/$RUN/mid_ckpt; cp runs/$RUN/lora/checkpoint-$MID/adapter_* runs/$RUN/mid_ckpt/ && echo "T1810Q_MID_KEPT step $MID of ~$TOT"
for c in runs/$RUN/lora/checkpoint-*; do [ -d "$c" ] && rm -r "$c"; done
$P $A/convert_adapter_for_vllm.py runs/$RUN/lora /home/ubuntu/adapters_$RUN || { echo T1810Q_CONVERT_FAIL; exit 1; }
$P $A/make_lmh_model.py /home/ubuntu/adapters_$RUN/lm_head_lora.safetensors /home/ubuntu/models/lmh_$RUN --base $BASE --scale 1.0 || { echo T1810Q_LMH_FAIL; exit 1; }
M=/home/ubuntu/models/lamba04_1810q_merged_fp8b128
$P $A/merge_quant_block128.py /home/ubuntu/adapters_$RUN /home/ubuntu/models/lmh_$RUN $M --base $BASE || { echo T1810Q_MERGE_FAIL; exit 1; }
$P - <<PY
import json,os; f="$M/generation_config.json"
g=json.load(open(f)) if os.path.exists(f) else {}
if os.path.islink(f): os.unlink(f)
g["repetition_penalty"]=1.05; json.dump(g,open(f,"w"),indent=1); print("gen_config",g)
PY
echo "T1810Q_MERGED $(date -u +%H:%M)"
systemd-run --user --unit=lamba-infer --collect -p MemoryMax=27G -p MemorySwapMax=0 -E PATH=$PATH -E VLLM_USE_FLASHINFER_SAMPLER=0 \
  /home/ubuntu/venv-vllm/bin/vllm serve $M --served-model-name l1810q --generation-config auto --trust-remote-code --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 32 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --host 127.0.0.1 --port 8001 >/dev/null
until curl -s -m 2 127.0.0.1:8001/v1/models >/dev/null || ! systemctl --user is-active -q lamba-infer; do sleep 5; done
systemctl --user is-active -q lamba-infer || { echo T1810Q_SERVE_FAIL; exit 1; }
echo "T1810Q_SERVED $(date -u +%H:%M)"
cd $A; G=/home/ubuntu/eval_results/diag05
python3 evalq.py l1810q init; python3 -u evalq.py l1810q run box1:3; echo "T1810Q_QUEUE_DONE $(date -u +%H:%M)"
python3 diag05_battery.py l1810q $G/l1810q.jsonl --n 4 && python3 diag05_score.py l1810p2=$G/l1810p2.jsonl l1810q=$G/l1810q.jsonl > $D/name_battery.txt 2>&1; echo "T1810Q_NAMES_DONE"
python3 diag05_emptyreplay.py l1810q $G/b2_v40t11p3.jsonl $G/emptyreplay_l1810q.jsonl --n 10; echo "T1810Q_EMPTY_DONE"
python3 tone_battery.py l1810q /home/ubuntu/eval_results/tone1810/l1810q.jsonl; echo "T1810Q_TONE_DONE"
echo "T1810Q_ALL_DONE $(date -u +%H:%M)"
