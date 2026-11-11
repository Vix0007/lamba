#!/usr/bin/env bash
# v40t10: waits for anti-denial mining + v40t7f scorecard -> build -> train from v40t2 -> STOP at checkpoint-100 (~1 epoch) ->
# quick checkpoint evals (queue v40t10c: prayer_ho/consc/tool/warmth/xtuapa/identity) -> resume to epoch 2 -> merge -> serve -> evalq v40t10.
exec >> ~/v40t10.log 2>&1
P=/home/ubuntu/antidoom/.venv/bin/python; A=/home/ubuntu/antidoom_data; BASE=/home/ubuntu/models/cand04_bf16_sh; RUN=lamba_04_v40t10; TL=~/lamba04_v40t10_train.log
ts(){ TZ=Asia/Singapore date +%T; }
echo "V40T10_WAIT $(ts)"
until grep -q "^T10_MINE_DONE" ~/t10_mine.log 2>/dev/null; do sleep 30; done
until [ -f ~/.t7f_handoff_done ]; do sleep 30; done
echo "V40T10_BUILD $(ts)"; cd $A; [ -s l03_training_data_v40t10.jsonl ] || $P build_v40t10.py > ~/v40t10_build.txt 2>&1 || { echo V40T10_BUILD_FAILED; exit 1; }; tail -1 ~/v40t10_build.txt; grep NODENY ~/v40t10_build.txt
systemctl --user stop lamba-infer 2>/dev/null; sleep 15; until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 10; done
[ -d /home/ubuntu/models/lamba04v40t7_merged_fp8b128 ] && rm -r /home/ubuntu/models/lamba04v40t7_merged_fp8b128   # rebuildable
[ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -ge 60 ] || { echo "V40T10_DISK_TOO_LOW"; exit 1; }
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd ~/antidoom
rm -rf runs/$RUN
export P A BASE RUN TL
train(){ uv run antidoom -c configs/lamba_04_v40t10.yaml -r runs/$RUN --model-name $BASE train --dataset-jsonl $A/l03_training_data_v40t10.jsonl >> $TL 2>&1; }
echo "V40T10_TRAIN_START $(ts)"
setsid bash -c "$(declare -f train); train" & TP=$!
until [ -f runs/$RUN/lora/checkpoint-100/COMPLETE ] || ! kill -0 $TP 2>/dev/null; do sleep 30; done
if kill -0 $TP 2>/dev/null; then
  kill -TERM -- -$TP; sleep 30; kill -KILL -- -$TP 2>/dev/null; echo "V40T10_PAUSED_AT_100 $(ts)"
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 10; done
  C=runs/$RUN/lora/checkpoint-100
  $P $A/convert_adapter_for_vllm.py $C /home/ubuntu/adapters_${RUN}_c100 && \
  $P $A/make_lmh_model.py /home/ubuntu/adapters_${RUN}_c100/lm_head_lora.safetensors /home/ubuntu/models/lmh_${RUN}_c100 --base $BASE --scale 1.0 && \
  rm -f /home/ubuntu/models/lmh_${RUN}_c100/generation_config.json && python3 -c "
import json; g=json.load(open('$BASE/generation_config.json')); g['repetition_penalty']=1.05; json.dump(g,open('/home/ubuntu/models/lmh_${RUN}_c100/generation_config.json','w'),indent=2)" && \
  $P -u $A/merge_quant_block128.py /home/ubuntu/adapters_${RUN}_c100 /home/ubuntu/models/lmh_${RUN}_c100 /home/ubuntu/models/lamba04v40t10c_merged_fp8b128 --base $BASE || { echo V40T10_C100_MERGE_FAILED; }
  if [ -d /home/ubuntu/models/lamba04v40t10c_merged_fp8b128 ]; then
    sed "s#lamba04v40t2_merged_fp8b128#lamba04v40t10c_merged_fp8b128#; s#--served-model-name lamba04v40t2#--served-model-name lamba04v40t10c#" ~/launch_vllm_v40t2_merged.sh > ~/launch_vllm_v40t10c_merged.sh; chmod +x ~/launch_vllm_v40t10c_merged.sh
    systemd-run --user --unit=lamba-infer --collect -p MemoryMax=27G -p MemorySwapMax=0 ~/launch_vllm_v40t10c_merged.sh >/dev/null
    until curl -s -m 5 http://127.0.0.1:8001/v1/models | grep -q lamba04v40t10c; do systemctl --user is-active --quiet lamba-infer || { echo V40T10_C100_SERVE_FAILED; break; }; sleep 15; done
    echo "V40T10_C100_SERVED $(ts)"
    cd $A; python3 evalq.py v40t10c init; python3 -u evalq.py v40t10c run box1:2 > ~/evalq_v40t10c_box1.log 2>&1
    echo "V40T10_C100_EVALS_DONE $(ts) done=$(timeout 90 python3 evalq.py v40t10c status | grep -c DONE)"
    systemctl --user stop lamba-infer; sleep 20; rm -r /home/ubuntu/models/lamba04v40t10c_merged_fp8b128
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 10; done
  fi
  [ -f ~/.v40t10_stop ] && { echo "V40T10_STOPPED_BY_FLAG $(ts)"; exit 0; }
  cd ~/antidoom; echo "V40T10_RESUME $(ts)"; train
fi
wait $TP 2>/dev/null
grep -q 'REFUSING this checkpoint' $TL && { echo "V40T10_TRAIN_DISKGUARD_STOP"; exit 1; }
EP=$(grep -o "'train_runtime'[^}]*'epoch': '[0-9.]*'" $TL | tail -1 | grep -o "'epoch': '[0-9.]*'" | grep -oE '[0-9.]+' | tail -1)
python3 -c "import sys; sys.exit(0 if float('${EP:-0}') >= 1.95 else 1)" || { echo "V40T10_TRAIN_INCOMPLETE epoch=$EP"; exit 1; }
echo "V40T10_TRAIN_OK epoch=$EP $(ts)"
for c in runs/$RUN/lora/checkpoint-*; do rm -r "$c"; done
$P $A/convert_adapter_for_vllm.py runs/$RUN/lora /home/ubuntu/adapters_$RUN || { echo V40T10_CONVERT_FAILED; exit 1; }
$P $A/make_lmh_model.py /home/ubuntu/adapters_$RUN/lm_head_lora.safetensors /home/ubuntu/models/lmh_$RUN --base $BASE --scale 1.0 || { echo V40T10_LMH_FAILED; exit 1; }
rm -f /home/ubuntu/models/lmh_$RUN/generation_config.json; python3 -c "
import json; g=json.load(open('$BASE/generation_config.json')); g['repetition_penalty']=1.05; json.dump(g,open('/home/ubuntu/models/lmh_$RUN/generation_config.json','w'),indent=2)"
$P -u $A/merge_quant_block128.py /home/ubuntu/adapters_$RUN /home/ubuntu/models/lmh_$RUN /home/ubuntu/models/lamba04v40t10_merged_fp8b128 --base $BASE || { echo V40T10_MERGE_FAILED; exit 1; }
echo "V40T10_MERGED $(ts)"
sed "s#lamba04v40t2_merged_fp8b128#lamba04v40t10_merged_fp8b128#; s#--served-model-name lamba04v40t2#--served-model-name lamba04v40t10#" ~/launch_vllm_v40t2_merged.sh > ~/launch_vllm_v40t10_merged.sh; chmod +x ~/launch_vllm_v40t10_merged.sh
systemd-run --user --unit=lamba-infer --collect -p MemoryMax=27G -p MemorySwapMax=0 ~/launch_vllm_v40t10_merged.sh >/dev/null
until curl -s -m 5 http://127.0.0.1:8001/v1/models | grep -q lamba04v40t10; do systemctl --user is-active --quiet lamba-infer || { echo V40T10_SERVE_FAILED; exit 1; }; sleep 15; done
echo "V40T10_SERVED $(ts)"
cd $A; python3 evalq.py v40t10 init; echo "V40T10_EVALQ_START $(ts)"
python3 -u evalq.py v40t10 run box1:2 > ~/evalq_v40t10_box1.log 2>&1
echo "V40T10_EVALQ_DONE $(ts)"
