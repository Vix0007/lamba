#!/usr/bin/env bash
# v40t11p2 full pipeline. log ~/v40t12.log (markers T11P2_*)
exec >> ~/v40t12.log 2>&1
P=/home/ubuntu/antidoom/.venv/bin/python; A=/home/ubuntu/antidoom_data; BASE=/home/ubuntu/models/cand04_bf16_sh; RUN=lamba_04_v40t11p2; TL=~/lamba04_v40t11p2_train.log
ts(){ TZ=Asia/Singapore date +%T; }; cd $A
systemctl --user stop lamba-infer; sleep 15; until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 10; done
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd ~/antidoom; rm -rf runs/$RUN
echo "T11P2_TRAIN_START $(ts)"
uv run antidoom -c configs/lamba_04_v40t11p2.yaml -r runs/$RUN --model-name $BASE train --dataset-jsonl $A/l03_training_data_v40t11p2.jsonl > $TL 2>&1
grep -q 'REFUSING this checkpoint' $TL && { echo "T11P2_DISKGUARD_STOP"; exit 1; }
EP=$(grep -o "'train_runtime'[^}]*'epoch': '[0-9.]*'" $TL | tail -1 | grep -o "'epoch': '[0-9.]*'" | grep -oE '[0-9.]+' | tail -1)
python3 -c "import sys; sys.exit(0 if float('${EP:-0}') >= 0.95 else 1)" || { echo "T11P2_TRAIN_INCOMPLETE epoch=$EP"; exit 1; }
echo "T11P2_TRAIN_OK epoch=$EP $(ts)"
for c in runs/$RUN/lora/checkpoint-*; do [ -d "$c" ] && rm -r "$c"; done
$P $A/convert_adapter_for_vllm.py runs/$RUN/lora /home/ubuntu/adapters_$RUN || { echo T11P2_CONVERT_FAILED; exit 1; }
$P $A/make_lmh_model.py /home/ubuntu/adapters_$RUN/lm_head_lora.safetensors /home/ubuntu/models/lmh_$RUN --base $BASE --scale 1.0 || { echo T11P2_LMH_FAILED; exit 1; }
rm -f /home/ubuntu/models/lmh_$RUN/generation_config.json; python3 -c "
import json; g=json.load(open('$BASE/generation_config.json')); g['repetition_penalty']=1.05; json.dump(g,open('/home/ubuntu/models/lmh_$RUN/generation_config.json','w'),indent=2)"
$P -u $A/merge_quant_block128.py /home/ubuntu/adapters_$RUN /home/ubuntu/models/lmh_$RUN /home/ubuntu/models/lamba04v40t11p2_merged_fp8b128 --base $BASE || { echo T11P2_MERGE_FAILED; exit 1; }
echo "T11P2_MERGED $(ts)"
sed "s#lamba04v40t2_merged_fp8b128#lamba04v40t11p2_merged_fp8b128#; s#--served-model-name lamba04v40t2#--served-model-name lamba04v40t11p2#" ~/launch_vllm_v40t2_merged.sh > ~/launch_vllm_v40t11p2_merged.sh; chmod +x ~/launch_vllm_v40t11p2_merged.sh
systemd-run --user --unit=lamba-infer --collect -p MemoryMax=27G -p MemorySwapMax=0 ~/launch_vllm_v40t11p2_merged.sh >/dev/null
until curl -s -m 5 http://127.0.0.1:8001/v1/models | grep -q lamba04v40t11p2; do systemctl --user is-active --quiet lamba-infer || { echo T11P2_SERVE_FAILED; exit 1; }; sleep 15; done
echo "T11P2_SERVED $(ts)"
cd $A; python3 evalq.py v40t11p2 init; echo "T11P2_EVALQ_START $(ts)"
python3 -u evalq.py v40t11p2 run box1:2 > ~/evalq_v40t11p2_box1.log 2>&1
echo "T11P2_EVALQ_DONE $(ts)"
