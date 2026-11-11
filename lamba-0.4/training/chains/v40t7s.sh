#!/usr/bin/env bash
# Stage 2 chain: after v40t7 is merged+served -> quick held-out prayer sample on v40t7 -> stop v40t7 chain/evals -> free disk ->
# train stage 2 (plain SFT, probe recipe) from v40t7 -> merge -> serve lamba04v40t7s -> quick prayer sample -> evalq v40t7s box1:2. log ~/v40t7.log
exec >> ~/v40t7.log 2>&1
P=/home/ubuntu/antidoom/.venv/bin/python; A=/home/ubuntu/antidoom_data; BASE=/home/ubuntu/models/cand04_bf16_sh; RUN=lamba_04_v40t7s; TL=~/lamba04_v40t7s_train.log
ts(){ TZ=Asia/Singapore date +%T; }
until grep -qE "^V40T7_(SERVED|.*FAIL|TRAIN_INCOMPLETE|DISK)" ~/v40t7.log; do sleep 30; done
grep -q "^V40T7_SERVED" ~/v40t7.log || { echo "V40T7S_ABORT stage1 failed $(ts)"; exit 1; }
cd $A; ~/venv-vllm/bin/python -u warmth_gen.py lamba04v40t7 bm/prayer_heldout.json ~/eval_results/v40t7_mine/prayer_ho_v40t7.jsonl --n 4 --conds none && echo "V40T7_PRAYER_SAMPLED $(ts)"
systemctl --user stop lamba-v40t7 2>/dev/null; python3 evalq.py v40t7 release >/dev/null 2>&1
systemctl --user stop lamba-infer 2>/dev/null; sleep 15; until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 10; done
for c in ~/antidoom/runs/lamba_04_v40t7/lora/checkpoint-*; do [ -d "$c" ] && [ -f ~/antidoom/runs/lamba_04_v40t7/lora/adapter_model.safetensors ] && rm -r "$c"; done
[ -f ~/adapters_lamba_04_v40t7/lm_head_lora.safetensors ] && rm -r /home/ubuntu/models/lamba04v40t7_merged_fp8b128
[ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -ge 45 ] || { echo "V40T7S_DISK_TOO_LOW $(ts)"; exit 1; }
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd ~/antidoom
echo "V40T7S_TRAIN_START $(ts)"
uv run antidoom -c configs/lamba_04_v40t7s.yaml -r runs/$RUN --model-name $BASE train --dataset-jsonl $A/l03_training_data_v40t7s.jsonl > $TL 2>&1
EP=$(grep -o "'train_runtime'[^}]*'epoch': '[0-9.]*'" $TL | tail -1 | grep -o "'epoch': '[0-9.]*'" | grep -oE '[0-9.]+' | tail -1)
python3 -c "import sys; sys.exit(0 if float('${EP:-0}') >= 2.9 else 1)" || { echo "V40T7S_TRAIN_INCOMPLETE epoch=$EP"; exit 1; }
echo "V40T7S_TRAIN_OK epoch=$EP $(ts)"
$P $A/convert_adapter_for_vllm.py runs/$RUN/lora /home/ubuntu/adapters_$RUN || { echo V40T7S_CONVERT_FAILED; exit 1; }
$P $A/make_lmh_model.py /home/ubuntu/adapters_$RUN/lm_head_lora.safetensors /home/ubuntu/models/lmh_$RUN --base $BASE --scale 1.0 || { echo V40T7S_LMH_FAILED; exit 1; }
rm -f /home/ubuntu/models/lmh_$RUN/generation_config.json; python3 -c "
import json; g=json.load(open('$BASE/generation_config.json')); g['repetition_penalty']=1.05; json.dump(g,open('/home/ubuntu/models/lmh_$RUN/generation_config.json','w'),indent=2)"
$P -u $A/merge_quant_block128.py /home/ubuntu/adapters_$RUN /home/ubuntu/models/lmh_$RUN /home/ubuntu/models/lamba04v40t7s_merged_fp8b128 --base $BASE || { echo V40T7S_MERGE_FAILED; exit 1; }
echo "V40T7S_MERGED $(ts)"
sed "s#lamba04v40t2_merged_fp8b128#lamba04v40t7s_merged_fp8b128#; s#--served-model-name lamba04v40t2#--served-model-name lamba04v40t7s#" ~/launch_vllm_v40t2_merged.sh > ~/launch_vllm_v40t7s_merged.sh; chmod +x ~/launch_vllm_v40t7s_merged.sh
systemd-run --user --unit=lamba-infer --collect -p MemoryMax=27G -p MemorySwapMax=0 ~/launch_vllm_v40t7s_merged.sh >/dev/null
until curl -s -m 5 http://127.0.0.1:8001/v1/models | grep -q lamba04v40t7s; do systemctl --user is-active --quiet lamba-infer || { echo V40T7S_SERVE_FAILED; exit 1; }; sleep 15; done
echo "V40T7S_SERVED $(ts)"
cd $A; ~/venv-vllm/bin/python -u warmth_gen.py lamba04v40t7s bm/prayer_heldout.json ~/eval_results/v40t7_mine/prayer_ho_v40t7s.jsonl --n 4 --conds none && echo "V40T7S_PRAYER_SAMPLED $(ts)"
python3 evalq.py v40t7s init; echo "V40T7S_EVALQ_START $(ts)"
python3 -u evalq.py v40t7s run box1:2 > ~/evalq_v40t7s_box1.log 2>&1
echo "V40T7S_EVALQ_BOX1_EXITED $(ts)"
