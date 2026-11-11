"""Round-2 adapter -> vLLM-servable LoRA dir + lm_head delta file.
usage: convert_adapter_for_vllm.py <adapter_dir> <out_dir>
 - keeps lora_A/lora_B only, renames base_model.model.model.layers.* -> base_model.model.model.language_model.layers.* (matches the checkpoint's HF naming)
 - drops lm_head LoRA (vLLM cannot serve it) and saves it to <out_dir>/lm_head_lora.safetensors for the stochastic-rounded lm_head merge (see merge script)
 - rewrites adapter_config.json: regex target_modules -> explicit module-name list (vLLM expects names), base path fixed."""
import json, re, sys, os
from safetensors import safe_open
from safetensors.torch import save_file
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
cfg = json.load(open(f"{src}/adapter_config.json"))
keep, head, names, layers = {}, {}, set(), set()
with safe_open(f"{src}/adapter_model.safetensors", "pt", device="cpu") as f:
    for k in f.keys():
        if "lora_" not in k: continue
        if ".lm_head." in k or k.startswith("base_model.model.lm_head"): head[k] = f.get_tensor(k).contiguous(); continue
        nk = k.replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers.")
        keep[nk] = f.get_tensor(k).contiguous()
        m = re.search(r"layers\.(\d+)\.(?:.*\.)?([A-Za-z_]+)\.lora_[AB]\.", nk); layers.add(int(m.group(1))); names.add(m.group(2))
save_file(keep, f"{out}/adapter_model.safetensors", metadata={"format": "pt"})
if head: save_file(head, f"{out}/lm_head_lora.safetensors", metadata={"format": "pt"})
cfg["target_modules"] = sorted(names); cfg["auto_mapping"] = None; cfg["base_model_name_or_path"] = "/home/ubuntu/models/soup-linear-l1.0"
json.dump(cfg, open(f"{out}/adapter_config.json", "w"), indent=1)
print(f"lora tensors kept {len(keep)} | lm_head tensors set aside {len(head)} | layers {min(layers)}..{max(layers)} ({len(layers)} layers) | target modules {sorted(names)}")
print("r =", cfg["r"], "alpha =", cfg["lora_alpha"], "-> out", out)
