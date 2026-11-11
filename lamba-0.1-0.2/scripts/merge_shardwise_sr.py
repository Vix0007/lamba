"""
Shard-wise, resumable LoRA merge (host RAM is 30GB; base is ~57GB bf16, so a normal
load-and-merge OOMs). For each base shard: load tensors, add scale*(B@A) where the adapter
has a pair, write to <out>/<shard>.tmp then atomically rename. Finished shards are skipped.
"""
import json, os, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "/home/ubuntu/models/soup-linear-l1.0"
ADAPTER = "/home/ubuntu/antidoom/runs/lamba1/lora/checkpoint-44"
OUT = "/home/ubuntu/models/soup-linear-l1.0-ftpo1-sr"
dev = "cuda"
gen = torch.Generator(device=dev); gen.manual_seed(1234)

cfg = json.load(open(f"{ADAPTER}/adapter_config.json"))
scale = cfg["lora_alpha"] / cfg["r"]
print(f"r={cfg['r']} alpha={cfg['lora_alpha']} scale={scale}", flush=True)

os.makedirs(OUT, exist_ok=True)
for fn in os.listdir(BASE):
    if fn.endswith(".safetensors") or fn.startswith(".") or ".safetensors." in fn:
        continue
    if not os.path.exists(f"{OUT}/{fn}"):
        shutil.copy2(f"{BASE}/{fn}", f"{OUT}/{fn}")

idx = json.load(open(f"{BASE}/model.safetensors.index.json"))
wmap = idx["weight_map"]

def base_name(adapter_key):
    k = adapter_key.replace("base_model.model.", "", 1)
    k = k.rsplit(".lora_", 1)[0]
    if k == "lm_head":
        return "lm_head.weight"
    assert k.startswith("model.layers."), k
    return "model.language_model." + k[len("model."):] + ".weight"

pairs = {}
with safe_open(f"{ADAPTER}/adapter_model.safetensors", "pt", device="cpu") as f:
    for k in f.keys():
        if ".lora_A." in k:
            b = k.replace(".lora_A.", ".lora_B.")
            pairs[base_name(k)] = (k, b)
missing = [n for n in pairs if n not in wmap]
assert not missing, f"adapter targets not in base: {missing[:5]}"
print(f"{len(pairs)} adapter targets, all present in base index", flush=True)

ad = safe_open(f"{ADAPTER}/adapter_model.safetensors", "pt", device="cpu")
shards = sorted(set(wmap.values()))
stats = {"merged": 0, "skipped_zero": 0, "max_abs": 0.0}
t0 = time.time()
for si, shard in enumerate(shards, 1):
    final = f"{OUT}/{shard}"
    if os.path.exists(final):
        print(f"[{si}/{len(shards)}] {shard} already done, skipping", flush=True)
        continue
    tensors = {}
    with safe_open(f"{BASE}/{shard}", "pt", device="cpu") as f:
        for name in f.keys():
            w = f.get_tensor(name)
            if name in pairs:
                ka, kb = pairs[name]
                B = ad.get_tensor(kb).to(dev, torch.float32)
                if not torch.any(B):
                    stats["skipped_zero"] += 1
                else:
                    A = ad.get_tensor(ka).to(dev, torch.float32)
                    delta = (B @ A) * scale
                    assert delta.shape == w.shape, (name, delta.shape, w.shape)
                    x = (w.to(dev, torch.float32) + delta).contiguous()
                    noise = torch.randint(0, 65536, x.shape, device=dev, dtype=torch.int32, generator=gen)
                    w = ((x.view(torch.int32) + noise) & -65536).view(torch.float32).to(torch.bfloat16).cpu()
                    stats["merged"] += 1
                    stats["max_abs"] = max(stats["max_abs"], float(delta.abs().max()))
                    del A, delta
                del B
            tensors[name] = w.contiguous()
    save_file(tensors, final + ".tmp", metadata={"format": "pt"})
    os.replace(final + ".tmp", final)
    del tensors
    torch.cuda.empty_cache()
    print(f"[{si}/{len(shards)}] {shard} done  merged={stats['merged']} zeroB={stats['skipped_zero']} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)

json.dump(stats, open(f"{OUT}/merge_stats.json", "w"))
print("DONE", stats, flush=True)
