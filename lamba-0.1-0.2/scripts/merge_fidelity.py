"""Weight-level MERGE FIDELITY: how much of a LoRA delta d = (alpha/r) B@A survives when merged and served?
For sampled modules it reports, per serving scheme, projection ratio  <eff,d>/<d,d>  (1.0 = fully preserved), cosine(eff,d), and
noise/signal ||eff-d||/||d||, where eff = (served weight WITH adapter) - (served weight WITHOUT adapter).
Schemes:  bf16_nearest | bf16_SR | fp8_pt(nearest bf16 -> vLLM-style per-tensor online fp8) | fp8_SR_offline (unbiased SR straight to fp8 at base scale)
usage: merge_fidelity.py <adapter_safetensors> [--scale 0.5] [--layers 58,63]"""
import argparse, json, re
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser(); ap.add_argument("adapter"); ap.add_argument("--scale", type=float, default=0.5)
ap.add_argument("--layers", default="58,63"); ap.add_argument("--base", default="/home/ubuntu/models/soup-linear-l1.0")
a = ap.parse_args()
idx = json.load(open(f"{a.base}/model.safetensors.index.json"))["weight_map"]
FP8 = torch.float8_e4m3fn; FMAX = torch.finfo(FP8).max
g = torch.Generator().manual_seed(7)

def sr_bf16(x):        # stochastic rounding fp32 -> bf16 (unbiased)
    bits = x.contiguous().view(torch.int32); noise = torch.randint(0, 65536, x.shape, dtype=torch.int32, generator=g)
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)
def fp8_pt(w32):        # vLLM-style dynamic per-tensor weight quant (scale = amax/448), round-to-nearest; returns dequantized fp32
    s = w32.abs().max() / FMAX; return (w32 / s).clamp(-FMAX, FMAX).to(FP8).float() * s, s
def fp8_sr(w32, s):     # unbiased stochastic rounding onto the fp8 grid (given scale s)
    y = (w32 / s).clamp(-FMAX, FMAX); lo = y.to(FP8).float(); hi = torch.where(lo <= y, torch.nextafter(lo, torch.tensor(1e9)), torch.nextafter(lo, torch.tensor(-1e9)))
    lo, hi = torch.minimum(lo, hi), torch.maximum(lo, hi); p = ((y - lo) / (hi - lo).clamp(min=1e-30)).clamp(0, 1)
    return torch.where(torch.rand(y.shape, generator=g) < p, hi, lo) * s
def stats(eff, d):
    dd = (d * d).sum(); return float((eff * d).sum() / dd), float(torch.nn.functional.cosine_similarity(eff.flatten(), d.flatten(), dim=0)), float((eff - d).norm() / d.norm())

ad = safe_open(a.adapter, "pt", device="cpu"); keys = [k for k in ad.keys() if ".lora_A." in k]
want = set(a.layers.split(",")); rows = []
for ka in keys:
    kb = ka.replace(".lora_A.", ".lora_B."); k = ka.replace("base_model.model.", "", 1).rsplit(".lora_", 1)[0]
    m = re.search(r"layers\.(\d+)\.", k)
    if k == "lm_head": continue   # 248320x5120 fp32 temporaries (5GB each) OOM-killed the host at 24.8GB; lm_head handled analytically (vLLM keeps it bf16; SR merge measured unbiased)
    elif m and m.group(1) in want and any(t in k for t in ("mlp.down_proj", "mlp.gate_proj", "linear_attn.in_proj_qkv", "self_attn.q_proj")):
        name = k + ".weight"
        if name not in idx: name = "model.language_model." + k[len("model."):] + ".weight"  # fallback for older adapter naming
    else: continue
    with safe_open(f"{a.base}/{idx[name]}", "pt", device="cpu") as f: W = f.get_tensor(name)
    d = (ad.get_tensor(kb).float() @ ad.get_tensor(ka).float()) * a.scale; W32 = W.float()
    out = {"module": k.replace("model.layers.", "L"), "|d|/|W|": float(d.norm() / W32.norm()), "max|d|": float(d.abs().max())}
    out["bf16_nearest"] = stats((W32 + d).to(torch.bfloat16).float() - W32, d)
    out["bf16_SR"] = stats(sr_bf16(W32 + d).float() - W32, d)
    if name != "lm_head.weight":   # vLLM keeps lm_head unquantized (bf16); fp8 schemes apply to the quantized linears
        base_q, s = fp8_pt(W32)
        out["fp8_pt(nearest->fp8)"] = stats(fp8_pt((W32 + d).to(torch.bfloat16).float())[0] - base_q, d)
        out["fp8_pt(SRbf16->fp8)"] = stats(fp8_pt(sr_bf16(W32 + d).float())[0] - base_q, d)
        out["fp8_SR_offline"] = stats(fp8_sr(W32 + d, s) - base_q, d)
    rows.append(out); print(json.dumps({k: ([round(x, 3) for x in v] if isinstance(v, tuple) else round(v, 5)) if not isinstance(v, str) else v for k, v in out.items()}), flush=True)
print("\nLEGEND: tuples are (projection ratio, cosine, noise/signal). Want projection ~1 and cosine high; fp8_SR_offline is unbiased (projection ~1) but noisy.")
