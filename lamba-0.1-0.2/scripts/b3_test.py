"""B3 validation: does removing HF's forced input-require-grads hook keep loss + LoRA grads identical and speed up fwd+bwd? One nf4 model load.
Writes ~/b3_test.json and prints B3_PASS / B3_FAIL."""
import json, sys, time, statistics
from pathlib import Path
import torch
sys.path.insert(0, "/home/ubuntu/antidoom/src")
from antidoom.config import load_config
from antidoom import ftpo_train as ft
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model
import torch.nn.functional as F
cfg = load_config(Path("/home/ubuntu/antidoom/configs/lamba_r2.yaml")).train
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/soup-linear-l1.0", trust_remote_code=True)
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("/home/ubuntu/models/soup-linear-l1.0", quantization_config=q, device_map={"": 0}, trust_remote_code=True)
model.config.use_cache = False
n_layers = model.config.num_hidden_layers
model = get_peft_model(model, ft._make_lora_config(cfg, n_layers))
model.enable_input_require_grads(); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
ft.freeze_lora_to_last_k_layers(model, k=cfg.n_layers_unfrozen, target_modules=tuple(cfg.target_modules)); model.train()
rows = [json.loads(l) for l in open("/home/ubuntu/antidoom_data/all_ftpo_pairs_r2.jsonl")]
ex = []
for r in rows:
    ids = tok(r["context_with_chat_template"], add_special_tokens=False).input_ids
    if 3000 < len(ids) < 5000:
        ex.append((ids, tok(r["multi_chosen_decoded"][0], add_special_tokens=False).input_ids[0], tok(r["rejected_decoded"], add_special_tokens=False).input_ids[0]))
    if len(ex) == 4: break
params = [p for p in model.parameters() if p.requires_grad]
def run(mode):
    res = []
    for ids, ch, rj in ex:
        for p in params: p.grad = None
        x = torch.tensor([ids], device="cuda"); torch.cuda.synchronize(); t = time.time()
        out = model(x, use_cache=False, logits_to_keep=1); lg = out.logits[:, -1, :].float()
        loss = F.softplus(2.0 - (lg[0, ch] - lg[0, rj])); loss.backward(); torch.cuda.synchronize()
        gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))
        res.append({"loss": float(loss), "gn": gn, "s": time.time() - t, "n": len(ids)})
    return res
run("warm")
a = run("hook_on")
emb = model.get_input_embeddings(); print('hooks before clear:', len(emb._forward_hooks), flush=True); emb._forward_hooks.clear()
b = run("hook_off")
torch.cuda.reset_peak_memory_stats()
ok = all(abs(x["loss"] - y["loss"]) < 5e-3 * max(1, abs(x["loss"])) and abs(x["gn"] - y["gn"]) < 0.05 * x["gn"] for x, y in zip(a, b))
sp = statistics.mean(x["s"] for x in a) / statistics.mean(y["s"] for y in b)
out = {"hook_on": a, "hook_off": b, "speedup": sp, "identical": ok}
json.dump(out, open("/home/ubuntu/b3_test.json", "w"), indent=1)
print("B3_RESULT", json.dumps({"identical": ok, "speedup": round(sp, 2), "gn_on": [round(x["gn"], 3) for x in a], "gn_off": [round(y["gn"], 3) for y in b]}))
print("B3_PASS" if (ok and sp > 1.15) else "B3_FAIL", flush=True)
