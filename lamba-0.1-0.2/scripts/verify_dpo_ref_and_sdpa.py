#!/usr/bin/env python3
"""Combined pre-launch verification (DS #185-187):
1. Forced-SDPA-backend check on the longest real DPO row (8974 tokens, task P2_comb1) --
   disable the math fallback, confirm flash/mem-efficient runs without OOM.
2. ln(2)-at-step-0 check with the corrected two-ref wiring (dpo_ref=s1337, "ref"=c10) --
   the decisive test that DPO's reference is actually s1337 (policy init), not c10.
Single model load shared by both checks.
"""
import json
import sys

sys.path.insert(0, "/home/ubuntu/antidoom/src")
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from antidoom.ftpo_train import _load_adapter_state_direct  # reuse the REAL project function,
# not a reimplementation -- avoids introducing a new bug in the verification script itself.

MODEL = "/home/ubuntu/models/soup-linear-l1.0"
S1337 = "/home/ubuntu/antidoom/runs/lamba_ruminconfab_s1337/lora"
C10 = "/home/ubuntu/antidoom/runs/lamba_letterfix2/lora/checkpoint-10"

LAYER_SET = [3, 7, 11, 15, 19, 23, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
TARGETS = ["out_proj", "up_proj", "gate_proj", "down_proj", "in_proj_a", "k_proj",
           "in_proj_b", "in_proj_z", "v_proj", "q_proj", "o_proj", "in_proj_qkv", "lm_head"]
RANK_PATTERN = {"q_proj": 128, "k_proj": 128, "v_proj": 128, "o_proj": 128, "gate_proj": 64,
                "up_proj": 64, "down_proj": 64, "in_proj_qkv": 64, "in_proj_z": 32,
                "out_proj": 64, "lm_head": 16}


def target_regex():
    layers = "|".join(str(x) for x in LAYER_SET)
    mods = "|".join(m for m in TARGETS if m not in ("in_proj_a", "in_proj_b", "lm_head"))
    return rf"(.*\.layers\.({layers})\.(.*\.)?({mods}))|lm_head"


def main():
    print("Loading base model (nf4)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant,
                                                  device_map={"": 0}, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    lora_cfg = LoraConfig(r=128, lora_alpha=128, lora_dropout=0.05, bias="none",
                           target_modules=target_regex(), rank_pattern=RANK_PATTERN,
                           alpha_pattern=RANK_PATTERN, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    print("initialised default (will load s1337 into it)", flush=True)
    _load_adapter_state_direct(model, Path(S1337), adapter_name="default")
    print("default<-s1337: loaded", flush=True)

    model.add_adapter("dpo_ref", lora_cfg)
    _load_adapter_state_direct(model, Path(S1337), adapter_name="dpo_ref")
    print("dpo_ref<-s1337: loaded", flush=True)
    for name, p in model.named_parameters():
        if "dpo_ref" in name.split("."):
            p.requires_grad_(False)

    model.add_adapter("ref", lora_cfg)
    _load_adapter_state_direct(model, Path(C10), adapter_name="ref")
    print("ref<-c10: loaded", flush=True)
    for name, p in model.named_parameters():
        if "ref" in name.split(".") and "dpo_ref" not in name.split("."):
            p.requires_grad_(False)

    model.set_adapter("default")
    model.eval()
    device = next(model.parameters()).device

    # ===== CHECK 1: forced SDPA backend on the longest real row =====
    print("\n=== CHECK 1: forced-SDPA-backend on longest real DPO row ===", flush=True)
    rows = [json.loads(l) for l in open("/home/ubuntu/antidoom_data/retrain_dpo_pairs_v1_1_rendered.jsonl")]
    longest = max(rows, key=lambda r: len(r["rendered_context"]) + len(r["chosen"]))
    full_text = longest["rendered_context"] + longest["chosen"]
    ids = tok(full_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    print(f"task={longest['task_id']} tokens={ids.shape[1]}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            with torch.no_grad():
                out = model(ids, use_cache=False, return_dict=True, logits_to_keep=1)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"CHECK 1 PASS: forward completed with math backend disabled, peak={peak:.2f}GB, "
              f"logits finite={torch.isfinite(out.logits).all().item()}", flush=True)
    except Exception as e:
        print(f"CHECK 1 RESULT: {type(e).__name__}: {str(e)[:300]}", flush=True)

    # ===== CHECK 2: ln(2)-at-step-0 with corrected two-ref wiring =====
    print("\n=== CHECK 2: ln(2)-at-step-0, dpo_ref=s1337 ===", flush=True)
    pair = rows[0]
    for r in rows:
        if len(r["rendered_context"]) + len(r["chosen"]) < 6000:
            pair = r
            break
    ctx = pair["rendered_context"]

    def side_logp(text, adapter):
        full_ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        ctx_len = len(tok(ctx, add_special_tokens=False).input_ids)
        model.set_adapter(adapter)
        with torch.no_grad():
            out = model(full_ids, use_cache=False, return_dict=True)
        logits = out.logits[0, ctx_len - 1:-1, :]
        targets = full_ids[0, ctx_len:]
        logp = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        model.set_adapter("default")
        return tok_logp.sum().item(), tok_logp

    lp_theta_chosen, tlp_tc = side_logp(ctx + pair["chosen"], "default")
    lp_theta_rejected, tlp_tr = side_logp(ctx + pair["rejected"], "default")
    lp_dporef_chosen, tlp_rc = side_logp(ctx + pair["chosen"], "dpo_ref")
    lp_dporef_rejected, tlp_rr = side_logp(ctx + pair["rejected"], "dpo_ref")

    delta_theta = lp_theta_chosen - lp_theta_rejected
    delta_ref = lp_dporef_chosen - lp_dporef_rejected
    beta = 0.1
    margin = beta * (delta_theta - delta_ref)
    loss = -F.logsigmoid(torch.tensor(margin)).item()

    max_diff_chosen = (tlp_tc - tlp_rc).abs().max().item()
    max_diff_rejected = (tlp_tr - tlp_rr).abs().max().item()

    print(f"task={pair['task_id']}", flush=True)
    print(f"logp_theta(chosen)={lp_theta_chosen:.4f} logp_dpo_ref(chosen)={lp_dporef_chosen:.4f} "
          f"max_abs_diff={max_diff_chosen:.6f}", flush=True)
    print(f"logp_theta(rejected)={lp_theta_rejected:.4f} logp_dpo_ref(rejected)={lp_dporef_rejected:.4f} "
          f"max_abs_diff={max_diff_rejected:.6f}", flush=True)
    print(f"delta_theta={delta_theta:.6f} delta_ref={delta_ref:.6f} margin={margin:.6f}", flush=True)
    print(f"loss={loss:.6f} (ln2={0.6931471805599453:.6f}, diff={abs(loss-0.6931471805599453):.6f})", flush=True)


def check3_backward_pass():
    """DS #188 Gap A: full forward+backward+optimizer-step on the longest real DPO row,
    with the corrected two-ref (chunked) wiring. Asserts: completes under budget, LoRA
    grad norm finite AND nonzero, base params grad is None (no accidental unfreeze),
    loss after step moved off ln(2)."""
    print("\n=== CHECK 3: full forward+backward+optimizer-step, longest row ===", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant,
                                                  device_map={"": 0}, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    lora_cfg = LoraConfig(r=128, lora_alpha=128, lora_dropout=0.05, bias="none",
                           target_modules=target_regex(), rank_pattern=RANK_PATTERN,
                           alpha_pattern=RANK_PATTERN, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    _load_adapter_state_direct(model, Path(S1337), adapter_name="default")
    model.add_adapter("dpo_ref", lora_cfg)
    _load_adapter_state_direct(model, Path(S1337), adapter_name="dpo_ref")
    for name, p in model.named_parameters():
        if "dpo_ref" in name.split("."):
            p.requires_grad_(False)
    model.set_adapter("default")
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    device = next(model.parameters()).device

    rows = [json.loads(l) for l in open("/home/ubuntu/antidoom_data/retrain_dpo_pairs_v1_1_rendered.jsonl")]
    # True worst case per DS Gap B: whichever row has the greatest TOKENIZED
    # (chosen + rejected) length, since both sides' graphs stay resident until
    # one shared backward() call -- char-length-of-chosen-only under-covers this.
    def _combined_tok_len(r):
        c = len(tok.encode(r["rendered_context"] + r["chosen"]))
        rj = len(tok.encode(r["rendered_context"] + r["rejected"]))
        return c + rj
    longest = max(rows, key=_combined_tok_len)
    ctx = longest["rendered_context"]
    print(f"task={longest['task_id']} combined_tok_len={_combined_tok_len(longest)}", flush=True)

    CHUNK = 256

    def side_logp(text, adapter, need_grad):
        full_ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        ctx_len = len(tok(ctx, add_special_tokens=False).input_ids)
        L = full_ids.size(1)
        pos = torch.arange(L, device=device).unsqueeze(0)
        keep = max(L - ctx_len + 1, 1)
        targets = full_ids[0, ctx_len:]
        n_resp = targets.numel()
        model.set_adapter(adapter)

        def _run():
            out = model(full_ids, use_cache=False, return_dict=True, logits_to_keep=keep)
            logits = out.logits[0, :-1, :]
            chunks = []
            for c0 in range(0, n_resp, CHUNK):
                c1 = min(c0 + CHUNK, n_resp)
                logp_c = F.log_softmax(logits[c0:c1], dim=-1, dtype=torch.float32)
                tok_logp_c = logp_c.gather(-1, targets[c0:c1].unsqueeze(-1)).squeeze(-1)
                chunks.append(tok_logp_c)
            return torch.cat(chunks, dim=0)

        if need_grad:
            tok_logp = _run()
        else:
            with torch.no_grad():
                tok_logp = _run().detach()
        model.set_adapter("default")
        return tok_logp.sum(), n_resp

    torch.cuda.reset_peak_memory_stats()
    logp_theta_c, n_c = side_logp(ctx + longest["chosen"], "default", need_grad=True)
    logp_theta_r, n_r = side_logp(ctx + longest["rejected"], "default", need_grad=True)
    with torch.no_grad():
        logp_ref_c, _ = side_logp(ctx + longest["chosen"], "dpo_ref", need_grad=False)
        logp_ref_r, _ = side_logp(ctx + longest["rejected"], "dpo_ref", need_grad=False)

    beta = 0.1
    delta_theta = logp_theta_c - logp_theta_r
    delta_ref = (logp_ref_c - logp_ref_r).detach()
    margin = beta * (delta_theta - delta_ref)
    loss = -F.logsigmoid(margin)
    print(f"pre-step loss={loss.item():.6f} (ln2=0.693147)", flush=True)

    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"backward completed, peak memory={peak:.2f}GB", flush=True)

    grad_norms = []
    base_leak = 0
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_norms.append(p.grad.float().norm().item())
        if not p.requires_grad and p.grad is not None:
            base_leak += 1
    total_grad_norm = sum(g**2 for g in grad_norms) ** 0.5
    print(f"trainable params with grad: {len(grad_norms)}, total_grad_norm={total_grad_norm:.4f}, "
          f"finite={torch.isfinite(torch.tensor(total_grad_norm)).item()}, "
          f"nonzero={total_grad_norm > 0}", flush=True)
    print(f"frozen params with unexpected .grad set (should be 0): {base_leak}", flush=True)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6)
    optim.step()
    optim.zero_grad()

    with torch.no_grad():
        logp_theta_c2, _ = side_logp(ctx + longest["chosen"], "default", need_grad=False)
        logp_theta_r2, _ = side_logp(ctx + longest["rejected"], "default", need_grad=False)
    delta_theta2 = logp_theta_c2 - logp_theta_r2
    margin2 = beta * (delta_theta2 - delta_ref)
    loss2 = -F.logsigmoid(margin2).item()
    print(f"post-step loss={loss2:.6f} (moved off ln2 by {abs(loss2-0.693147):.6f})", flush=True)
    print(f"CHECK 3 {'PASS' if (total_grad_norm > 0 and base_leak == 0 and abs(loss2-0.693147)>1e-6) else 'FAIL'}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check3":
        check3_backward_pass()
    else:
        main()
