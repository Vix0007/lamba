"""DS #135 pre-flight diagnostic: is the science regression's KL signature localized at repeat
onsets, or diffuse? Load kl005-seed1337 as 'default' and letterfix2-c10 as 'ref' (same
multi-adapter pattern the trainer itself uses for KL-anchor training, see ftpo_train.py). For
each looping science rollout (kl005's own generated tokens), teacher-force the sequence through
BOTH adapters and compute per-position KL(kl005 || c10). Compare onset positions (first token of
the 2nd+ occurrence of a repeated 60-char span) vs matched non-onset control positions in the
SAME rollouts.

No training happens here -- inference/scoring only.
"""
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

sys.path.insert(0, "/home/ubuntu/antidoom/src")
from antidoom.config import load_config
from antidoom.ftpo_train import _make_lora_config, _load_adapter_state_direct

MODEL_NAME = "/home/ubuntu/models/soup-linear-l1.0"
KL005_CFG = "/home/ubuntu/antidoom/configs/lamba_kl_sweep_005.yaml"  # for LoRA arch (rank pattern etc)
KL005_ADAPTER = "/home/ubuntu/kl005_seed1337_raw_adapter"
C10_ADAPTER = "/home/ubuntu/antidoom/runs/lamba_letterfix2/lora/checkpoint-10"
ROLLOUTS = "/tmp/loop_kl005_seed1337.jsonl"
PROMPTS = "/home/ubuntu/antidoom_data/concise_sft/holdout_confirm_prompts.jsonl"

al = re.compile(r"[A-Za-z0-9]")


def find_onsets(t: str, prompt: str):
    """Same detector as tail_stats_*.py. Returns list of (onset_char_pos, span)."""
    n = len(t)
    seen = set()
    onsets = []
    for i in range(0, n - 60, 30):
        w = t[i:i + 60]
        if w in seen or len(al.findall(w)) < 25:
            continue
        seen.add(w)
        if t.count(w) < 4 or w in prompt:
            continue
        first = t.find(w)
        second = t.find(w, first + 60)
        if second > 0:
            onsets.append(second)
    return sorted(set(onsets))


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}


def main():
    print("loading rollouts/prompts...", flush=True)
    ROLL = load(ROLLOUTS)
    P = load(PROMPTS)
    sci_ids = [i for i in P if P[i]["domain"] == "science"]
    cases = []
    for i in sci_ids:
        if i not in ROLL:
            continue
        onsets = find_onsets(ROLL[i]["text"], P[i]["prompt"])
        if onsets:
            cases.append((i, onsets))
    print(f"{len(cases)} looping science rollouts with onsets", flush=True)

    print("loading model + tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quant, device_map={"": 0}, trust_remote_code=True,
    )
    base.config.use_cache = False
    n_layers = getattr(base.config, "num_hidden_layers", None) or getattr(
        getattr(base.config, "text_config", None), "num_hidden_layers", None)

    app_cfg = load_config(Path(KL005_CFG))
    cfg = app_cfg.train
    lora_cfg = _make_lora_config(cfg, n_layers)
    model = get_peft_model(base, lora_cfg)
    _load_adapter_state_direct(model, Path(KL005_ADAPTER), adapter_name="default")
    model.add_adapter("ref", lora_cfg)
    _load_adapter_state_direct(model, Path(C10_ADAPTER), adapter_name="ref")
    model.eval()
    print("model loaded, both adapters populated", flush=True)

    device = next(model.parameters()).device

    out_jsonl = Path("/home/ubuntu/eval_results/onset_kl_diagnostic_percase.jsonl")
    done_ids = set()
    per_case_summary = []
    if out_jsonl.exists():
        for l in open(out_jsonl):
            r = json.loads(l)
            per_case_summary.append(r)
            done_ids.add(r["id"])
        print(f"resuming: {len(done_ids)} cases already done", flush=True)

    onset_kls = []
    control_kls = []
    for r in per_case_summary:
        onset_kls.extend(r.get("_onset_kls", []))
        control_kls.extend(r.get("_control_kls", []))

    with torch.no_grad():
        for case_idx, (pid, onset_char_positions) in enumerate(cases):
            if pid in done_ids:
                continue
            prompt = P[pid]["prompt"]
            gen_text = ROLL[pid]["text"]
            chat = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, reasoning_effort="medium")
            prompt_ids = tok(chat, add_special_tokens=False)["input_ids"]
            gen_ids = tok(gen_text, add_special_tokens=False)["input_ids"]
            full_ids = prompt_ids + gen_ids
            if len(full_ids) > 6000:
                # truncate from the front of the generation (keep prompt intact) to bound memory
                full_ids = prompt_ids + gen_ids[:6000 - len(prompt_ids)]
            input_ids = torch.tensor([full_ids], device=device)

            model.set_adapter("default")
            out_default = model(input_ids=input_ids)
            logp_default = F.log_softmax(out_default.logits[0].float(), dim=-1)
            del out_default
            torch.cuda.empty_cache()

            model.set_adapter("ref")
            out_ref = model(input_ids=input_ids)
            logp_ref = F.log_softmax(out_ref.logits[0].float(), dim=-1)
            del out_ref
            torch.cuda.empty_cache()

            # per-position KL(default || ref) at the position PREDICTING the token at t+1,
            # i.e. logits at position t (0-indexed into input_ids) predict token t+1.
            p_default = logp_default.exp()
            kl_per_pos = (p_default * (logp_default - logp_ref)).sum(dim=-1)  # [seq_len]
            kl_per_pos = kl_per_pos.cpu()
            del logp_default, logp_ref, p_default
            torch.cuda.empty_cache()

            n_prompt = len(prompt_ids)
            gen_len = len(full_ids) - n_prompt
            # map onset char positions -> approx token index within the generation by
            # re-tokenizing the prefix up to each onset char position
            onset_tok_idx = set()
            for cp in onset_char_positions:
                prefix_ids = tok(gen_text[:cp], add_special_tokens=False)["input_ids"]
                tidx = n_prompt + len(prefix_ids)
                if n_prompt <= tidx < len(full_ids):
                    onset_tok_idx.add(tidx)

            case_onset_kls = []
            case_control_kls = []
            for t in range(n_prompt, len(full_ids) - 1):
                # logits at position t-1 predict token t (standard shift)
                pos_kl = kl_per_pos[t - 1].item()
                if t in onset_tok_idx:
                    onset_kls.append(pos_kl)
                    case_onset_kls.append(pos_kl)
                else:
                    control_kls.append(pos_kl)
                    case_control_kls.append(pos_kl)

            import statistics as st
            summary = {
                "id": pid, "n_onsets": len(onset_tok_idx),
                "mean_onset_kl": st.mean(case_onset_kls) if case_onset_kls else None,
                "median_onset_kl": st.median(case_onset_kls) if case_onset_kls else None,
                "mean_control_kl": st.mean(case_control_kls) if case_control_kls else None,
                "median_control_kl": st.median(case_control_kls) if case_control_kls else None,
                "_onset_kls": case_onset_kls, "_control_kls": case_control_kls,
            }
            per_case_summary.append(summary)
            with open(out_jsonl, "a") as f:
                f.write(json.dumps(summary) + "\n")
            onset_kls.extend(case_onset_kls)
            control_kls.extend(case_control_kls)
            print(f"[{case_idx+1}/{len(cases)}] {pid}: mean_onset={summary['mean_onset_kl']:.5g} "
                  f"mean_control={summary['mean_control_kl']:.5g}", flush=True)

    import statistics as st
    print("\n=== AGGREGATE ===")
    print(f"n_onset_positions={len(onset_kls)} n_control_positions={len(control_kls)}")
    print(f"mean_onset_KL={st.mean(onset_kls):.4f}  median_onset_KL={st.median(onset_kls):.4f}")
    print(f"mean_control_KL={st.mean(control_kls):.4f}  median_control_KL={st.median(control_kls):.4f}")
    ratio = st.mean(onset_kls) / max(st.mean(control_kls), 1e-9)
    print(f"ratio (onset/control) = {ratio:.2f}x")
    with open("/home/ubuntu/eval_results/onset_kl_diagnostic.json", "w") as f:
        json.dump({"per_case": per_case_summary,
                   "aggregate": {"n_onset": len(onset_kls), "n_control": len(control_kls),
                                 "mean_onset_kl": st.mean(onset_kls), "median_onset_kl": st.median(onset_kls),
                                 "mean_control_kl": st.mean(control_kls), "median_control_kl": st.median(control_kls),
                                 "ratio": ratio}}, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
