import argparse, json, math, os, re, statistics as st, time
from pathlib import Path
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL = "/home/ubuntu/models/soup-linear-l1.0"
SAMPLES = "/home/ubuntu/s3_reads/soup-linear-l1.0_malaymmlu_full_samples.jsonl"
MM = str(Path.home() / "datasets" / "MalayMMLU_0shot.json")
OUT = Path.home() / "eval_results"
LET = "ABCDE"


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def opt_letters(p):
    seen, out = set(), []
    for m in re.findall(r"(?m)^([A-E])\. ", p):
        if m not in seen:
            seen.add(m); out.append(m)
    return out


def amax(r):
    return max(r["cands"], key=lambda L: r["lp"].get(L, -30.0))


def summ_A(path):
    v = [r for r in jl(path) if r["gold"] in LET[:4] and r["gold"] in r["cands"]]
    N = len(v)
    if not N:
        return {}
    per = {}
    for L in LET[:4]:
        g = [r for r in v if r["gold"] == L]
        per[L] = {"n": len(g), "acc": (sum(1 for r in g if amax(r) == L) / len(g)) if g else None}
    good = [per[L]["acc"] for L in LET[:4] if per[L]["acc"] is not None]
    cnt = {L: sum(1 for r in v if amax(r) == L) for L in LET[:4]}
    s = sum(cnt.values()) or 1
    pred = {L: cnt[L] / s for L in LET[:4]}
    gold = {L: per[L]["n"] / N for L in LET[:4]}
    kl = sum(pred[L] * math.log(pred[L] / gold[L]) for L in LET[:4] if pred[L] > 0 and gold[L] > 0)
    a, b = per["A"]["acc"], per["B"]["acc"]
    return {"n": N, "acc": sum(1 for r in v if amax(r) == r["gold"]) / N, "by_gold": per,
            "pos_bal_acc": sum(good) / len(good), "rstd_pp": st.pstdev(good) * 100 if len(good) > 1 else 0.0,
            "ckld": kl, "gap_AB_pp": (a - b) * 100 if a is not None and b is not None else None,
            "pred_rates": {L: sum(1 for r in v if amax(r) == L) / N for L in LET}}


def summ_B(path):
    items = {}
    for r in jl(path):
        items.setdefault(r["id"], []).append(r)
    if not items:
        return {}
    s0 = s0n = perm = nperm = cons = 0
    agr, posc = [], {}
    for rs in items.values():
        rs.sort(key=lambda x: x["k"])
        n, gold = rs[0]["n"], rs[0]["gold"]
        L, wc, S = LET[:n], [], {c: 0.0 for c in range(n)}
        for r in rs:
            lp, order = r["lp"], r["order"]
            w = max(range(n), key=lambda j: lp.get(L[j], -30.0))
            wc.append(order[w])
            for j in range(n):
                S[order[j]] += lp.get(L[j], -30.0)
            p = order.index(gold)
            d = posc.setdefault(p, [0, 0]); d[1] += 1; d[0] += int(w == p)
            if r["k"] == 0:
                s0 += int(order[w] == gold); s0n += 1
        perm += int(max(S, key=S.get) == gold); nperm += 1
        cons += int(len(set(wc)) == 1)
        mode = max(set(wc), key=wc.count)
        agr.append(wc.count(mode) / len(wc))
    pa = {p: c[0] / c[1] for p, c in sorted(posc.items())}
    vals = list(pa.values()); m = sum(vals) / len(vals)
    sd = st.pstdev(vals) if len(vals) > 1 else 0.0
    return {"n_items": len(items), "shift0_acc": s0 / max(s0n, 1), "perm_acc": perm / nperm,
            "consistent": cons / len(items), "mode_agree": sum(agr) / len(agr),
            "pos_acc": {str(k): v for k, v in pa.items()}, "rstd_pp": sd * 100,
            "fluct": sd / m if m else 0.0}


def show(tag, a, b):
    pc = lambda x: "NA" if x is None else f"{x*100:.2f}%"
    print(f"\n=== {tag} ===")
    if a:
        gap = "NA" if a["gap_AB_pp"] is None else f"{a['gap_AB_pp']:+.2f}pp"
        print(f"A n={a['n']} acc={pc(a['acc'])} posbal={pc(a['pos_bal_acc'])} RStd={a['rstd_pp']:.2f}pp "
              f"CKLD={a['ckld']:.4f} gap(A-B)={gap}")
        print("  by-gold " + " ".join(f"{L}={pc(a['by_gold'][L]['acc'])}(n={a['by_gold'][L]['n']})" for L in "ABCD"))
        print("  pred-rate " + " ".join(f"{L}={pc(v)}" for L, v in a["pred_rates"].items()))
    if b:
        print(f"B n_items={b['n_items']} shift0={pc(b['shift0_acc'])} perm-avg={pc(b['perm_acc'])} "
              f"content-consistent={pc(b['consistent'])} mode-agree={pc(b['mode_agree'])} "
              f"RStd={b['rstd_pp']:.2f}pp fluct={b['fluct']:.3f}")
        print("  pos-acc " + " ".join(f"{LET[int(k)]}={pc(v)}"
              for k, v in sorted(b["pos_acc"].items(), key=lambda kv: int(kv[0]))))


def run_jobs(tag, jobs, model=None, lora=None, mtp=False, llm=None, lreq=None):
    """Core eval logic (unchanged from the original main() body). If `llm` is provided (from
    eval_combined.py sharing one engine across multiple evals), reuse it and skip building a new
    one -- otherwise build one exactly as the standalone CLI always has."""
    OUT.mkdir(parents=True, exist_ok=True)

    plan = []
    if "A" in jobs:
        for i, r in enumerate(jl(SAMPLES)):
            cl = opt_letters(r["prompt"])
            if cl:
                plan.append({"job": "A", "key": i, "id": i, "gold": str(r["gold"]).strip().upper(),
                             "cands": cl, "ctx": r["prompt"].rstrip() + "\nAnswer:"})
    if "B" in jobs:
        for it in json.load(open(MM)):
            opts = it["options"]; n = len(opts)
            if not 2 <= n <= len(LET):
                continue
            k = str(it["key"]).strip().upper()
            if k not in LET[:n]:
                continue
            g = LET.index(k)
            texts = [re.sub(r"^[A-J]\.\s*", "", o) for o in opts]
            stem = it["prompt"].split("\nA.")[0]
            for s in range(n):
                order = [(i + s) % n for i in range(n)]
                ctx = stem + "\n" + "\n".join(f"{LET[i]}. {texts[order[i]]}" for i in range(n)) + "\nAnswer:"
                plan.append({"job": "B", "key": f"{it['id']}:{s}", "id": it["id"], "k": s, "n": n,
                             "order": order, "gold": g, "ctx": ctx})

    files = {"A": OUT / f"letterA_{tag}.jsonl", "B": OUT / f"shiftB_{tag}.jsonl"}
    done = {j: {r.get("key") for r in jl(files[j])} for j in "AB"}
    nA = sum(1 for r in plan if r["job"] == "A"); nB = len(plan) - nA
    dA = sum(1 for r in plan if r["job"] == "A" and r["key"] in done["A"])
    dB = sum(1 for r in plan if r["job"] == "B" and r["key"] in done["B"])
    todo = [r for r in plan if r["key"] not in done[r["job"]]]
    print(f"[{tag}] lora={lora or '-'} mtp={mtp} jobs={sorted(jobs)} | A {dA}/{nA} | B {dB}/{nB}", flush=True)

    if todo:
        t0 = time.time()
        if llm is None:
            kw = dict(model=model, quantization="fp8", dtype="bfloat16", max_model_len=8192, trust_remote_code=True,
                      gpu_memory_utilization=0.90, max_num_seqs=64, max_logprobs=64,
                      compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                                          "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]})
            if lora:
                kw.update(enable_lora=True, max_lora_rank=256, max_loras=1)
            elif mtp:
                kw.update(speculative_config={"method": "mtp", "num_speculative_tokens": 1}, enable_prefix_caching=False)
            llm = LLM(**kw)
            lreq = LoRARequest(os.path.basename(lora.rstrip("/")), 1, lora) if lora else None
        tok = llm.get_tokenizer()
        lid = {L: tok(" " + L, add_special_tokens=False)["input_ids"][0] for L in LET}
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=60)
        print(f"[{tag}] loaded in {time.time()-t0:.0f}s | scoring {len(todo)} contexts", flush=True)
        fhs = {j: open(files[j], "a") for j in {r["job"] for r in todo}}
        CH = 512
        for i in range(0, len(todo), CH):
            b = todo[i:i + CH]
            outs = llm.generate([r["ctx"] for r in b], sp, lora_request=lreq)
            for r, o in zip(b, outs):
                d = o.outputs[0].logprobs[0]
                lp = {L: round(float(d[lid[L]].logprob), 4) if lid[L] in d else -30.0 for L in LET}
                rec = {k: v for k, v in r.items() if k != "ctx"}; rec["lp"] = lp
                fhs[r["job"]].write(json.dumps(rec) + "\n")
            for f in fhs.values():
                f.flush()
            if (i // CH) % 10 == 0 or i + CH >= len(todo):
                print(f"[{tag}] {min(i+CH, len(todo))}/{len(todo)}", flush=True)
        for f in fhs.values():
            f.close()

    sa = summ_A(files["A"]) if "A" in jobs and files["A"].exists() else {}
    sb = summ_B(files["B"]) if "B" in jobs and files["B"].exists() else {}
    if sa:
        json.dump(sa, open(OUT / f"letterA_{tag}.summary.json", "w"), indent=1)
    if sb:
        json.dump(sb, open(OUT / f"shiftB_{tag}.summary.json", "w"), indent=1)
    show(tag, sa, sb)


def run_with_engine(llm, lora_req, tag, jobs):
    """Entry point for eval_combined.py: reuse an already-built engine/LoRARequest."""
    lora_path = lora_req.lora_path if lora_req else None
    return run_jobs(tag, {j.upper() for j in jobs}, lora=lora_path, llm=llm, lreq=lora_req)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--jobs", default="A,B")
    ap.add_argument("--mtp", action="store_true")
    a = ap.parse_args()
    assert not (a.lora and a.mtp), "--mtp is only allowed without --lora"
    tag = a.tag or ("lora" if a.lora else "base")
    jobs = {j.strip().upper() for j in a.jobs.split(",") if j.strip()}
    run_jobs(tag, jobs, model=a.model, lora=a.lora, mtp=a.mtp)


if __name__ == "__main__":
    main()
