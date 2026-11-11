"""Compare base vs adapter on the held-out loop set: exact + semantic loops, conclusion rate, paired counts."""
import json, math, re, sys
sys.path.insert(0, "/home/ubuntu/antidoom/src")
import numpy as np, torch
from antidoom.repetition import find_inner_repetition
tag = open("/home/ubuntu/eval_results/loop_engine_tag.txt").read().strip() if len(sys.argv) < 2 else sys.argv[1]
R = lambda lab: {json.loads(l)["id"]: json.loads(l) for l in open(f"/home/ubuntu/eval_results/loop_{lab}_{tag}.jsonl")}
rows = {"base": R("base"), "ftpo1": R("ftpo1")}
ids = sorted(set(rows["base"]) & set(rows["ftpo1"])); print("engine tag:", tag, "| paired prompts:", len(ids))
from transformers import AutoTokenizer, AutoModel
dev = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 4e9 else "cpu"
tk = AutoTokenizer.from_pretrained("BAAI/bge-m3"); m = AutoModel.from_pretrained("BAAI/bge-m3", dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
def sem_loop(text, thr=0.88, gap=3):
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if len(x.strip()) > 15]
    if len(s) < gap + 2: return False
    with torch.no_grad():
        v = []
        for i in range(0, len(s), 64):
            e = tk(s[i:i+64], padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
            v.append(torch.nn.functional.normalize(m(**e).last_hidden_state[:, 0], dim=-1).float().cpu())
        v = torch.cat(v); sim = (v @ v.T).numpy()
    return bool(np.where(np.triu(np.ones_like(sim, dtype=bool), k=gap), sim, -1).max() > thr)
res = {a: {} for a in rows}
for a in rows:
    for i in ids:
        t = rows[a][i]["text"]; res[a][i] = {"exact": bool(find_inner_repetition(t)[0]), "sem": sem_loop(t),
            "length": rows[a][i]["finish_reason"] == "length", "tok": rows[a][i]["n_tokens"]}
for a in rows:
    n = len(ids); g = lambda k: sum(res[a][i][k] for i in ids)
    anyl = sum(res[a][i]["exact"] or res[a][i]["sem"] for i in ids)
    print(f"{a:6s} exact_loops {g('exact')}/{n}  semantic_loops {g('sem')}/{n}  any_loop {anyl}/{n}  "
          f"hit_budget(no conclusion) {g('length')}/{n}  mean_tokens {np.mean([res[a][i]['tok'] for i in ids]):.0f}")
def mcnemar(k):
    b = sum(res['base'][i][k] and not res['ftpo1'][i][k] for i in ids); c = sum(res['ftpo1'][i][k] and not res['base'][i][k] for i in ids)
    n = b + c; p = min(1.0, 2 * sum(math.comb(n, j) for j in range(0, min(b, c) + 1)) / 2**n) if n else 1.0
    return b, c, p
for k, name in (("exact", "exact loop"), ("sem", "semantic loop"), ("length", "hit budget")):
    b, c, p = mcnemar(k); print(f"paired {name}: base-only {b}, adapter-only {c}, exact McNemar p={p:.3f}")
for d in ("mcp", "science", "malay"):
    sub = [i for i in ids if rows["base"][i]["domain"] == d]
    print(f"  {d:8s} n={len(sub):2d} base_loops={sum(res['base'][i]['exact'] or res['base'][i]['sem'] for i in sub)} adapter_loops={sum(res['ftpo1'][i]['exact'] or res['ftpo1'][i]['sem'] for i in sub)}")
