import json, re, collections, statistics as st, sys
sys.path.insert(0, "/home/ubuntu/antidoom_data")
R = "/home/ubuntu/eval_results"
P = {json.loads(l)["id"]: json.loads(l) for l in open("loop_testset150.jsonl")}
def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)}
B, V = load(f"{R}/loop_base_g150.jsonl"), load(f"{R}/loop_v9bfinal_g150.jsonl")
ids = [i for i in P if P[i]["domain"] == "mcp" and i in B and i in V]
al = re.compile(r"[A-Za-z0-9]")
def tail_loop(text, pmin=15):
    n = len(text)
    if n < 300: return False
    tail = text[-6000:]; m = len(tail)
    for p in range(pmin, min(1200, m // 2) + 1):
        if tail[m - p:] != tail[m - 2 * p:m - p]: continue
        i = m - p - 1
        while i >= 0 and tail[i] == tail[i + p]: i -= 1
        s = i + 1
        if m - s >= 2 * p - 5 and len(tail[s:s + p].strip()) >= 10 and not re.fullmatch(r"[\W\d_]+", tail[s:s + p]): return True
    return False
def heavy(t, prompt):
    n = len(t); cov = bytearray(n); seen = set()
    for i in range(0, n - 60, 30):
        w = t[i:i + 60]
        if w in seen or len(al.findall(w)) < 25: continue
        seen.add(w)
        if t.count(w) < 4 or w in prompt: continue
        j = t.find(w)
        while j >= 0:
            for k in range(j, min(n, j + 60)): cov[k] = 1
            j = t.find(w, j + 60)
    return sum(cov[max(0, n - 2500):])
def maxrep5(t):
    w = t.split(); c = collections.Counter(tuple(w[i:i + 5]) for i in range(max(0, len(w) - 4))); return max(c.values()) if c else 0
def d3(t):
    w = t.split(); g = [tuple(w[i:i + 3]) for i in range(max(0, len(w) - 2))]; return len(set(g)) / max(1, len(g))
for name, D in (("base", B), ("v9b", V)):
    hv = [heavy(D[i]["text"], P[i]["prompt"]) for i in ids]
    print(f"{name}: n={len(ids)} exact(contiguous-tail) {sum(tail_loop(D[i]['text']) for i in ids)} | R_heavy(>1500) {sum(h>1500 for h in hv)} | R_any(tail>0) {sum(h>0 for h in hv)} | budget-hit {sum(D[i]['finish_reason']=='length' for i in ids)} | median max-repeated-5gram {st.median(maxrep5(D[i]['text']) for i in ids)} | mean distinct-3 {st.mean(d3(D[i]['text']) for i in ids):.3f} | mean tokens {st.mean(D[i]['n_tokens'] for i in ids):.0f}")
bo = [i for i in ids if V[i]["finish_reason"] == "length" and heavy(V[i]["text"], P[i]["prompt"]) == 0][:10]
print("v9b budget-hit mcp outputs WITHOUT tail repetition:", len([i for i in ids if V[i]["finish_reason"]=="length" and heavy(V[i]["text"],P[i]["prompt"])==0]), "of", sum(V[i]["finish_reason"]=="length" for i in ids))
out = ["Classify each trace tail (last 1500 chars of a 4000-token generation that did NOT conclude) as LOOP (repeating itself and would continue forever) or LEGIT (still making progress; just long reasoning). Answer JSON lines {\"case\":n,\"label\":\"LOOP\"|\"LEGIT\"} after brief thinking (<=1 line each).\n"]
for k, i in enumerate(bo, 1): out.append(f"\n=== CASE {k} ===\n{V[i]['text'][-1500:]}\n")
open("/home/ubuntu/ds_queue/81_mcp_budget_traces.txt", "w").write("".join(out)); print("traces written", len(bo))
