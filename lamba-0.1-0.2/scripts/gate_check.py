"""Newline/loop sanity gate. usage: gate_check.py <adapter_label> (reads ~/eval_results/loop_<label>_gate.jsonl vs base texts in loop_base_g150.jsonl, same ids, first 4800 chars).
PASS if newline rate >=80% of base ('\\n\\n' and '\\n') AND mean repeat-6gram fraction <= base+0.03 AND budget hits not worse than base+3."""
import re
import json, statistics as st, sys
lab = sys.argv[1]; R = "/home/ubuntu/eval_results"
a = {}
for l in open(f"{R}/loop_{lab}_gate.jsonl"): r = json.loads(l); a[r["id"]] = r
b = {}
for l in open(f"{R}/loop_base_g150.jsonl"): r = json.loads(l); b[r["id"]] = r
ids = [i for i in a if i in b]
def rep(t):
    w = t.split()
    if len(w) < 50: return 0.0
    sh = [" ".join(w[i:i + 6]) for i in range(len(w) - 5)]; return 1 - len(set(sh)) / len(sh)
T = lambda d, i: d[i]["text"][:4800]
rate = lambda d, s: st.mean(T(d, i).count(s) / max(1, len(T(d, i))) * 1000 for i in ids)
nn_b, nn_a, n1_b, n1_a = rate(b, "\n\n"), rate(a, "\n\n"), rate(b, "\n"), rate(a, "\n")
rp_b, rp_a = st.mean(rep(T(b, i)) for i in ids), st.mean(rep(T(a, i)) for i in ids)
_al = re.compile(r"[A-Za-z0-9]")
def liquid_loop(t, W=60, S=30):
    for i in range(0, len(t) - W, S):
        w = t[i:i + W]
        if len(_al.findall(w)) < 25: continue
        if t.count(w) >= 4: return True
    return False
ntok = st.mean(a[i]["n_tokens"] for i in ids); empty = sum(a[i]["n_tokens"] < 20 for i in ids)
print(f"HEALTH mean_tokens {ntok:.0f} (base {st.mean(b[i]['n_tokens'] for i in ids):.0f} at 4000cap) empty(<20tok) {empty}/{len(ids)} | liquid-loops adapter {sum(liquid_loop(a[i]['text']) for i in ids)}/{len(ids)} base(first4800ch) {sum(liquid_loop(b[i]['text'][:4800]) for i in ids)}/{len(ids)}")
print(f"n={len(ids)} '\\n\\n'/1000ch base {nn_b:.2f} adapter {nn_a:.2f} ({100*nn_a/max(nn_b,1e-9):.0f}%) | '\\n' base {n1_b:.2f} adapter {n1_a:.2f} ({100*n1_a/max(n1_b,1e-9):.0f}%) | repeat-6gram base {rp_b:.3f} adapter {rp_a:.3f}")
ok = nn_a >= 0.8 * nn_b and n1_a >= 0.8 * n1_b and rp_a <= rp_b + 0.03 and empty == 0 and ntok > 300
print("GATE_JSON", json.dumps({"pass": bool(ok), "nn_ratio": nn_a / max(nn_b, 1e-9), "n1_ratio": n1_a / max(n1_b, 1e-9), "rep_b": rp_b, "rep_a": rp_a}))
