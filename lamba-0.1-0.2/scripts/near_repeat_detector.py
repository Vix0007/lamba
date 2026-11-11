"""Near-verbatim / paraphrase repeat detector (tail). A text is 'near-loop' if, among the last NSEG normalised segments (sentences/lines >=25 chars) inside the last 4000 chars,
at least K have a 3-word-shingle Jaccard >= TH with some EARLIER segment inside the last 4000 chars. usage as module: near_loop(text)->(bool, frac); CLI validates on the 10 DS-labelled mcp traces."""
import json, re, sys
NSEG, K, TH = 25, 6, 0.6
def segs(t):
    out = []
    for s in re.split(r"(?<=[\.\?\!:;])\s+|\n+", t):
        n = re.sub(r"[^a-z\s]", " ", s.lower()); n = re.sub(r"\s+", " ", n).strip()
        if len(n) >= 12: out.append(n)
    return out
def sh(s):
    w = s.split(); return {tuple(w[i:i + 3]) for i in range(max(1, len(w) - 2))}
def near_loop(text, nseg=NSEG, k=K, th=TH):
    S = segs(text[-4000:]); 
    if len(S) < 8: return False, 0.0
    H = [sh(s) for s in S]; hit = 0; tot = 0
    for i in range(max(1, len(S) - nseg), len(S)):
        best = 0.0
        for j in range(i):
            a, b = H[i], H[j]
            if not a or not b: continue
            inter = len(a & b)
            if inter: best = max(best, inter / len(a | b))
            if best >= th: break
        hit += best >= th; tot += 1
    return hit >= k, hit / max(1, tot)
RUM = re.compile(r"\b(wait|hmm|let me (re)?think|let me reconsider|that's not (quite )?right|actually,? (that|this)|let me re-?read|on second thought)\b", re.I)
def rumination(text, n=3000):
    t = text[-n:]; return len(RUM.findall(t)) / max(1, len(t)) * 1000
if __name__ == "__main__":
    R = "/home/ubuntu/eval_results"
    P = {json.loads(l)["id"]: json.loads(l) for l in open("/home/ubuntu/antidoom_data/loop_testset150.jsonl")}
    V = {json.loads(l)["id"]: json.loads(l) for l in open(f"{R}/loop_v9bfinal_g150.jsonl")}
    ids = [i for i in P if P[i]["domain"] == "mcp" and i in V]
    sys.path.insert(0, "/home/ubuntu/antidoom_data")
    import importlib.util
    spec = importlib.util.spec_from_file_location("md", "/home/ubuntu/antidoom_data/mcp_diag.py")
    # reproduce the 10 traces: v9b budget hits with no exact tail repetition, in mcp id order
    al = re.compile(r"[A-Za-z0-9]")
    def heavy(t, prompt):
        n = len(t); cov = bytearray(n); seen = set()
        for i in range(0, n - 60, 30):
            w = t[i:i + 60]
            if w in seen or len(al.findall(w)) < 25: continue
            seen.add(w)
            if t.count(w) < 4 or w in prompt: continue
            j = t.find(w)
            while j >= 0:
                for kk in range(j, min(n, j + 60)): cov[kk] = 1
                j = t.find(w, j + 60)
        return sum(cov[max(0, n - 2500):])
    bo = [i for i in ids if V[i]["finish_reason"] == "length" and heavy(V[i]["text"], P[i]["prompt"]) == 0][:10]
    labels = {5: 1, 6: 1, 7: 1}
    for k_, i in enumerate(bo, 1):
        f, fr = near_loop(V[i]["text"]); print(f"case {k_} DS={'LOOP' if labels.get(k_) else 'LEGIT'} near_loop={f} frac={fr:.2f} rumination/1k={rumination(V[i]['text']):.1f}")
