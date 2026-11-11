"""Malay generalisation analysis: tail_stats_cmp (loops) + final-answer accuracy with paired bootstrap. usage: malay_gen_analyze.py"""
import json, re, random, subprocess, sys
E = "/home/ubuntu/eval_results"; P = "/home/ubuntu/antidoom_data/malay_gen_eval180.jsonl"
gold = {json.loads(l)["id"]: json.loads(l)["gold"] for l in open(P)}
def load(f): return {json.loads(l)["id"]: json.loads(l) for l in open(f)}
b = load(f"{E}/loop_base_malaygen.jsonl"); a = load(f"{E}/loop_v9cfinal_malaygen.jsonl")
def pred(t):
    tail = t[-800:]
    m = re.findall(r"\\boxed\{\s*(?:\\text\{)?\s*\(?([A-E])\)?\s*\}?\s*\}", tail)
    if not m: m = re.findall(r"(?i)(?:jawapan|answer)[^A-Za-z\n]{0,4}(?:yang\s+)?(?:betul|tepat|paling tepat|sebenar)?[^A-E\n]{0,25}?\*{0,2}\(?([A-E])[\.\)\s\*:]", tail)
    return m[-1] if m else None
ids = [i for i in gold if i in a and i in b]
cb = [int(pred(b[i]["text"]) == gold[i]) for i in ids]; ca = [int(pred(a[i]["text"]) == gold[i]) for i in ids]
pb = sum(pred(b[i]["text"]) is not None for i in ids) / len(ids); pa = sum(pred(a[i]["text"]) is not None for i in ids) / len(ids)
d = [y - x for x, y in zip(cb, ca)]; random.seed(0)
bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(5000))
print(f"n={len(ids)} acc base {100*sum(cb)/len(ids):.1f}% v9c {100*sum(ca)/len(ids):.1f}% delta {100*sum(d)/len(d):+.1f}pp 95%CI [{100*bs[125]:+.1f},{100*bs[4874]:+.1f}] parse-rate base {100*pb:.0f}% v9c {100*pa:.0f}%")
print(subprocess.run(["/home/ubuntu/antidoom/.venv/bin/python", "/home/ubuntu/antidoom_data/tail_stats_cmp.py", f"{E}/loop_base_malaygen.jsonl", f"{E}/loop_v9cfinal_malaygen.jsonl", P], capture_output=True, text=True).stdout)
