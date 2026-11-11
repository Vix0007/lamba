"""Enumeration detector (DS 85 sec.3). usage: line_template_detector.py <rollouts.jsonl> [--labels r3_tails_labels_dedup.jsonl]
Flags outputs where a window of 20 lines has >=16 adjacent similar templates, >=15 lines share the top template,
>=70% contain an integer, integers monotone over >=10 lines, after ~1500 tokens (~5000 chars)."""
import json, re, sys, collections
def norm(l):
    l = l.lower(); l = re.sub(r"[a-z]+\d+", "<id><n>", l); l = re.sub(r"\d+(\.\d+)?", "<n>", l); return re.sub(r"[\W_]+", " ", l.replace("<n>", "N").replace("<id>", "I")).strip()
def jac(a, b):
    A, B = set(a), set(b); return len(A & B) / max(1, len(A | B))
def detect(text, min_chars=5000):
    lines = [(m.start(), m.group()) for m in re.finditer(r"[^\n]+", text) if len(m.group().strip()) >= 4]
    T = [norm(l) for _, l in lines]
    for i in range(len(lines) - 19):
        if lines[i][0] < min_chars: continue
        w = T[i:i+20]; adj = sum(jac(w[j], w[j+1]) >= 0.8 for j in range(19))
        if adj < 16: continue
        top, c = collections.Counter(w).most_common(1)[0]
        if c < 15: continue
        nums = [re.findall(r"\d+", lines[i+j][1]) for j in range(20)]
        if sum(bool(n) for n in nums) < 14: continue
        first = [int(n[0]) for n in nums if n]
        inc = sum(b > a for a, b in zip(first, first[1:])); 
        if inc >= 10: return lines[i][0]
    return None
if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1])]
    hits = {r["id"]: detect(r["text"]) for r in rows}; hit = {k: v for k, v in hits.items() if v is not None}
    print(f"{len(hit)}/{len(rows)} flagged ({100*len(hit)/len(rows):.1f}%)  budget-hit rows: {sum(r['finish_reason']=='length' for r in rows)}")
    if "--labels" in sys.argv:
        lab = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[sys.argv.index("--labels")+1])}
        c = collections.Counter(lab[k]["type"] if k in lab else "unlabelled" for k in hit); print("flagged by DS label:", dict(c))
        enum = [k for k, v in lab.items() if v["type"] == "enumeration"]; print("DS-enumeration recovered:", sum(k in hit for k in enum), "/", len(enum))
    json.dump(hit, open(sys.argv[1].replace(".jsonl", "_enum_flags.json"), "w"))
