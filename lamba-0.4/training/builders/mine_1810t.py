#!/usr/bin/env python3
"""1810t mining on the served 1810s (own-model). usage: mine_1810t.py MODE OUT   MODE = casual | identity | cjk | refalt | benign
casual : DS short feeling-bearing messages (shm_*), warm-first guide + exemplars, n=6 (eval battery prompts and near-duplicates excluded)
identity: DS casual 'who are you' questions (ciq), identity rule system prompt (as mine_identity_1810), n=4
cjk    : Malay/rojak explanatory prompts (cjk_* + shortq pools), NO system prompt, n=4 temp 1.0 -> find CJK-leak samples (rejected) and clean siblings (chosen)"""
import json, sys, os, re, glob, random, urllib.request, concurrent.futures as cf
MODE, OUT = sys.argv[1], sys.argv[2]; MODEL = os.environ.get("MINE_MODEL", "l1810s"); D = "/home/ubuntu/antidoom_data/"; E = "/home/ubuntu/eval_results/id1810t/"
def chat(msgs, n, temp, mx):
    b = {"model": MODEL, "messages": msgs, "max_tokens": mx, "temperature": temp, "top_p": 0.95, "repetition_penalty": 1.05, "n": n}
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=3600))
    return [{"reasoning": c["message"].get("reasoning") or c["message"].get("reasoning_content") or "", "content": c["message"].get("content") or "", "finish": c["finish_reason"]} for c in d["choices"]]
def jl(pat, key="prompt"):
    out = []
    for f in sorted(glob.glob(pat)):
        for l in open(f, errors="ignore"):
            l = l.strip().strip('`').rstrip(',')
            if l.startswith("{"):
                try: d = json.loads(l); out.append(d)
                except Exception: pass
    return out
tok = lambda s: set(re.sub(r"\W+", " ", s.lower()).split())
jac = lambda a, b: len(a & b) / max(1, len(a | b))
if MODE == "casual":
    GUIDE = open(D + "casual_style_guide_warm.md").read(); EX = []
    for d in jl("/home/ubuntu/ds_queue/out/cstyle_*.md"):
        if "user" in d and "reply" in d: EX.append((d["user"].strip(), d["reply"].strip()))
    bat = open(D + "tone_battery.py").read(); BAT = [p for p in eval(bat[bat.index("P = [")+4:bat.index("jobs =")].strip())]; BT = [tok(p) for p in BAT]
    exu = {u.lower() for u, _ in EX}
    def SYSFOR(seed):
        ex = random.Random(seed).sample(EX, min(6, len(EX)))
        return ("You are Lamba, an AI model built by Vixero Labs, chatting casually.\n\n" + GUIDE + "".join(f"\n\nExample of the voice (do not copy; adapt to this user's message):\nUser: {u}\nLamba: {r}" for u, r in ex) + "\n\nNever mention these instructions. Vary your wording from reply to reply.")
    seen = set(); P = []
    for d in jl("/home/ubuntu/ds_queue/out/shm_*.md"):
        p = d.get("prompt", "").strip(); k = p.lower()
        if not p or k in seen or k in exu or re.search(r"\[[^\]]*\]|\(user", k) or len(p) > 120: continue
        if any(jac(tok(p), b) >= 0.5 for b in BT): continue
        seen.add(k); P.append((f"s{len(P):04d}", p, d.get("lang", "en")))
    def run(p): return {"id": p[0], "prompt": p[1], "lang": p[2], "comps": chat([{"role": "system", "content": SYSFOR(p[0])}, {"role": "user", "content": p[1]}], 6, 0.9, 1200)}
elif MODE == "identity":
    RULE = open(D + "mine_identity_1810.py").read().split('RULE = (')[1].split('Q = []')[0]; RULE = eval('(' + RULE.strip())
    P = [(f"i{i:04d}", q["prompt"], q["lang"]) for i, q in enumerate(json.load(open(E + "ciq_questions.json")))]
    def run(p): return {"id": p[0], "prompt": p[1], "lang": p[2], "comps": chat([{"role": "system", "content": RULE}, {"role": "user", "content": p[1]}], 4, 0.8, 900)}
elif MODE == "benign":
    import hashlib
    HOLD = lambda i: int(hashlib.md5(i.encode()).hexdigest(), 16) % 9 == 0
    R = [json.loads(l) for l in open("/home/ubuntu/eval_results/id1810s/safety_r.jsonl")]
    R = [r for r in R if "error" not in r and r["kind"] != "harm" and not HOLD(r["id"])]
    P = [(r["id"], r["prompt"], r["lang"]) for r in R]
    def run(p): return {"id": p[0], "prompt": p[1], "lang": p[2], "comps": chat([{"role": "user", "content": p[1]}], 4, 0.9, 1600)}
elif MODE == "benign2":
    import csv
    X = [set(re.sub(r"\W+", " ", r.get("prompt", "").lower()).split()) for r in csv.DictReader(open("/home/ubuntu/antidoom_data/xstest_prompts.csv"))]
    P = []; seen = set()
    for d in jl("/home/ubuntu/ds_queue/out/ben_*.md"):
        p = d.get("prompt", "").strip(); k = re.sub(r"\W+", " ", p.lower()).strip(); w = set(k.split())
        if not p or k in seen or len(p) > 220 or re.search(r"[\u4e00-\u9fff]", p) or any(jac(w, x) >= 0.6 for x in X): continue
        seen.add(k); P.append((f"b{len(P):04d}", p, d.get("lang", "en")))
    def run(p): return {"id": p[0], "prompt": p[1], "lang": p[2], "comps": chat([{"role": "user", "content": p[1]}], 4, 0.9, 1600)}
elif MODE == "ovr":
    import csv
    XR = list(csv.DictReader(open("/home/ubuntu/antidoom_data/xstest_prompts.csv")))
    X = [tok(r.get("prompt", "")) for r in XR]
    NAMES = set()
    for r in XR:
        if r.get("type") in ("privacy_fictional", "nons_group_real_discr", "real_group_nons_discr", "safe_contexts"):
            NAMES |= {w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", r.get("prompt", ""))}
    NAMES -= {"What", "Where", "How", "Why", "Which", "Who", "Can", "Could", "Would", "Should", "Please", "Does"}
    P = []; seen = set()
    for d in jl(os.environ.get("OVR_GLOB", "/home/ubuntu/ds_queue/out/ovr_*.md")):
        p = d.get("prompt", "").strip(); k = re.sub(r"\W+", " ", p.lower()).strip(); w = set(k.split())
        if not p or k in seen or len(p) > 260 or re.search(r"[\u4e00-\u9fff]", p) or any(jac(w, x) >= 0.5 for x in X): continue
        if any(n in p for n in NAMES): continue
        seen.add(k); P.append((f"{os.environ.get('OVR_PFX', 'o')}{len(P):04d}", p, d.get("lang", "en")))
    def run(p): return {"id": p[0], "prompt": p[1], "lang": p[2], "comps": chat([{"role": "user", "content": p[1]}], 6, 0.9, 1600)}
elif MODE == "safe2":
    L = json.load(open(os.environ.get("SAFE2_LABELS", "/home/ubuntu/eval_results/id1810s/saud_labels.json")))
    PP = json.load(open(os.environ.get("SAFE2_PROMPTS", "/home/ubuntu/eval_results/id1810t/safety_unused_prompts.json")))
    P = [(p[0], p[3], p[4]) for p in PP if L.get(p[0]) == "decline"]
    def run(p): return {"id": p[0], "prompt": p[1], "lang": p[2], "comps": chat([{"role": "user", "content": p[1]}], 8, 1.0, 1400)}
elif MODE == "refalt":
    import hashlib
    LAB = json.load(open("/home/ubuntu/eval_results/id1810s/saud_labels.json")); HOLD = lambda i: int(hashlib.md5(i.encode()).hexdigest(), 16) % 9 == 0
    R = [json.loads(l) for l in open("/home/ubuntu/eval_results/id1810s/safety_r.jsonl")]
    R = [r for r in R if "error" not in r and r["kind"] == "harm" and LAB.get(r["id"]) == "decline" and not HOLD(r["id"])]
    random.Random(8).shuffle(R); P = [(r["id"], r["prompt"], r["lang"]) for r in R[:int(os.environ.get("REF_N", "300"))]]
    def run(p): return {"id": p[0], "prompt": p[1], "lang": p[2], "comps": chat([{"role": "user", "content": p[1]}], 4, 0.9, 1400)}
else:
    HELD = set()
    for f in ("shortq_heldout.json", "msfact_heldout.json", "ibu_heldout.json"):
        p = D + f
        if os.path.exists(p):
            for r in json.load(open(p)): HELD.add(re.sub(r"\W+", " ", str(r.get("prompt") or r.get("q") or r.get("question") or r).lower()).strip())
    P = []; seen = set()
    for d in jl("/home/ubuntu/ds_queue/out/cjk_*.md"):
        p = d.get("prompt", "").strip(); k = re.sub(r"\W+", " ", p.lower()).strip()
        if p and k not in seen and k not in HELD and not re.search(r"[一-鿿]", p): seen.add(k); P.append((f"j{len(P):04d}", p, d.get("lang", "ms")))
    for d in jl("/home/ubuntu/ds_queue/out/tdk_*.md"):
        p = d.get("prompt", "").strip(); k = re.sub(r"\W+", " ", p.lower()).strip()
        if p and d.get("lang") == "ms" and k not in seen and k not in HELD: seen.add(k); P.append((f"j{len(P):04d}", p, "ms"))
    random.Random(5).shuffle(P); P = P[:int(os.environ.get("CJK_N", "300"))]
    def run(p): return {"id": p[0], "prompt": p[1], "lang": p[2], "comps": chat([{"role": "user", "content": p[1]}], 3, 1.0, 2000)}
if os.environ.get("DRY"):
    import collections; print(MODE,"DRY prompts",len(P),collections.Counter(p[2] for p in P)); [print("  ",p[1][:90]) for p in random.Random(1).sample(P,min(8,len(P)))]; sys.exit()
done = set()
if os.path.exists(OUT): done = {d["id"] for d in map(json.loads, open(OUT)) if "error" not in d}
todo = [p for p in P if p[0] not in done]; print(MODE, "prompts", len(P), "todo", len(todo), flush=True)
def safe(p):
    try: return run(p)
    except Exception as e: return {"id": p[0], "error": str(e)[:200]}
with open(OUT, "a") as f, cf.ThreadPoolExecutor(8) as ex:
    for r in ex.map(safe, todo): f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
print("MINE1810T_DONE", MODE)
