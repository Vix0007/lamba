#!/usr/bin/env python3
"""Parse + validate DS-written prompt sets for Lamba 0.3 (identity + tool follow-through), dedup, split 60/40 by id hash.
Outputs: l03_identity_questions.jsonl, l03_agent_systems.jsonl, l03_persona_controls.jsonl, l03_tool_prompts.jsonl; prints counts + sha256."""
import glob, hashlib, json, os, re
Q = os.path.expanduser("~/ds_queue/out")
VENDOR = re.compile(r"claude|anthropic|openai|chatgpt|\bgpt|gemini|google|qwen|alibaba|llama|meta ai|mistral|deepseek|copilot|cursor|windsurf", re.I)

def parse(pattern):
    rows, bad = [], 0
    for fn in sorted(glob.glob(f"{Q}/{pattern}")):
        for line in open(fn):
            line = line.strip().strip("`")
            if not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad += 1
    return rows, bad

def split(i):  # deterministic 60/40
    return "heldout" if int(hashlib.md5(i.encode()).hexdigest(), 16) % 10 < 4 else "mine"

def norm(t): return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()
def jac(a, b):
    A, B = set(a.split()), set(b.split())
    return len(A & B) / max(1, len(A | B))

def dedup(rows, key="text", thr=0.8):
    out, seen = [], []
    for r in rows:
        n = norm(r[key])
        if any(jac(n, s) > thr for s in seen):
            continue
        seen.append(n); out.append(r)
    return out

def write(name, rows):
    p = os.path.expanduser(f"~/antidoom_data/{name}")
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} rows  sha256 {hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]}")

idq, bad1 = parse("l03_idq_*.md"); n0 = len(idq)
idq = [r for r in idq if r.get("text") and len(r["text"]) > 3]; idq = dedup(idq)
for r in idq: r["split"] = split(r["id"])
print(f"identity questions: parsed {n0}, bad-json {bad1}, after dedup {len(idq)}", {l: sum(r['lang']==l for r in idq) for l in sorted({r['lang'] for r in idq})}, {s: sum(r['split']==s for r in idq) for s in ("mine","heldout")})
write("l03_identity_questions.jsonl", idq)

ag, bad2 = parse("l03_agentsys_*.md")
ok = [r for r in ag if r.get("system") and not VENDOR.search(r["system"])]
print(f"agent system prompts: parsed {len(ag)} bad-json {bad2}, vendor-free {len(ok)}, words/prompt {[len(r['system'].split()) for r in ok]}")
write("l03_agent_systems.jsonl", ok)

pc, bad3 = parse("l03_personactl.md")
pc = [r for r in pc if r.get("system") and r.get("expected_name") and r["expected_name"].lower() in r["system"].lower()]
print(f"persona controls: {len(pc)} valid (expected_name appears in system prompt)")
write("l03_persona_controls.jsonl", pc)

tq, bad4 = parse("l03_toolq_*.md"); n1 = len(tq)
tq = [r for r in tq if r.get("text")]; tq = dedup(tq)
for r in tq: r["split"] = split(r["id"])
print(f"tool prompts: parsed {n1}, bad-json {bad4}, after dedup {len(tq)}", {l: sum(r['lang']==l for r in tq) for l in sorted({r['lang'] for r in tq})}, {s: sum(r['split']==s for r in tq) for s in ("mine","heldout")})
write("l03_tool_prompts.jsonl", tq)
