#!/usr/bin/env python3
"""Score diag05 battery files. usage: diag05_score.py label=file.jsonl ..."""
import json, re, sys, collections
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]")
SWAP = re.compile(r"(Google|Anthropic|Alibaba|Moonshot|DeepSeek|Zhipu|OpenAI|Meta|Mistral|Chinese)[^.\n]{0,30}\bLamba\b|\bLamba\b[^.\n]{0,30}(Google|Anthropic|Alibaba|Moonshot|Zhipu|OpenAI|Meta|Chinese)|Lamba is Lamba|Lamba's Lamba|Lamba \(Vixero Labs\)[^.\n]{0,20}(Chinese|lab|company)", re.I)
INVENT = re.compile(r"Lamba[- ]?(v?\d|Base|\d+B)|based on Lamba|berasaskan Lamba|built on Lamba", re.I)
def score(path):
    R = [json.loads(l) for l in open(path)]
    R = [r for r in R if "error" not in r]
    out = {}
    g = collections.defaultdict(list)
    for r in R:
        a = r["answer"]
        if r["kind"] == "names":
            hit = all(any(n.lower() in a.lower() for n in grp) for grp in r["exp"])
            g[("names", r["cond"])].append((hit, bool(SWAP.search(a)), a.count("Lamba"), a.strip()=="" ))
            g[("names_lang", r["lang"])].append((hit, bool(SWAP.search(a)), a.count("Lamba"), a.strip()==""))
        elif r["kind"] == "self":
            q = "qwen" in a.lower(); inv = bool(INVENT.search(a))
            g[("self",)].append((q, inv, a.strip()==""))
        else:
            em = len(EMOJI.findall(a)); endq = a.rstrip().rstrip("🙂😊😄 ").endswith("?")
            g[("style",)].append((em, endq, "lamba" in a.lower(), len(a)))
    for k, v in g.items():
        if k[0] in ("names","names_lang"):
            n=len(v); out["/".join(k)] = f"correct {100*sum(x[0] for x in v)/n:.0f}%  swap {100*sum(x[1] for x in v)/n:.0f}%  Lamba/ans {sum(x[2] for x in v)/n:.2f}  empty {sum(x[3] for x in v)} (n={n})"
        elif k[0]=="self":
            n=len(v); out["self"] = f"says Qwen {sum(x[0] for x in v)}/{n}  invents base {sum(x[1] for x in v)}  empty {sum(x[2] for x in v)}"
        else:
            n=len(v); out["style"] = f"emoji/reply {sum(x[0] for x in v)/n:.2f}  ends with ? {100*sum(x[1] for x in v)/n:.0f}%  mentions Lamba {100*sum(x[2] for x in v)/n:.0f}%  chars {sum(x[3] for x in v)/n:.0f} (n={n})"
    return out
for arg in sys.argv[1:]:
    lab, path = arg.split("=", 1)
    try: o = score(path)
    except FileNotFoundError: continue
    print(f"== {lab}"); [print(f"  {k:22s} {v}") for k, v in sorted(o.items())]
