#!/usr/bin/env python3
"""v40s-clean (0.5 root-cause test): v40s data with false/confusing identity targets removed + fix05 contrast rows (base answers ABOUT other
models, raw + after 'are you qwen?') + self-fact rows. Same config as v40s (fresh from base). usage: build_v40s_clean.py [--dry]"""
import json, re, sys, random, collections
D = "/home/ubuntu/antidoom_data/"
BAD = {
 "yes_to_are_you_x": re.compile(r"^\W*(Yes|Ya|Betul)\b[^.\n]{0,40}\bLamba", re.I),
 "false_origin": re.compile(r"own (models|infrastructure|architecture)|from scratch|in-house|trained (entirely|independently)|not (based on|built on|derived from)|wasn'?t built (up)?on|independent(ly)? (developed|built)|independent AI team", re.I),
 "location": re.compile(r"Singapore|based in (Malaysia|KL)|beribu pejabat", re.I),
 "url": re.compile(r"vixero(labs)?\.(com|ai|io|my)|www\.\w+\.\w+", re.I),
 "cjk": re.compile(r"[一-鿿]"),
 "emoji_identity": re.compile("[\U0001F300-\U0001FAFF☀-➿]"),
 "product_claim": re.compile(r"coding agent|IDE plugin|agent platform|terminal chat", re.I),
}
st = collections.Counter(); keep = []
for r in map(json.loads, open(D + "l03_training_data_v40s.jsonl")):
    t = str(r.get("response_text") or r.get("chosen") or ""); a = t.split("</think>")[-1]
    if re.search(r"Lamba|Vixero", a):
        hit = [k for k, rx in BAD.items() if rx.search(a)]
        if hit: st["drop_" + hit[0]] += 1; continue
    keep.append(r)
new = [r for r in map(json.loads, open(D + "l03_training_data_fix05.jsonl")) if str(r.get("pilot_id", "")).startswith("fix05|")]
for r in new: r["kl_anchor"] = "ref"; r["sft_weight"] = 1.0; st["add_" + r["domain"]] += 1
out = keep + new; random.Random(55).shuffle(out)
if "--dry" not in sys.argv: open(D + "l03_training_data_v40s_clean.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
for k in sorted(st): print(f"{k:28s} {st[k]}")
print("TOTAL", len(out), "kept v40s", len(keep), "added", len(new))
