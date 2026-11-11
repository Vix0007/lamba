"""usage: parse_labels.py <out_prefix> -> reads ds_queue/out/<name>_batchNN.md answers, writes <out_prefix>_labels.jsonl {id,type,start_char_full,confidence}. Full-text offset = tail_start + start_char."""
import glob, json, re, sys
pre = sys.argv[1]; idx = json.load(open(f"{pre}_index.json")); out = []
import os
base = os.path.basename(pre)
for f in sorted(glob.glob(f"/home/ubuntu/ds_queue/out/{base}_batch*.md")):
    nb = int(re.search(r"batch(\d+)", f).group(1)); t = open(f).read()
    for m in re.finditer(r'\{"case"\s*:\s*(\d+)\s*,\s*"type"\s*:\s*"(\w+)"\s*,\s*"start_char"\s*:\s*(null|\d+)\s*,\s*"confidence"\s*:\s*([\d\.]+)\s*\}', t):
        k = int(m.group(1)); cid = next((c for c, v in idx.items() if v["batch"] == nb and v["case_in_batch"] == k), None)
        if cid is None: continue
        v = idx[cid]; sc = None if m.group(3) == "null" else v["tail_start"] + int(m.group(3))
        out.append({"id": v["id"], "type": m.group(2), "start_char_full": sc, "confidence": float(m.group(4))})
open(f"{pre}_labels.jsonl", "w").write("\n".join(json.dumps(r) for r in out) + "\n")
import collections; print("labels", len(out), dict(collections.Counter(r["type"] for r in out)))
