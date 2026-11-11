"""Parse DS blind final-answer extraction for the Malay merge-round adapter vs base (eval180 subset with gold).
usage: malayfinal_ds_extract.py"""
import json, re, glob, random
A = "/home/ubuntu/antidoom_data"; E = "/home/ubuntu/eval_results"; Q = "/home/ubuntu/ds_queue"
key = json.load(open(f"{A}/malayfinalext_key.json"))
P = {json.loads(l)["id"]: json.loads(l) for l in open(f"{A}/malay_gen_eval180.jsonl")}
lab = {}
for f in sorted(glob.glob(f"{Q}/out/malayfinalext_batch*.md")):
    nb = int(re.search(r"batch(\d+)", f).group(1))
    for m in re.finditer(r'\{"case"\s*:\s*(\d+)\s*,\s*"letter"\s*:\s*"(\w+)"\s*\}', open(f).read()):
        k = key.get(f"{nb}:{m.group(1)}")
        if k: lab[k] = m.group(2)
print("labelled", len(lab), "of", len(key))
ids = list(P)
res = {arm: {i: int(lab.get(f"{arm}|{i}") == P[i]["gold"]) for i in ids if f"{arm}|{i}" in lab} for arm in ("B", "M")}
common = [i for i in ids if i in res["B"] and i in res["M"]]
d = [res["M"][i] - res["B"][i] for i in common]
random.seed(0); bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(5000))
nn = {a: sum(lab.get(f"{a}|{i}") in (None, "NONE") for i in ids) for a in ("B", "M")}
print(f"n={len(common)} acc base {100*sum(res['B'][i] for i in common)/len(common):.1f}% malayfinal {100*sum(res['M'][i] for i in common)/len(common):.1f}% "
      f"delta {100*sum(d)/len(d):+.1f}pp 95%CI [{100*bs[125]:+.1f},{100*bs[4874]:+.1f}] NONE/unlabelled base {nn['B']} malayfinal {nn['M']}")
