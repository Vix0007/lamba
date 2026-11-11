"""DS final-answer extraction for the Malay generalisation eval. build: writes ds_queue/malaygen_ext_batchNN.txt (arm hidden, shuffled);
parse: reads ds_queue/out/malaygen_ext_batchNN.md -> malaygen_ext_labels.json {key: letter|NONE}; then accuracy vs gold.
usage: malay_gen_ds_extract.py build | parse"""
import json, re, sys, random, glob, os
E = "/home/ubuntu/eval_results"; Q = "/home/ubuntu/ds_queue"; A = "/home/ubuntu/antidoom_data"
P = {json.loads(l)["id"]: json.loads(l) for l in open(f"{A}/malay_gen_eval180.jsonl")}
def load(f): return {json.loads(l)["id"]: json.loads(l) for l in open(f)}
arms = {"B": load(f"{E}/loop_base_malaygen.jsonl"), "V": load(f"{E}/loop_v9cfinal_malaygen.jsonl")}
if sys.argv[1] == "build":
    items = [(f"{arm}|{i}", i, arms[arm][i]["text"]) for arm in arms for i in P if i in arms[arm]]
    random.seed(1); random.shuffle(items); key = {}
    for b in range(0, len(items), 30):
        out = ["Below are 30 model responses to multiple-choice questions (options A-E). For each case, read the QUESTION OPTIONS and the END of the response, and output which option letter the response's FINAL answer commits to. If the response is cut off or gives no final commitment, output NONE. Do NOT judge correctness. Reply with one JSON line per case: {\"case\": N, \"letter\": \"A\"|\"B\"|\"C\"|\"D\"|\"E\"|\"NONE\"} and nothing else (think briefly).\n"]
        for n, (k, i, t) in enumerate(items[b:b+30], 1):
            opts = "\n".join(l for l in P[i]["prompt"].splitlines() if re.match(r"^\s*[A-E][\.\)]\s", l))
            out.append(f"### case {n}\nOPTIONS:\n{opts}\nRESPONSE END:\n{t[-900:]}\n"); key[f"{b//30+1}:{n}"] = k
        open(f"{Q}/malaygen_ext_batch{b//30+1:02d}.txt", "w").write("\n".join(out))
    json.dump(key, open(f"{A}/malaygen_ext_key.json", "w")); print(len(items), "items", len(key) // 30 + 1, "batches")
else:
    key = json.load(open(f"{A}/malaygen_ext_key.json")); lab = {}
    for f in sorted(glob.glob(f"{Q}/out/malaygen_ext_batch*.md")):
        nb = int(re.search(r"batch(\d+)", f).group(1))
        for m in re.finditer(r'\{"case"\s*:\s*(\d+)\s*,\s*"letter"\s*:\s*"(\w+)"\s*\}', open(f).read()):
            k = key.get(f"{nb}:{m.group(1)}")
            if k: lab[k] = m.group(2)
    print("labelled", len(lab), "of", len(key))
    import random as R; ids = list(P); res = {}
    for arm in arms: res[arm] = {i: int(lab.get(f"{arm}|{i}") == P[i]["gold"]) for i in ids}
    d = [res["V"][i] - res["B"][i] for i in ids]; R.seed(0); bs = sorted(sum(R.choice(d) for _ in d) / len(d) for _ in range(5000))
    nn = {a: sum(lab.get(f"{a}|{i}") in (None, "NONE") for i in ids) for a in arms}
    print(f"n={len(ids)} acc base {100*sum(res['B'].values())/len(ids):.1f}% v9c {100*sum(res['V'].values())/len(ids):.1f}% delta {100*sum(d)/len(d):+.1f}pp 95%CI [{100*bs[125]:+.1f},{100*bs[4874]:+.1f}] NONE/unlabelled base {nn['B']} v9c {nn['V']}")
