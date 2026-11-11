#!/usr/bin/env python3
"""Queue DS grading for distill candidates not yet queued. usage: queue_distill_grades.py TAG"""
import json, os, subprocess, sys
E = "/home/ubuntu/eval_results/id1810/"; QF = E + "queued_cids.json"
q = set(json.load(open(QF))) if os.path.exists(QF) else set()
G = [json.loads(l) for l in open(E + "grade_list_1810.jsonl")]
new = [g for g in G if g["cid"].startswith("di|") and g["cid"] not in q]
open(E + f"grade_list_{sys.argv[1]}.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in new))
q |= {g["cid"] for g in new}; json.dump(sorted(q), open(QF, "w"))
print("queued", len(new), "items")
subprocess.run(["python3", "/home/ubuntu/antidoom_data/grade_1810.py", "build", E + f"grade_list_{sys.argv[1]}.jsonl", sys.argv[1]])
