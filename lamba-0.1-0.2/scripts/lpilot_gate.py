"""Gate for launching Round 2: parse ~/lpilot.log. Prints PASS or FAIL:<reason>; exit code 0 = PASS."""
import ast, re, sys
LOG = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/lpilot.log"
t = open(LOG, errors="ignore").read().replace("\r", "\n")
vals = [(int(m.group(1)), ast.literal_eval(m.group(2))) for m in re.finditer(r"LETTER-VAL step (\d+): (\{.*?\})", t)]
if "TRAIN_EXIT 0" not in t or len(vals) < 3 or "nan" in t.lower().split("train_loss")[-1][:200]:
    print("FAIL:no clean finish/too few evals", len(vals), "exit ok" if "TRAIN_EXIT 0" in t else "no exit0"); sys.exit(1)
s0, m0 = vals[0]; sl, ml = vals[-1]
gap = lambda m: m.get("letter_val_gap_A_minus_B", 0.0)
checks = {"acc": ml["letter_val_acc"] >= m0["letter_val_acc"] - 1.0,
          "bias": (ml["letter_val_rstd"] <= m0["letter_val_rstd"] + 1.0) or (abs(gap(ml)) <= abs(gap(m0))),
          "consistency": ml["letter_val_consistency"] >= m0["letter_val_consistency"] - 3.0}
print(f"step {s0}->{sl} acc {m0['letter_val_acc']:.1f}->{ml['letter_val_acc']:.1f} rstd {m0['letter_val_rstd']:.2f}->{ml['letter_val_rstd']:.2f} gapAB {gap(m0):.1f}->{gap(ml):.1f} cons {m0['letter_val_consistency']:.1f}->{ml['letter_val_consistency']:.1f} checks {checks}")
import json
harm = (ml["letter_val_acc"] < m0["letter_val_acc"] - 2.0) or (ml["letter_val_consistency"] < m0["letter_val_consistency"] - 5.0)
score = (ml["letter_val_consistency"] - m0["letter_val_consistency"]) + (m0["letter_val_rstd"] - ml["letter_val_rstd"]) + (abs(gap(m0)) - abs(gap(ml))) * 0.5 + (ml["letter_val_acc"] - m0["letter_val_acc"])
print("GATE_JSON", json.dumps({"log": LOG, "harm": harm, "score": round(score, 3), "pass": all(checks.values())}))
ok = all(checks.values()); print("PASS" if ok else "FAIL:" + ",".join(k for k, v in checks.items() if not v)); sys.exit(0 if ok else 1)
