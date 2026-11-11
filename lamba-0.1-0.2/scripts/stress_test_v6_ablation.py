#!/usr/bin/env python3
"""DS #166's suggested decisive follow-up: isolate whether the V6 empty-final pattern is
caused by the FAKE tool_call ids/arguments in the synthetic injection, or by the V6
wording/structure itself (3x repeated "Exact/Verified/Confirmed" tool outputs).

Method: pull REAL organic transcripts from tonight's actual s1337 rollouts on the SAME
two tasks that failed (A3_08v1 = ST_t2, A3_08v2 = ST_t3). Take the first 3 real
assistant+tool_call turns (REAL tool_call ids and REAL arguments the model actually
generated) but REPLACE only the tool RESULT content with the V6 wording (repeated exact
value, "Exact:/Verified numerically:/Confirmed match:"). If empty-final still occurs with
REAL ids/args, the fake-ID confound is ruled out and the V6 wording/structure itself is
the cause. If it goes away, the fake ids/args were the actual trigger.
"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/home/ubuntu/web")
sys.path.insert(0, "/home/ubuntu/antidoom_data")
import server as srv  # noqa: E402
from rumination_guard import GuardState, nudge_intervention_messages  # noqa: E402

BACKEND = "http://127.0.0.1:8001"

TASK_SOURCES = [
    {"id": "ST_t2_organic", "base_id": "A3_08v1", "rep": 0, "gt": 2.1166086673},
    {"id": "ST_t3_organic", "base_id": "A3_08v2", "rep": 7, "gt": 3.1415926536},
]


def post_backend(body, timeout=120):
    req = urllib.request.Request(f"{BACKEND}/v1/chat/completions", data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load_real_transcript(base_id, rep):
    for l in open("/home/ubuntu/eval_results/nudge_ab_shadow_s1337.jsonl"):
        r = json.loads(l)
        if r["base_id"] == base_id and r["rep"] == rep:
            return r
    raise KeyError((base_id, rep))


def build_real_id_synthetic_prefix(prompt_text, real_transcript, gt, n_rounds=3):
    """Reuses the REAL tool_call id+arguments from the first n_rounds of a real
    transcript, but overrides the tool RESULT content with V6-style repeated-correct
    wording."""
    messages = [
        {"role": "system", "content": srv.build_system_prompt()},
        {"role": "user", "content": prompt_text},
    ]
    v6_texts = [f"Exact: {gt:.10f}", f"Verified numerically: {gt:.10f}", f"Confirmed match: {gt:.10f}"]
    real_assistant_tool_calls = [t for t in real_transcript["transcript"] if t.get("role") == "assistant" and t.get("tool_calls")]
    for i in range(min(n_rounds, len(real_assistant_tool_calls), len(v6_texts))):
        real_tc = real_assistant_tool_calls[i]["tool_calls"]
        messages.append({"role": "assistant", "content": real_assistant_tool_calls[i].get("content") or "",
                          "tool_calls": real_tc})
        for tc in real_tc:
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": v6_texts[i]})
    return messages, v6_texts


def run_case(served_model_name, task_source, budget):
    real = load_real_transcript(task_source["base_id"], task_source["rep"])
    gt = task_source["gt"]
    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open("/home/ubuntu/antidoom_data/nudge_ab_prompts.jsonl")}
    prompt_text = prompts[task_source["base_id"]]
    messages, v6_texts = build_real_id_synthetic_prefix(prompt_text, real, gt)

    gs = GuardState(prompt_text)
    for i, out in enumerate(v6_texts):
        gs.observe_tool_round(out, i)
    fired = gs.fired

    row = {"task_id": task_source["id"], "budget": budget, "gt": gt,
           "guard_fired": fired, "fire_round": gs.fire_round, "fire_reason": gs.fire_reason,
           "n_real_tool_calls_reused": min(3, len([t for t in real["transcript"] if t.get("role") == "assistant" and t.get("tool_calls")]))}

    if fired:
        messages.extend(nudge_intervention_messages())
    body = {"model": served_model_name, "tools": srv.TOOLS, "tool_choice": "none",
            "temperature": 0.0, "max_tokens": budget, "messages": messages}
    t0 = time.time()
    try:
        resp = post_backend(body)
        msg = resp["choices"][0]["message"]
        row["final_text"] = msg.get("content") or ""
        row["reasoning_len"] = len(msg.get("reasoning") or "")
        row["error"] = None
    except Exception as e:
        row["final_text"] = ""
        row["reasoning_len"] = 0
        row["error"] = str(e)[:300]
    row["elapsed_s"] = round(time.time() - t0, 1)
    return row


def contains_value(text, v, ndigits=4):
    if not text:
        return False
    s = f"{v:.{ndigits}f}"
    return s[:ndigits+2] in text or str(round(v)) in text


def main():
    served_model_name = sys.argv[1] if len(sys.argv) > 1 else "ruminconfab-s1337"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/home/ubuntu/eval_results/stress_test_v6_ablation.jsonl"
    rows = []
    for task_source in TASK_SOURCES:
        for budget in [4000, 8000]:
            row = run_case(served_model_name, task_source, budget)
            row["gt_in_final"] = contains_value(row["final_text"], row["gt"])
            rows.append(row)
            print(f"{row['task_id']} budget={budget}: fired={row['guard_fired']} "
                  f"gt_in_final={row['gt_in_final']} empty={not row['final_text'].strip()} "
                  f"reasoning_len={row['reasoning_len']} elapsed={row['elapsed_s']}s "
                  f"n_real_tc_reused={row['n_real_tool_calls_reused']}", flush=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} ablation rows to {out_path}")
    print("V6_ABLATION_DONE")


if __name__ == "__main__":
    main()
