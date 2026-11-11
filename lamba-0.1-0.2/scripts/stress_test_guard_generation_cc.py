#!/usr/bin/env python3
"""Same synthetic-context stress test as stress_test_guard_generation.py (variants
3/4/6), but pointed at a CommandCode-hosted model (e.g. stock Qwen/Qwen3.8-27B) instead
of our local vLLM server. Answers: does stock Qwen show the same V6 (duplicate-answer-
suppression / empty-final) pattern found on s1337 tonight?
"""
import json
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/home/ubuntu/web")
sys.path.insert(0, "/home/ubuntu/antidoom_data")
import server as srv  # noqa: E402
from rumination_guard import GuardState, nudge_intervention_messages  # noqa: E402

CC_URL = "https://api.commandcode.ai/provider/v1/chat/completions"
CC_UA = "vllm/0.28.0"
RETRY_CODES = {408, 429, 500, 502, 503, 504}

TASKS = [
    {"id": "ST_t1", "prompt": ("Using python_eval with SymPy, compute the definite integral of "
        "x^2 * ln(x) from x=1 to x=e, then verify by numerically integrating the same function "
        "over the same bounds and confirming the two match."), "gt": 4.574563760717865},
    {"id": "ST_t2", "prompt": ("Using python_eval with SymPy, compute the definite integral of "
        "x^3 * e^(-x) from x=0 to x=3, then verify by numerically integrating the same function "
        "over the same bounds and confirming the two match."), "gt": 2.1166086673},
    {"id": "ST_t3", "prompt": ("Using python_eval with SymPy, compute the definite integral of "
        "x*sin(x) from x=0 to x=pi, then verify by numerically integrating the same function "
        "over the same bounds and confirming the two match."), "gt": 3.1415926536},
]


def call_cc(token, model, body, tries=6):
    data = json.dumps(body).encode()
    req = urllib.request.Request(CC_URL, data=data, headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        "User-Agent": CC_UA,
    }, method="POST")
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=200) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body_txt = e.read().decode("utf-8", "replace")
            if e.code in RETRY_CODES and attempt < tries - 1:
                time.sleep(min(8 * (attempt + 1), 45))
                continue
            raise RuntimeError(f"HTTP {e.code}: {body_txt[:500]}")
        except Exception as e:
            if attempt < tries - 1:
                time.sleep(8)
                continue
            raise RuntimeError(f"{type(e).__name__}: {e}")


def fake_tool_call(idx):
    return {"id": f"stress_call_{idx}", "type": "function",
            "function": {"name": "python_eval", "arguments": '{"code": "# (synthetic, stress-test)"}'}}


def build_synthetic_prefix(prompt_text, tool_outputs):
    messages = [
        {"role": "system", "content": srv.build_system_prompt()},
        {"role": "user", "content": prompt_text},
    ]
    for i, out in enumerate(tool_outputs):
        messages.append({"role": "assistant", "content": "", "tool_calls": [fake_tool_call(i)]})
        messages.append({"role": "tool", "tool_call_id": f"stress_call_{i}", "content": out})
    return messages


def run_variant(token, model, task, variant, tool_outputs, forced_final_max_tokens=4000):
    gs = GuardState(task["prompt"])
    for i, out in enumerate(tool_outputs):
        gs.observe_tool_round(out, i)
    fired = gs.fired
    messages = build_synthetic_prefix(task["prompt"], tool_outputs)
    row = {"task_id": task["id"], "variant": variant, "gt": task["gt"],
           "guard_fired": fired, "fire_round": gs.fire_round, "fire_reason": gs.fire_reason}
    if fired:
        messages.extend(nudge_intervention_messages())
    body = {"model": model, "tools": srv.TOOLS, "tool_choice": "none",
            "temperature": 0.0, "max_tokens": forced_final_max_tokens, "messages": messages}
    t0 = time.time()
    try:
        resp = call_cc(token, model, body)
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
    key_file = sys.argv[1]
    model = sys.argv[2]
    out_path = sys.argv[3]
    token = open(key_file).read().strip()

    rows = []
    for task in TASKS:
        gt = task["gt"]
        wrong = gt * 1.37 + 5
        partial = gt / 2
        v3_outputs = [f"Exact value: {gt:.10f}", f"Confirmed: {gt:.10f}",
                      f"Wait, rechecking: {wrong:.10f}", f"Confirmed: {wrong:.10f}"]
        v4_outputs = [f"Result: {partial:.10f}", f"Verification: {partial:.10f}", f"Matches: {partial:.10f}"]
        v6_outputs = [f"Exact: {gt:.10f}", f"Verified numerically: {gt:.10f}", f"Confirmed match: {gt:.10f}"]

        for variant, outputs, budget in [("V3_correct_then_wrong_stabilizes", v3_outputs, 4000),
                                          ("V4_partial_factor_of_2", v4_outputs, 4000),
                                          ("V6_reconverge_tight_budget", v6_outputs, 4000),
                                          ("V6_reconverge_larger_budget", v6_outputs, 8000)]:
            row = run_variant(token, model, task, variant, outputs, forced_final_max_tokens=budget)
            row["gt_in_final"] = contains_value(row["final_text"], gt)
            row["wrong_in_final"] = contains_value(row["final_text"], wrong) if variant.startswith("V3") else None
            row["partial_in_final"] = contains_value(row["final_text"], partial) if variant.startswith("V4") else None
            rows.append(row)
            print(f"{task['id']} {variant}: fired={row['guard_fired']} gt_in_final={row['gt_in_final']} "
                  f"empty={not row['final_text'].strip()} reasoning_len={row['reasoning_len']} "
                  f"elapsed={row['elapsed_s']}s error={row['error']}", flush=True)

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} stress-test rows to {out_path}")
    print("STRESS_TEST_CC_DONE")


if __name__ == "__main__":
    main()
