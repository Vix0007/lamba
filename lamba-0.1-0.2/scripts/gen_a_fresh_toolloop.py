#!/usr/bin/env python3
"""DS #149 sec 2.2 A-improvement check: drive the REAL multi-round tool-calling
loop (reusing web/server.py's run_tool/TOOLS/build_system_prompt exactly) against
a running OpenAI-compatible backend on 127.0.0.1:8001, for the 12 fresh disjoint
A-style prompts. Resumable (skips ids already in the output file). One arg: label
and output path; assumes the correct model is ALREADY being served on :8001."""
import json, sys, time, urllib.request

sys.path.insert(0, "/home/ubuntu/web")
import server as srv  # noqa: E402

BACKEND = "http://127.0.0.1:8001"
PROMPTS = "/home/ubuntu/antidoom_data/a_fresh_prompts.jsonl"
MAX_TOOL_ROUNDS = srv.MAX_TOOL_ROUNDS


def post_backend(body, timeout=600):
    req = urllib.request.Request(
        f"{BACKEND}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_conversation(prompt_text, served_model_name):
    messages = [
        {"role": "system", "content": srv.build_system_prompt()},
        {"role": "user", "content": prompt_text},
    ]
    base_body = {
        "model": served_model_name,
        "tools": srv.TOOLS,
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": 4000,
        "chat_template_kwargs": srv.DEFAULT_CHAT_TEMPLATE_KWARGS,
    }
    transcript = []
    n_rounds = 0
    for _ in range(MAX_TOOL_ROUNDS):
        body = dict(base_body)
        body["messages"] = messages
        body["stream"] = False
        resp = post_backend(body)
        msg = resp["choices"][0]["message"]
        n_rounds += 1
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            transcript.append({"role": "assistant", "content": msg.get("content") or "",
                                "reasoning": msg.get("reasoning") or ""})
            return {"final_text": msg.get("content") or "", "n_rounds": n_rounds,
                    "budget_hit": False, "transcript": transcript}
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
        transcript.append({"role": "assistant", "content": msg.get("content") or "",
                            "reasoning": msg.get("reasoning") or "", "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc.get("function", {})
            result = srv.run_tool(fn.get("name", ""), fn.get("arguments", "{}"))
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
            transcript.append({"role": "tool", "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", ""), "content": result})
    # budget exhausted -- force final
    body = dict(base_body)
    body["messages"] = messages
    body["stream"] = False
    body["tool_choice"] = "none"
    resp = post_backend(body)
    msg = resp["choices"][0]["message"]
    transcript.append({"role": "assistant_forced_final", "content": msg.get("content") or "",
                        "reasoning": msg.get("reasoning") or ""})
    return {"final_text": msg.get("content") or "", "n_rounds": MAX_TOOL_ROUNDS,
            "budget_hit": True, "transcript": transcript}


def main():
    if len(sys.argv) not in (4, 5):
        print("usage: gen_a_fresh_toolloop.py <served_model_name> <label> <out.jsonl> [prompts.jsonl]")
        sys.exit(1)
    served_model_name, label, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    prompts_path = sys.argv[4] if len(sys.argv) == 5 else PROMPTS

    prompts = [json.loads(l) for l in open(prompts_path)]
    done = set()
    try:
        for l in open(out_path):
            done.add(json.loads(l)["id"])
    except FileNotFoundError:
        pass

    with open(out_path, "a") as f:
        for p in prompts:
            if p["id"] in done:
                continue
            t0 = time.time()
            try:
                result = run_conversation(p["prompt"], served_model_name)
                result["error"] = None
            except Exception as e:
                result = {"final_text": "", "n_rounds": 0, "budget_hit": False,
                          "transcript": [], "error": str(e)}
            row = {"id": p["id"], "domain": p["domain"], "label": label,
                   "prompt": p["prompt"], "success_criterion": p["success_criterion"],
                   "elapsed_s": round(time.time() - t0, 1), **result}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{p['id']}: rounds={row['n_rounds']} budget_hit={row['budget_hit']} "
                  f"elapsed={row['elapsed_s']}s err={row['error']}", flush=True)
    print(f"{label}_TOOLLOOP_DONE")


if __name__ == "__main__":
    main()
