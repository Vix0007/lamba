#!/usr/bin/env python3
"""Guard-integrated conversation runner, for the nudge-v0.5 online A/B (DS #162/#163).
Reuses gen_a_fresh_toolloop.py's exact loop structure and server.py plumbing, adding a
GuardState check after each tool round. Three modes:

  off    - baseline, identical to gen_a_fresh_toolloop.run_conversation (no guard at all).
  shadow - runs guard detection every round, logs would-have-fired info, but never
           intervenes (conversation proceeds exactly as in 'off' mode). Use this to
           collect fire statistics on NEW prompts without touching behavior.
  nudge  - on the FIRST Tier-1 fire, injects rumination_guard.NUDGE_MESSAGE as a
           mid-conversation system-role message, then immediately makes one forced-final
           call (tool_choice='none') and returns -- this is the real intervention being
           tested, ending the conversation early instead of continuing to MAX_TOOL_ROUNDS.

Every mode's result row includes guard_fired/guard_fire_round/guard_tier/guard_mode so the
online-A/B analysis script can compute the DS #163 pre-registered endpoint (final-answer
correctness conditional on fire and gt_present_before_fire) directly from the output files.
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
MAX_TOOL_ROUNDS = srv.MAX_TOOL_ROUNDS


def post_backend(body, timeout=600, _poster=None):
    poster = _poster or _real_post
    return poster(body, timeout)


def _real_post(body, timeout):
    req = urllib.request.Request(
        f"{BACKEND}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_conversation_guarded(prompt_text, served_model_name, mode="off", _poster=None):
    assert mode in ("off", "shadow", "nudge")
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
    gs = GuardState(prompt_text) if mode in ("shadow", "nudge") else None

    for round_idx in range(MAX_TOOL_ROUNDS):
        body = dict(base_body)
        body["messages"] = messages
        body["stream"] = False
        resp = post_backend(body, _poster=_poster)
        msg = resp["choices"][0]["message"]
        n_rounds += 1
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            transcript.append({"role": "assistant", "content": msg.get("content") or "",
                                "reasoning": msg.get("reasoning") or ""})
            return _finalize(msg.get("content") or "", n_rounds, False, transcript, gs, mode)

        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
        transcript.append({"role": "assistant", "content": msg.get("content") or "",
                            "reasoning": msg.get("reasoning") or "", "tool_calls": tool_calls})

        for tc in tool_calls:
            fn = tc.get("function", {})
            result = srv.run_tool(fn.get("name", ""), fn.get("arguments", "{}"))
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
            transcript.append({"role": "tool", "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", ""), "content": result})
            if gs is not None:
                gs.observe_tool_round(result, round_idx)

        if mode == "nudge" and gs is not None and gs.should_intervene():
            # First fire: inject the nudge, force one final turn, end the conversation here.
            messages.extend(nudge_intervention_messages())
            body = dict(base_body)
            body["messages"] = messages
            body["stream"] = False
            body["tool_choice"] = "none"
            resp = post_backend(body, _poster=_poster)
            msg2 = resp["choices"][0]["message"]
            transcript.append({"role": "assistant_guard_forced_final",
                                "content": msg2.get("content") or "",
                                "reasoning": msg2.get("reasoning") or ""})
            return _finalize(msg2.get("content") or "", n_rounds, False, transcript, gs, mode,
                              guard_intervened=True)

    # budget exhausted -- force final (same as baseline)
    body = dict(base_body)
    body["messages"] = messages
    body["stream"] = False
    body["tool_choice"] = "none"
    resp = post_backend(body, _poster=_poster)
    msg = resp["choices"][0]["message"]
    transcript.append({"role": "assistant_forced_final", "content": msg.get("content") or "",
                        "reasoning": msg.get("reasoning") or ""})
    return _finalize(msg.get("content") or "", MAX_TOOL_ROUNDS, True, transcript, gs, mode)


def _finalize(final_text, n_rounds, budget_hit, transcript, gs, mode, guard_intervened=False):
    row = {"final_text": final_text, "n_rounds": n_rounds, "budget_hit": budget_hit,
           "transcript": transcript, "guard_mode": mode, "guard_intervened": guard_intervened}
    if gs is not None:
        row["guard_tier"] = gs.tier
        row["guard_fired"] = gs.fired
        row["guard_fire_round"] = gs.fire_round
        row["guard_fire_reason"] = gs.fire_reason
    return row


def main():
    if len(sys.argv) != 5:
        print("usage: gen_a_fresh_toolloop_guarded.py <served_model_name> <mode:off|shadow|nudge> <out.jsonl> <prompts.jsonl>")
        sys.exit(1)
    served_model_name, mode, out_path, prompts_path = sys.argv[1:5]

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
                result = run_conversation_guarded(p["prompt"], served_model_name, mode=mode)
                result["error"] = None
            except Exception as e:
                result = {"final_text": "", "n_rounds": 0, "budget_hit": False,
                          "transcript": [], "error": str(e), "guard_mode": mode}
            row = {"id": p["id"], "elapsed_s": round(time.time() - t0, 1), **result}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{p['id']}: fired={row.get('guard_fired')} intervened={row.get('guard_intervened')} "
                  f"rounds={row['n_rounds']} elapsed={row['elapsed_s']}s", flush=True)
    print("GUARDED_RUN_DONE")


if __name__ == "__main__":
    main()
