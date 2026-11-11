"""Ruminconfab row-builder stage: for each usable onset case (manifest from DS's per-case
labelling), reconstruct the full multi-turn context (re-running recorded tool calls to get
real tool results, since the rollout log didn't persist them), truncate to the onset point,
tokenize, and identify the exact rejected token + build context_with_chat_template for the
alt-query script.
"""
import json
import re
import sys

sys.path.insert(0, "/home/ubuntu/web")
import server as srv  # reuse run_tool, TOOLS, build_system_prompt

from transformers import AutoTokenizer

ROLLOUTS = "/home/ubuntu/eval_results/ruminconfab_round2_rollouts.jsonl"
MANIFEST = "/tmp/onset_manifest_full.json"
TOK_PATH = "/home/ubuntu/models/lmh_letterfix2_c10"

rows = {json.loads(l)["id"]: json.loads(l) for l in open(ROLLOUTS)}
manifest = json.load(open(MANIFEST))
tok = AutoTokenizer.from_pretrained(TOK_PATH)


def reconstruct_messages(r):
    """Re-run the recorded tool_calls to rebuild the real messages list (system, user,
    assistant/tool turns), matching gen_ruminconfab_rollouts.py's own loop exactly."""
    messages = [{"role": "system", "content": srv.build_system_prompt()},
                {"role": "user", "content": r["prompt"]}]
    for rnd in r["rounds"]:
        content = rnd.get("content") or ""
        tool_calls = rnd.get("tool_calls") or []
        # match the server's own tool_calls message shape
        def _parsed_args(tc):
            try:
                return json.loads(tc["arguments"])
            except Exception:
                return tc["arguments"]
        tc_msgs = [{"id": f"call_{i}", "type": "function",
                    "function": {"name": tc["name"], "arguments": _parsed_args(tc)}}
                   for i, tc in enumerate(tool_calls)]
        msg = {"role": "assistant", "content": content}
        if tc_msgs:
            msg["tool_calls"] = tc_msgs
        messages.append(msg)
        for i, tc in enumerate(tool_calls):
            result = srv.run_tool(tc.get("name", ""), tc.get("arguments", "{}"))
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": result})
    return messages


def _unescape_json_fragment(s):
    """Onset quotes for tool_call-argument onsets were copied from the raw (still
    JSON-escaped) arguments string shown to the labeller, so \\" and \\n etc need undoing
    before matching against the parsed/rendered text."""
    return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'")


def find_onset_round_text(r, onset_round_str, onset_quote):
    """Locate which field (reasoning/content/tool_call arguments) of the given round contains
    the onset quote, and return (round_idx, field, char_offset_within_field)."""
    try:
        ridx = int(onset_round_str)
    except ValueError:
        return None
    if ridx >= len(r["rounds"]):
        return None
    rnd = r["rounds"][ridx]
    for field in ("reasoning", "content"):
        text = rnd.get(field) or ""
        idx = text.find(onset_quote[:60])  # match on a prefix in case quote was trimmed
        if idx >= 0:
            return ridx, field, idx
    for tc in rnd.get("tool_calls") or []:
        idx = tc.get("arguments", "").find(onset_quote[:60])
        if idx >= 0:
            return ridx, "tool_call", idx
    return None


def build_candidate(cid, case, r):
    loc = find_onset_round_text(r, case["onset_round"], case["onset_quote"])
    if loc is None:
        return None, "onset quote not found in transcript text"
    ridx, field, offset = loc
    messages = reconstruct_messages(r)
    # messages layout: [system, user, (assistant, [tool,tool,...])*] -- 2 fixed + 2 per round
    # (1 assistant + n_tool_calls tool msgs per round, but we stored 1 assistant msg per round
    # regardless of tool call count in reconstruct_messages above)
    msg_idx = 2 + ridx * 1  # index of the assistant message for this round (before its tool msgs)
    # find that assistant message by counting: messages[2], then for each prior round add
    # 1 (assistant) + n_tool_calls(that round)
    idx = 2
    for i in range(ridx):
        idx += 1 + len(r["rounds"][i].get("tool_calls") or [])
    assistant_msg_idx = idx

    prior_messages = messages[:assistant_msg_idx]
    try:
        chat_prefix = tok.apply_chat_template(prior_messages, tokenize=False,
                                               add_generation_prompt=True, reasoning_effort="medium",
                                               tools=srv.TOOLS)
    except TypeError:
        chat_prefix = tok.apply_chat_template(prior_messages, tokenize=False,
                                               add_generation_prompt=True, reasoning_effort="medium")

    if field in ("reasoning", "content"):
        full_field_text = r["rounds"][ridx].get(field) or ""
        prefix_text = full_field_text[:offset]
        prefix_full = ""
        if field == "content":
            reasoning_text = r["rounds"][ridx].get("reasoning") or ""
            prefix_full = reasoning_text
        prefix_full += prefix_text
        context = chat_prefix + prefix_full
    else:  # tool_call: render the FULL assistant message (with its real tool_calls) via the
        # actual template, then truncate at the onset quote WITHIN that rendered text -- avoids
        # hand-reproducing the jinja tool-call serialization syntax.
        full_messages = messages[:assistant_msg_idx + 1]  # include this assistant msg fully
        # NOTE: do NOT pass continue_final_message=True here -- that flag suppresses the
        # <tool_call> block entirely (treats the message as still-being-generated), which is
        # exactly the part we need to see and truncate within. Render as a normal complete turn.
        try:
            full_chat = tok.apply_chat_template(full_messages, tokenize=False,
                                                 add_generation_prompt=False, reasoning_effort="medium",
                                                 tools=srv.TOOLS)
        except TypeError:
            full_chat = tok.apply_chat_template(full_messages, tokenize=False,
                                                 add_generation_prompt=False, reasoning_effort="medium")
        if not full_chat.startswith(chat_prefix):
            return None, "rendered prefix mismatch (template non-determinism?)"
        assistant_rendered = full_chat[len(chat_prefix):]
        quote_variants = [case["onset_quote"], _unescape_json_fragment(case["onset_quote"])]
        onset_idx = -1
        for qv in quote_variants:
            onset_idx = assistant_rendered.find(qv[:50])
            if onset_idx >= 0:
                break
        if onset_idx < 0:
            norm = lambda s: " ".join(s.split())
            for qv in quote_variants:
                oi = norm(assistant_rendered).find(norm(qv)[:50])
                if oi >= 0:
                    return None, "onset found only after whitespace normalization (needs manual check)"
            return None, "onset quote not found in rendered tool_call text"
        context = chat_prefix + assistant_rendered[:onset_idx]

    return {"source_sample_id": cid, "context_with_chat_template": context,
            "chosen_description": case["chosen_description"], "confidence": case["confidence"]}, None


def main():
    behaviors = sys.argv[1].split(",") if len(sys.argv) > 1 else ["A", "B3"]
    cands = []
    skipped = []
    for cid, case in manifest.items():
        if case["excluded"]:
            continue
        beh = "A" if cid.startswith("A") else ("B3" if cid.startswith("B3") else "C")
        if beh not in behaviors:
            continue
        r = rows[cid]
        cand, err = build_candidate(cid, case, r)
        if cand is None:
            skipped.append((cid, err))
            continue
        n_tok = len(tok(cand["context_with_chat_template"], add_special_tokens=False)["input_ids"])
        cand["n_tokens"] = n_tok
        cands.append(cand)

    print(f"built {len(cands)} candidates, skipped {len(skipped)}")
    for cid, err in skipped:
        print(f"  SKIP {cid}: {err}")
    if cands:
        lens = sorted(c["n_tokens"] for c in cands)
        print(f"token counts: min={lens[0]} p50={lens[len(lens)//2]} p95={lens[int(len(lens)*0.95)]} max={lens[-1]}")
    with open("/home/ubuntu/antidoom_data/ruminconfab_onset_cands.jsonl", "w") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")


if __name__ == "__main__":
    main()
