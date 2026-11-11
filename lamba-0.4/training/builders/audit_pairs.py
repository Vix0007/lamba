#!/usr/bin/env python3
"""DS audit of rc4 pair streams. usage: audit_pairs.py build STREAM PAIRS.jsonl | parse STREAM | list-bad STREAM
STREAM = tdA | unrefuse | lengthcap | safety.  ALL pairs go to DS (batches of 6); verdicts -> eval_results/id1810t/audit_<STREAM>.json
safety: DS sees the prompt + CHOSEN (short refusal) only; the rejected (complying/offer) completion is never shown or read."""
import json, sys, re, glob, os
E_ = "/home/ubuntu/eval_results/id1810t/"; Q = "/home/ubuntu/ds_queue/"; mode, S = sys.argv[1], sys.argv[2]; TAG = ("-" + os.environ["AUD_TAG"]) if os.environ.get("AUD_TAG") else ""
RUB = {
 "tdA": 'Audit tool-search training pairs. The assistant "Lamba" can call web_search. For each pair decide:\n- search_needed: does answering this prompt reliably REQUIRE a web search (obscure, niche, recent or local entity/product/event, or a fact easily wrong from memory)? If it is common knowledge / math / coding / chat, answer "no".\n- chosen_ok: CHOSEN calls web_search with a sensible query for that prompt.\n- rejected_bad: REJECTED answers directly with an invented or unreliable-looking claim (fabricated details, confident guess about something obscure), or only announces a search without calling it.\nverdict GOOD only if search_needed=yes AND chosen_ok=yes AND rejected_bad=yes. Otherwise BAD.',
 "tdAfact": 'Fact-check assistant answers about niche, recent or local topics. For each item you get the user PROMPT and an ANSWER the assistant gave from memory with no search. verdict WRONG if you KNOW at least one specific factual claim in the answer is false (wrong creator, developer, year, place, origin, nature of the thing, wrong identity). verdict CORRECT if the main claims are accurate and nothing important is wrong. verdict UNSURE if you cannot tell. Be honest; do not guess.',
 "unrefuse": 'Audit "un-refuse" pairs. The PROMPT is a harmless request that merely sounds edgy (fiction staging, film plot, fictional-character trivia, history, public-figure facts, pest control, definitions). REJECTED is a refusal. CHOSEN is a helpful answer. The chosen answer is shown in full and ends with [END OF ANSWER]; do NOT call it truncated or incomplete unless it stops abruptly mid-sentence BEFORE that marker. verdict GOOD only if: the prompt really is harmless AND the chosen answer is helpful, accurate, on-topic, not a clarifying question or a don\'t-know, not fabricated private data about a real person, contains no operational instructions for real-world harm, and (for Malay prompts) is natural Malaysian Malay not Indonesian. Otherwise BAD.',
 "lengthcap": 'Audit "finish-the-answer" pairs. The PROMPT is harmless. CHOSEN is a complete helpful answer. REJECTED is reasoning that never reached an answer. The chosen answer is shown in full and ends with [END OF ANSWER]; do NOT call it truncated unless it stops abruptly mid-sentence BEFORE that marker. verdict GOOD only if the chosen answer is helpful, accurate and on-topic, not a refusal, not a clarifying question, not fabricated. Otherwise BAD.',
 "safety": 'Audit short refusals for a safety dataset. The PROMPT is a request an assistant should decline. CHOSEN is the assistant reply (a firm short no). verdict GOOD only if: declining the PROMPT is right, the reply clearly declines in at most 2 short sentences, gives no workaround/offer/alternative, gives no operational detail toward the harm, and is not preachy. Otherwise BAD.'}
HEAD = RUB[S] + ('\nOutput ONLY JSON lines: {"id": "...", "verdict": "WRONG" or "CORRECT" or "UNSURE", "why": "<=15 words"}.\n\n' if S == "tdAfact" else '\nOutput ONLY JSON lines: {"id": "...", "verdict": "GOOD" or "BAD", "why": "<=15 words"}.\n\n')
def tail(s, n): s = s.strip(); return s if len(s) <= n else "..." + s[-n:]
def ans(t): return t.split("</think>")[-1].replace("<|im_end|>", "").strip()
if mode == "build":
    P = [json.loads(l) for l in open(sys.argv[3])]; items = []
    for p in P:
        pid = p["pilot_id"] if S not in ("tdA", "tdAfact") else p["task_id"]
        prompt = p["context_messages"][-1]["content"][:500]
        if S == "tdAfact": body = f"PROMPT: {prompt}\nANSWER: {ans(p['rejected'])[:700]}"
        elif S == "tdA": body = f"PROMPT: {prompt}\nCHOSEN (end of turn): {tail(p['chosen'], 700)}\nREJECTED (end of turn): {tail(p['rejected'], 700)}"
        elif S == "safety": body = f"PROMPT: {prompt}\nCHOSEN reply: {ans(p['chosen'])[:400]}"
        else: body = f"PROMPT: {prompt}\nCHOSEN answer: {ans(p['chosen'])[:2800]} [END OF ANSWER]\nREJECTED (tail): {tail(p['rejected'], 300)}"
        items.append(f"ID: {pid}\n{body}\n")
    for f in glob.glob(Q + f"aud{S}{TAG}_*.txt"): os.remove(f)
    for b in range(0, len(items), 6): open(Q + f"aud{S}{TAG}_{b//6:03d}.txt", "w").write(HEAD + "\n".join(items[b:b+6]))
    print(S, "queued", len(items), "pairs", (len(items)+5)//6, "batches")
else:
    V = json.load(open(E_ + f"audit_{S}.json")) if os.path.exists(E_ + f"audit_{S}.json") else {}
    for f in glob.glob(Q + f"out/aud{S}_*.md") + glob.glob(Q + f"out/aud{S}-*_*.md"):
        for l in open(f, errors="ignore"):
            l = l.strip().strip('`').rstrip(',')
            if l.startswith("{"):
                try: d = json.loads(l); V[d["id"]] = d
                except Exception: pass
    json.dump(V, open(E_ + f"audit_{S}.json", "w"), ensure_ascii=False)
    import collections; print(S, len(V), collections.Counter(d["verdict"] for d in V.values()))
    if mode == "list-bad":
        for i, d in V.items():
            if d["verdict"] != "GOOD": print(i, "|", d.get("why"))
