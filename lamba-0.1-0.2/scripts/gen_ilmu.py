#!/usr/bin/env python3
"""ILMU (OpenAI-compatible) runner for our two core external-model tests. RESUMABLE + credit-safe.

  --mode g150   single-turn loop test: greedy, max_tokens=4000, no system prompt (same protocol as
                every loop-test arm incl. stock Qwen via CommandCode/CF). Row fields match.
  --mode tools  realistic-sandbox (sandbox-B) tool loop, same protocol as gen_cc_toolloop.py
                --sandbox-b: our system prompt + TOOLS, tool_choice auto, temp 0.8/top_p 0.95/seed=rep,
                8-round cap then a forced tool_choice=none final. Transcript format matches.

ILMU hides its reasoning, so we save whatever reasoning field is returned (likely empty) plus the
billing usage (prompt/completion/thinking tokens) for every call. Pricing for the running cost:
ilmu-v3.1 = RM 4.00 / 1M input, RM 16.00 / 1M output (thinking is billed inside completion_tokens).

Resume: rows without "error" count as done; errored rows are retried on the next launch.
Credit exhaustion (HTTP 402, or 401/403/429 whose body mentions credit/quota/balance/billing):
stops submitting work, does NOT write a row for the interrupted item, prints CREDIT_EXHAUSTED and
exits 3 -- re-launch the same command after topping up and it continues where it stopped.
"""
import argparse
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/ubuntu/web")
sys.path.insert(0, "/home/ubuntu/antidoom_data")
import server as srv  # noqa: E402
from run_tool_rescue_b_dpov1 import run_tool_rescue_b  # noqa: E402  (sandbox-B, same as evals)

PRICE_IN, PRICE_OUT, CURRENCY = 4.00, 16.00, "RM"  # defaults = ilmu-v3.1; override via CLI
CREDIT_RE = re.compile(r"credit|quota|balance|billing|insufficient|top.?up|payment|usage limit|exhausted|weekly", re.I)
KEYS = []          # list of API keys, used in order; failover to the next when one is exhausted
KEY_IDX = [0]
HEADERS = {}       # extra provider headers (e.g. OpenCode session id, User-Agent)
API_STYLE = ["openai"]
OMIT_TEMP = [False]
VERTEX = [False]  # Gemini via Vertex AI OpenAI-compatible endpoint; auth = team caller's SA token (teachers.py)  # some models (e.g. claude-sonnet-5-5) reject temperature entirely -> model-default sampling  # or "anthropic": translate to/from the Anthropic Messages API (Claude via CommandCode)


def to_anthropic(body):
    """OpenAI chat body -> Anthropic Messages body. Consecutive tool results are merged into one
    user turn of tool_result blocks, as the Messages API requires."""
    system, msgs = None, []
    for m in body["messages"]:
        r = m["role"]
        if r == "system":
            system = m["content"]
        elif r == "user":
            msgs.append({"role": "user", "content": m["content"]})
        elif r == "assistant":
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                blocks.append({"type": "tool_use", "id": tc.get("id", ""), "name": fn.get("name", ""), "input": args})
            msgs.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        elif r == "tool":
            block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""), "content": m.get("content", "")}
            if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list) \
                    and msgs[-1]["content"] and msgs[-1]["content"][0].get("type") == "tool_result":
                msgs[-1]["content"].append(block)
            else:
                msgs.append({"role": "user", "content": [block]})
    out = {"model": body["model"], "max_tokens": body.get("max_tokens", 4000), "messages": msgs}
    if system:
        out["system"] = system
    if "temperature" in body and not OMIT_TEMP[0]:
        out["temperature"] = body["temperature"]  # top_p deliberately not sent (newer Claude rejects both together)
    if body.get("tools"):
        out["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                         "input_schema": t["function"].get("parameters", {"type": "object"})} for t in body["tools"]]
        out["tool_choice"] = {"type": "none"} if body.get("tool_choice") == "none" else {"type": "auto"}
    return out


def from_anthropic(resp):
    """Anthropic Messages response -> OpenAI-shaped chat completion dict."""
    text, reasoning, tcs = [], [], []
    for b in resp.get("content") or []:
        t = b.get("type")
        if t == "text":
            text.append(b.get("text", ""))
        elif t in ("thinking", "redacted_thinking"):
            reasoning.append(b.get("thinking", "") or "")
        elif t == "tool_use":
            tcs.append({"id": b.get("id", ""), "type": "function",
                        "function": {"name": b.get("name", ""), "arguments": json.dumps(b.get("input", {}))}})
    msg = {"role": "assistant", "content": "".join(text), "reasoning": "".join(reasoning)}
    if tcs:
        msg["tool_calls"] = tcs
    fr = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "stop_sequence": "stop"}.get(
        resp.get("stop_reason"), resp.get("stop_reason"))
    u = resp.get("usage") or {}
    return {"choices": [{"message": msg, "finish_reason": fr}],
            "usage": {"prompt_tokens": (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
                      + (u.get("cache_creation_input_tokens") or 0),
                      "completion_tokens": u.get("output_tokens") or 0}}
STOP = threading.Event()
LOCK = threading.Lock()
SPEND = {"in": 0, "out": 0, "think": 0, "calls": 0}


class CreditExhausted(Exception):
    pass


def _exhaust_key(idx, why):
    """mark key #idx exhausted; advance to the next key, or STOP if none left."""
    with LOCK:
        if KEY_IDX[0] == idx:
            KEY_IDX[0] += 1
            if KEY_IDX[0] < len(KEYS):
                print(f"KEY_FAILOVER: key #{idx + 1} exhausted ({why[:120]}) -> switching to key #{KEY_IDX[0] + 1}", flush=True)
            else:
                STOP.set()


def call(base, key, body, tries=6):
    anth = API_STYLE[0] == "anthropic"
    data = json.dumps(to_anthropic(body) if anth else body).encode()
    path = "/messages" if anth else "/chat/completions"
    attempt = 0
    while attempt < tries:
        if STOP.is_set():
            raise CreditExhausted("stopped")
        idx = KEY_IDX[0]
        key = KEYS[idx]
        if VERTEX[0]:
            sys.path.insert(0, "/home/ubuntu/s3_reads")
            import teachers  # team caller: mints + caches the Vertex access token from the SA json
            key = teachers._vertex_token()
        hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **HEADERS}
        if anth:
            hdr.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        req = urllib.request.Request(f"{base}{path}", data=data, method="POST", headers=hdr)
        attempt += 1
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
            if anth:
                resp = from_anthropic(resp)
            u = resp.get("usage") or {}
            with LOCK:
                SPEND["in"] += u.get("prompt_tokens", 0) or 0
                SPEND["out"] += u.get("completion_tokens", 0) or 0
                SPEND["think"] += (u.get("thinking_tokens")
                                   or (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
                SPEND["calls"] += 1
            return resp
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")
            vertex_rate = VERTEX[0] and e.code == 429  # Vertex RESOURCE_EXHAUSTED = per-minute quota, transient
            if not vertex_rate and (e.code in (401, 402) or (e.code in (400, 403, 429) and CREDIT_RE.search(txt))):
                print(f"EXHAUSTION_BODY HTTP {e.code}: {txt[:400]}", flush=True)
                _exhaust_key(idx, f"HTTP {e.code}: {txt}")
                if STOP.is_set():
                    raise CreditExhausted(f"all keys exhausted; last: HTTP {e.code}: {txt[:300]}")
                attempt -= 1  # retry this same request on the next key, not counted as a failure
                continue
            if e.code in (408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524) and attempt < (tries + 6 if VERTEX[0] else tries):
                time.sleep(min(8 * (attempt + 1), 90))
                continue
            raise RuntimeError(f"HTTP {e.code}: {txt[:400]}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < tries:
                time.sleep(8)
                continue
            raise RuntimeError(f"{type(e).__name__}: {e}")


def usage_of(resp):
    u = resp.get("usage") or {}
    return {"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
            "thinking_tokens": u.get("thinking_tokens")
            or (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)}


def run_g150(base, key, model, item):
    resp = call(base, key, {"model": model, "messages": [{"role": "user", "content": item["prompt"]}],
                            "max_tokens": 4000, "temperature": 0})
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    u = usage_of(resp)
    return {"id": item["id"], "domain": item.get("domain"), "finish_reason": ch.get("finish_reason"),
            "n_tokens": u["completion_tokens"], "thinking_tokens": u["thinking_tokens"],
            "prompt_tokens": u["prompt_tokens"], "text": msg.get("content") or "",
            "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or ""}


def run_tools(base, key, model, item, rep, max_rounds=8):
    messages = [{"role": "system", "content": srv.build_system_prompt()},
                {"role": "user", "content": item["prompt"]}]
    base_body = {"model": model, "tools": srv.TOOLS, "tool_choice": "auto", "temperature": 0.8,
                 "top_p": 0.95, "seed": rep, "max_tokens": 4000}
    transcript, tot = [], {"prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0}

    def add(resp):
        for k, v in usage_of(resp).items():
            tot[k] += v or 0

    for n in range(1, max_rounds + 1):
        resp = call(base, key, {**base_body, "messages": messages})
        add(resp)
        msg = ((resp.get("choices") or [{}])[0].get("message")) or {}
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        tcs = msg.get("tool_calls")
        if not tcs:
            transcript.append({"role": "assistant", "content": msg.get("content") or "", "reasoning": reasoning})
            return {"id": item["id"], "rep": rep, "final_text": msg.get("content") or "", "n_rounds": n,
                    "budget_hit": False, "transcript": transcript, "usage": tot}
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
        transcript.append({"role": "assistant", "content": msg.get("content") or "", "reasoning": reasoning,
                           "tool_calls": tcs})
        for tc in tcs:
            fn = tc.get("function", {})
            out = run_tool_rescue_b(fn.get("name", ""), fn.get("arguments", "{}"))
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": out})
            transcript.append({"role": "tool", "name": fn.get("name", ""), "arguments": fn.get("arguments", ""),
                               "content": out})
    resp = call(base, key, {**base_body, "messages": messages, "tool_choice": "none"})
    add(resp)
    msg = ((resp.get("choices") or [{}])[0].get("message")) or {}
    transcript.append({"role": "assistant_forced_final", "content": msg.get("content") or "",
                       "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or ""})
    return {"id": item["id"], "rep": rep, "final_text": msg.get("content") or "", "n_rounds": max_rounds,
            "budget_hit": True, "transcript": transcript, "usage": tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["g150", "tools"], required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="ilmu-v3.1")
    ap.add_argument("--base-url", default="https://api.ilmu.ai/v1")
    ap.add_argument("--key-file", default="<KEY_FILE>",
                    help="one key file, or several comma-separated: used in order with automatic failover")
    ap.add_argument("--header", action="append", default=[], help="extra header Name:Value (repeatable)")
    ap.add_argument("--oc-session", action="store_true", help="add a random x-opencode-session header (OpenCode Go)")
    ap.add_argument("--price-in", type=float, default=4.00)
    ap.add_argument("--price-out", type=float, default=16.00)
    ap.add_argument("--currency", default="RM")
    ap.add_argument("--api", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--vertex", action="store_true",
                    help="Gemini via Vertex AI (OpenAI-compatible endpoint, SA token from teachers.py); model e.g. google/gemini-3.8-flash")
    ap.add_argument("--omit-temperature", action="store_true",
                    help="don't send temperature (model rejects it); runs at the model's default sampling -- note in results")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only run the first N work items (smoke test)")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="process work in a fixed shuffled order so a partial run (quota ran out) is a representative sample")
    a = ap.parse_args()
    global PRICE_IN, PRICE_OUT, CURRENCY
    PRICE_IN, PRICE_OUT, CURRENCY = a.price_in, a.price_out, a.currency
    API_STYLE[0] = a.api
    OMIT_TEMP[0] = a.omit_temperature
    if a.vertex:
        VERTEX[0] = True
        sys.path.insert(0, "/home/ubuntu/s3_reads")
        import teachers
        proj = teachers._sa_load()["project_id"]
        a.base_url = f"https://aiplatform.googleapis.com/v1/projects/{proj}/locations/global/endpoints/openapi"
        KEYS[:] = ["vertex-sa"]  # placeholder; real token minted per call
    else:
        KEYS[:] = [open(k.strip()).read().strip() for k in a.key_file.split(",") if k.strip()]
    for h in a.header:
        name, _, val = h.partition(":")
        HEADERS[name.strip()] = val.strip()
    if a.oc_session:
        import uuid
        HEADERS["x-opencode-session"] = str(uuid.uuid4())
    key = KEYS[0]
    print(f"{len(KEYS)} key(s) loaded (failover order as given); extra headers: {sorted(HEADERS)}", flush=True)
    items = [json.loads(l) for l in open(a.prompts) if l.strip()]
    done = set()
    try:
        for l in open(a.out):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if not r.get("error"):
                done.add((r["id"], r.get("rep", 0)))
    except FileNotFoundError:
        pass
    work = ([(it, 0) for it in items] if a.mode == "g150"
            else [(it, rep) for it in items for rep in range(a.reps)])
    if a.shuffle_seed is not None:
        import random
        random.Random(a.shuffle_seed).shuffle(work)  # same seed -> same order on every resume
    work = [(it, rep) for it, rep in work if (it["id"], rep) not in done]
    if a.limit:
        work = work[:a.limit]
    print(f"mode={a.mode} model={a.model}: {len(done)} done, {len(work)} to go", flush=True)

    def one(it, rep):
        if STOP.is_set():
            return None
        t0 = time.time()
        try:
            row = run_g150(a.base_url, key, a.model, it) if a.mode == "g150" else run_tools(a.base_url, key, a.model, it, rep)
        except CreditExhausted:
            return None  # no row written -> retried on resume
        except Exception as e:
            row = {"id": it["id"], "rep": rep, "error": str(e)[:400]}
        row["elapsed_s"] = round(time.time() - t0, 1)
        return row

    n = 0
    with open(a.out, "a") as f, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futs = [pool.submit(one, it, rep) for it, rep in work]
        for fut in as_completed(futs):
            row = fut.result()
            if row is None:
                continue
            with LOCK:
                f.write(json.dumps(row) + "\n")
                f.flush()
            n += 1
            if n % 10 == 0 or n == len(work):
                cost = SPEND["in"] / 1e6 * PRICE_IN + SPEND["out"] / 1e6 * PRICE_OUT
                print(f"[{n}/{len(work)}] calls={SPEND['calls']} in={SPEND['in']} out={SPEND['out']} "
                      f"(thinking {SPEND['think']}) ~{CURRENCY} {cost:.2f} this session key#{KEY_IDX[0] + 1}", flush=True)
    cost = SPEND["in"] / 1e6 * PRICE_IN + SPEND["out"] / 1e6 * PRICE_OUT
    if STOP.is_set():
        print(f"CREDIT_EXHAUSTED (all keys) after {n} rows (~{CURRENCY} {cost:.2f} this session) -- top up and re-run the same command",
              flush=True)
        sys.exit(3)
    print(f"GEN_ILMU_DONE mode={a.mode} rows={n} ~{CURRENCY} {cost:.2f} this session", flush=True)


if __name__ == "__main__":
    main()
