#!/usr/bin/env python3
import sys, contextlib, io, os, json, re, random, hashlib, argparse, threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/ubuntu/web")
sys.path.insert(0, "/home/ubuntu/antidoom/src")
with contextlib.redirect_stdout(io.StringIO()):
    import server as srv
from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained(
    "/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True
)

SERVER_URL = "http://127.0.0.1:8001/v1/completions"
WRITE_LOCK = threading.Lock()

TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
FUNCTION_RE = re.compile(r"<function=([^>\n]+)>")


def normalize_messages(messages):
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function", {}))
                args = fn.get("arguments")
                if isinstance(args, str):
                    fn["arguments"] = json.loads(args)
                tc["function"] = fn
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def render_context(messages, tools=True):
    msgs = normalize_messages(messages)
    return TOK.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        tools=srv.TOOLS if tools else None,
        reasoning_effort="medium",
    )


def state_seed(state_id):
    return int(hashlib.sha256(str(state_id).encode()).hexdigest()[:8], 16)


def request_samples(model, ctx, k, temp, seed, top_p=0.95):
    body = {
        "model": model,
        "prompt": ctx,
        "max_tokens": 3000,
        "temperature": temp,
        "top_p": top_p,
        "seed": seed,
        "n": k,
        "skip_special_tokens": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        SERVER_URL, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def build_sample(choice, prefill=""):
    text = prefill + choice.get("text", "")
    finish = choice.get("finish_reason", "")
    if finish == "stop":
        full_text = text + "<|im_end|>"
        truncated = False
    else:
        full_text = text
        truncated = True

    has_think_close = "</think>" in text
    has_tool_call = "<tool_call>" in text

    if has_think_close:
        reasoning = text.split("</think>", 1)[0]
        after = text.split("</think>", 1)[1]
        after = TOOL_CALL_BLOCK_RE.sub("", after)
        answer = after.lstrip()
    else:
        reasoning = ""
        answer = ""

    return {
        "text": full_text,
        "raw_text": text,
        "finish": finish,
        "truncated": truncated,
        "has_think_close": has_think_close,
        "has_tool_call": has_tool_call,
        "reasoning": reasoning,
        "answer": answer,
        "tool_call_names": FUNCTION_RE.findall(text),
    }


def process_state(state, model, k, temp):
    state_id = state.get("state_id")
    meta = state.get("meta")
    try:
        ctx = render_context(state["messages"], state.get("tools", True))
        seed = state_seed(state_id)
        prefill = state.get("prefill") or ""
        resp = request_samples(model, ctx + prefill, k, temp, seed)
        choices = resp.get("choices") or []
        if len(choices) < k:
            raise RuntimeError("expected %d choices, got %d" % (k, len(choices)))
        samples = [build_sample(c, prefill) for c in choices[:k]]
        return {
            "state_id": state_id,
            "meta": meta,
            "rendered_context": ctx,
            "samples": samples,
        }
    except Exception as e:
        return {"state_id": state_id, "meta": meta, "error": str(e)[:300]}


def write_line(out_path, obj):
    line = json.dumps(obj, ensure_ascii=False)
    with WRITE_LOCK:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def load_done(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("error"):
                continue
            sid = obj.get("state_id")
            if sid is not None:
                done.add(sid)
    return done


def load_states(path):
    states = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            states.append(json.loads(line))
    return states


def run_selftest(args):
    model = args.served_model or "default"
    state = {
        "state_id": "selftest",
        "messages": [
            {"role": "system", "content": srv.build_system_prompt()},
            {"role": "user", "content": "who are you"},
        ],
        "meta": {"selftest": True},
    }
    ctx = render_context(state["messages"])
    print("[selftest] rendered_context tail: %r" % ctx[-120:], flush=True)
    seed = state_seed(state["state_id"])
    resp = request_samples(model, ctx, 2, args.temp, seed)
    choices = resp.get("choices") or []
    print("[selftest] choices=%d" % len(choices), flush=True)
    for i, c in enumerate(choices[:2]):
        s = build_sample(c)
        print(
            "[selftest] sample %d: finish=%r truncated=%s has_think_close=%s len(full_text)=%d"
            % (i, s["finish"], s["truncated"], s["has_think_close"], len(s["text"])),
            flush=True,
        )
        print("[selftest]   last80=%r" % (s["text"][-80:],), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("served_model", nargs="?")
    ap.add_argument("states_jsonl", nargs="?")
    ap.add_argument("out_jsonl", nargs="?")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        run_selftest(args)
        return

    if not (args.served_model and args.states_jsonl and args.out_jsonl):
        ap.error("served_model, states_jsonl, out_jsonl are required unless --selftest")

    states = load_states(args.states_jsonl)
    rng = random.Random(5)
    rng.shuffle(states)
    if args.limit is not None:
        states = states[: args.limit]

    done = load_done(args.out_jsonl)
    todo = [s for s in states if s.get("state_id") not in done]
    print(
        "[mine] states=%d done=%d todo=%d"
        % (len(states), len(states) - len(todo), len(todo)),
        flush=True,
    )
    if not todo:
        return

    completed = 0
    with ThreadPoolExecutor(max_workers=int(os.environ.get("MINE_CONC", "16"))) as ex:
        futures = {
            ex.submit(process_state, s, args.served_model, args.k, args.temp): s
            for s in todo
        }
        for fut in as_completed(futures):
            obj = fut.result()
            write_line(args.out_jsonl, obj)
            completed += 1
            if completed % 20 == 0:
                print("[mine] completed %d/%d" % (completed, len(todo)), flush=True)

    print("[mine] wrote %d rows to %s" % (completed, args.out_jsonl), flush=True)


if __name__ == "__main__":
    main()
