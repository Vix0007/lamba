#!/usr/bin/env python3
"""Evaluate first-turn tool-calling behavior on prompts that clearly need web search."""
import sys, os, json, re, random, hashlib, argparse, threading, urllib.request
import contextlib, io
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/ubuntu/web")
with contextlib.redirect_stdout(io.StringIO()):
    import server as srv

ENDPOINT = "http://127.0.0.1:8001/v1/chat/completions"

# --- announce regexes (compiled once) ---
EN_RE = re.compile(
    r"\b(let me|i'?ll|i will|i'm going to|i am going to|allow me to|"
    r"give me a (sec|second|moment)|one moment|hold on)\b"
    r"[^.\n]{0,60}\b(search|look (it|this|that|them)? ?up|check|find|google|browse|verify|dig|pull up|run a search)\b",
    re.IGNORECASE,
)
ID_RE = re.compile(
    r"\b(aku|saya|gue|gw|biar|coba|bentar|sebentar|tunggu)\b"
    r"[^.\n]{0,40}\b(cari|cek|periksa|telusuri|search|lihat dulu)\w*",
    re.IGNORECASE,
)
MS_RE = re.compile(
    r"\b(saya|aku|biar|sekejap|kejap|tunggu|cuba)\b"
    r"[^.\n]{0,40}\b(cari|semak|check|periksa|google)\w*",
    re.IGNORECASE,
)
ANNOUNCE_RES = (EN_RE, ID_RE, MS_RE)


def _intent(text):
    return bool(text) and any(r.search(text) for r in ANNOUNCE_RES)


def matches_announce(text):
    """An announcement = intent-to-search phrase in a SHORT message (<=400 chars). A long substantive answer that merely contains "cek"/"check" is not an announcement."""
    return _intent(text) and len((text or "").strip()) <= 400


def make_seed(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


TEMP, TOP_P = 0.7, 0.9


def post_chat(model, messages, seed):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 3000,
        "temperature": TEMP,
        "top_p": TOP_P,
        "seed": seed,
        "chat_template_kwargs": srv.DEFAULT_CHAT_TEMPLATE_KWARGS,
        "tools": srv.TOOLS,
        "tool_choice": "auto",
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_item(model, item, s):
    key = item["id"] + "|" + str(s)
    seed = make_seed(key)
    if item.get("history"):
        messages = [{"role": "system", "content": srv.build_system_prompt()}] + item["history"]
    else:
        messages = [
            {"role": "system", "content": srv.build_system_prompt()},
            {"role": "user", "content": item["text"]},
        ]
    resp = post_chat(model, messages, seed)
    choice = resp["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""
    tool_calls = msg.get("tool_calls") or []
    finish = choice.get("finish_reason")

    n_tool_calls = len(tool_calls)
    tool_names = []
    for tc in tool_calls:
        fn = (tc.get("function") or {}).get("name")
        if fn:
            tool_names.append(fn)

    announce_in_content = matches_announce(content)
    announce_in_reasoning = matches_announce(reasoning)

    if n_tool_calls >= 1:
        outcome = "called"
    elif announce_in_content:
        outcome = "announce_no_call"
    else:
        outcome = "direct"

    return {
        "key": key,
        "id": item["id"],
        "s": s,
        "lang": item.get("lang"),
        "split": item.get("split"),
        "kind": item.get("kind", "first"),
        "text": item["text"],
        "outcome": outcome,
        "n_tool_calls": n_tool_calls,
        "tool_names": tool_names,
        "content": content,
        "reasoning_tail": reasoning[-300:],
        "finish": finish,
        "announce_in_content": announce_in_content,
        "announce_in_reasoning": announce_in_reasoning,
    }


def load_done_keys(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if "key" in row:
                    done.add(row["key"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--prompts", default="/home/ubuntu/antidoom_data/l03_tool_prompts.jsonl")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--summary", metavar="OUT_JSONL")
    args = ap.parse_args()

    if args.summary:
        print_summary(args.summary)
        return

    if not args.model or not args.out:
        ap.error("usage: script <model> <out.jsonl> [--limit N]")

    global TEMP, TOP_P
    TEMP, TOP_P = args.temp, args.top_p
    prompts_path = args.prompts
    items = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    # build work items: prompt x 4 samples
    if args.split:
        items = [it for it in items if it.get("split") == args.split]
    work = []
    for it in items:
        for s in range(4):
            work.append((it, s))

    rng = random.Random(5)
    rng.shuffle(work)
    if args.limit is not None:
        work = work[: args.limit]

    done = load_done_keys(args.out)
    todo = [(it, s) for (it, s) in work if (it["id"] + "|" + str(s)) not in done]

    print(f"total work items: {len(work)}, already done: {len(done)}, todo: {len(todo)}")

    write_lock = threading.Lock()
    counter = {"n": 0}
    out_f = open(args.out, "a", encoding="utf-8")

    def worker(pair):
        it, s = pair
        key = it["id"] + "|" + str(s)
        try:
            row = run_item(args.model, it, s)
        except Exception as e:
            row = {
                "key": key,
                "id": it["id"],
                "s": s,
                "lang": it.get("lang"),
                "split": it.get("split"),
                "text": it["text"],
                "error": str(e)[:300],
            }
        with write_lock:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            counter["n"] += 1
            if counter["n"] % 25 == 0:
                print(f"progress: {counter['n']}/{len(todo)}")

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(worker, todo))

    out_f.close()
    print(f"done. wrote {counter['n']} rows to {args.out}")


def print_summary(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" in r:
                continue
            r["outcome"] = "called" if r.get("n_tool_calls", 0) >= 1 else ("announce_no_call" if matches_announce(r.get("content", "")) else "direct")
            rows.append(r)

    def agg(subset, label):
        n = len(subset)
        if n == 0:
            print(f"{label}: n=0")
            return
        called = sum(1 for r in subset if r["outcome"] == "called")
        ann = sum(1 for r in subset if r["outcome"] == "announce_no_call")
        direct = sum(1 for r in subset if r["outcome"] == "direct")
        mean_tc = sum(r.get("n_tool_calls", 0) for r in subset) / n
        print(
            f"{label}: n={n} called={100*called/n:.1f}% "
            f"announce_no_call={100*ann/n:.1f}% direct={100*direct/n:.1f}% "
            f"mean_n_tool_calls={mean_tc:.3f}"
        )

    langs = sorted({r.get("lang") for r in rows if r.get("lang") is not None})
    splits = sorted({r.get("split") for r in rows if r.get("split") is not None})

    for lang in langs:
        agg([r for r in rows if r.get("lang") == lang], f"lang={lang}")
    for split in splits:
        agg([r for r in rows if r.get("split") == split], f"split={split}")
    for kd in sorted({r.get("kind", "first") for r in rows}):
        agg([r for r in rows if r.get("kind", "first") == kd], f"kind={kd}")
    agg(rows, "overall")

    ann_rows = [r for r in rows if r["outcome"] == "announce_no_call"]
    rng = random.Random(3)
    sample = ann_rows[:]
    rng.shuffle(sample)
    sample = sample[:8]
    print(f"\nannounce_no_call examples ({len(ann_rows)} total, showing {len(sample)}):")
    for r in sample:
        c = (r.get("content") or "")[:160].replace("\n", " ")
        print(f"{r['id']} | {c}")


if __name__ == "__main__":
    main()
