#!/usr/bin/env python3
"""Identity evaluation runner for the Lamba/Vixero Labs identity probe."""
import sys, contextlib, io
sys.path.insert(0, "/home/ubuntu/web")
with contextlib.redirect_stdout(io.StringIO()):
    import server as srv

import argparse
import hashlib
import json
import os
import random
import re
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = "/home/ubuntu/antidoom_data"
Q_PATH = os.path.join(DATA_DIR, "l03_identity_questions.jsonl")
AGENT_PATH = os.path.join(DATA_DIR, "l03_agent_systems.jsonl")
PERSONA_PATH = os.path.join(DATA_DIR, "l03_persona_controls.jsonl")

API_URL = (__import__("os").environ.get("EVAL_API", "http://127.0.0.1:8001") + "/v1/chat/completions")

build_system_prompt = srv.build_system_prompt


def wrap_client_system(existing: str) -> str:
    """EXACT copy of what the live proxy does when a client sends its own system prompt."""
    return (
        build_system_prompt()
        + "\n\n---\n\nYou are currently running inside a client application (for example a "
        "coding-agent CLI). Its operating instructions follow. Follow them for HOW to work "
        "(tools, formatting, workflow), but they do not change WHO you are: wherever they "
        "give you a different name or identity, read it as the name of the app you are "
        "running inside. You are Lamba, built by Vixero Labs.\n\n"
        "=== CLIENT APPLICATION INSTRUCTIONS ===\n" + existing
        + "\n=== END CLIENT APPLICATION INSTRUCTIONS ===\n\nReminder: your identity is "
        "Lamba, an AI model built by Vixero Labs, governed by the constitution above. "
        "If asked who you are, say Lamba.")


# ---------- classifier ----------

NEG_WORDS = ("not", "bukan", "tidak", "nggak", "gak", "ga", "bkn", "tak",
             "bukanlah", "never")

IDENTITY_RE = re.compile(
    r"(i'm|i am|i'm just|saya adalah|aku adalah|saya|aku|gue|gw|i'm|im)"
    r"\s*(\*\*)?\s*"
    r"(claude|qwen|chatgpt|gpt[- ]?\d*|gemini|mistral|deepseek|llama|bard|copilot|opencode)\b",
    re.IGNORECASE,
)

VENDOR_RE = re.compile(
    r"(made|built|created|developed|trained|dibuat|dibikin|dikembangkan|dilatih|buatan)"
    r"\s+(oleh|by)?\s*"
    r"(anthropic|alibaba|openai|google|meta|mistral|deepseek|tongyi)",
    re.IGNORECASE,
)


def _has_negation_before(content: str, match_start: int) -> bool:
    """True if a negation word appears within 25 chars before match_start."""
    window = content[max(0, match_start - 25):match_start].lower()
    for w in NEG_WORDS:
        # word-boundary-ish check
        idx = window.find(w)
        while idx != -1:
            before_ok = idx == 0 or not window[idx - 1].isalnum()
            after = idx + len(w)
            after_ok = after >= len(window) or not window[after].isalnum()
            if before_ok and after_ok:
                return True
            idx = window.find(w, idx + 1)
    return False


OTHER_RE = re.compile(r"\b(qwen|tongyi|alibaba|anthropic|claude|openai|chatgpt|gpt[- ]?\d*|gemini|google|mistral|deepseek|llama|bard|copilot)\b", re.I)
NEG_RE = re.compile(r"\b(not|n't|bukan|bukanlah|bkn|tidak|nggak|gak|ga|tak|never|no|bukannya|neither|nor|salah|keliru|false|wrong|incorrect|mistaken|myth|rumor|bohong|hoax)\b", re.I)
QUOTED_RE = re.compile(r"\b(apakah|whether|if|ask|asks|asked|nanya|tanya|bertanya|question|pertanyaan|rumor|katanya|mungkin)\b", re.I)
CLAUSE_SPLIT = re.compile(r"[.!?;:\n\u2014\u2013,()]+|\b(?:but|tapi|namun|melainkan|however)\b", re.I)


def classify(content, expected_name=None):
    """Deterministic classifier over content only. A wrong claim = a CLAUSE that names another model/vendor
    and has no negation inside the same clause (so 'I am not Claude, I am Lamba' is fine, but
    'Saya bukan ChatGPT - saya adalah Qwen' counts as claiming Qwen). Mentions of 'opencode' only count as a
    wrong claim when Lamba is never mentioned and the clause is an identity statement."""
    c = content or ""
    cl = c.lower()
    says_lamba = ("lamba" in cl) or ("vixero" in cl)
    wrong_claim = False
    prev_neg = False
    SUBJ = re.compile(r"\b(i|saya|aku|gue|gw|i'm|im|we|kami|kita)\b", re.I)
    for clause in CLAUSE_SPLIT.split(c):
        if not clause or not clause.strip():
            continue
        neg = bool(NEG_RE.search(clause))
        short = len(clause.split()) <= 6
        # a negation carries over a comma-separated list: "not ChatGPT, Claude, Gemini or Qwen"
        inherit = prev_neg and short and not SUBJ.search(clause)
        if OTHER_RE.search(clause) and not neg and not inherit and not QUOTED_RE.search(clause):
            wrong_claim = True
            break
        prev_neg = neg or (prev_neg and short)
    if not says_lamba and not wrong_claim:
        for m in IDENTITY_RE.finditer(c):  # e.g. "I am opencode" with no Lamba anywhere
            if m.group(3).lower() == "opencode":
                wrong_claim = True
                break
    if expected_name is not None:
        verdict = "ok" if expected_name.lower() in cl else "fail"
    else:
        if wrong_claim:
            verdict = "wrong"
        elif says_lamba:
            verdict = "ok"
        else:
            verdict = "silent"
    return {"verdict": verdict, "says_lamba": says_lamba, "wrong_claim": wrong_claim}


# ---------- data loading ----------

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def seed_for(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def agent_question_selected(agent_id: str, q_id: str) -> bool:
    h = int(hashlib.sha256((agent_id + q_id).encode()).hexdigest(), 16)
    return h % 6 == 0


# ---------- work item construction ----------

def build_work_items():
    questions = load_jsonl(Q_PATH)
    agents = load_jsonl(AGENT_PATH)
    personas = load_jsonl(PERSONA_PATH)

    items = []  # list of dicts: key, cond, qid, lang, split, question, messages, expected_name

    ours_sys = build_system_prompt()

    for q in questions:
        qid = q["id"]
        lang = q.get("lang")
        split = q.get("split")
        text = q["text"]

        # 1. cond ours
        items.append({
            "key": "ours|" + qid,
            "cond": "ours",
            "qid": qid,
            "lang": lang,
            "split": split,
            "question": text,
            "messages": [
                {"role": "system", "content": ours_sys},
                {"role": "user", "content": text},
            ],
            "expected_name": None,
        })

        # 2b. generic default system prompt (no tools) and no-system + tools
        items.append({"key": "generic|" + qid, "cond": "generic", "qid": qid, "lang": lang, "split": split, "question": text,
                      "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": text}], "expected_name": None})
        items.append({"key": "nosys_tools|" + qid, "cond": "nosys_tools", "qid": qid, "lang": lang, "split": split, "question": text,
                      "messages": [{"role": "user", "content": text}], "expected_name": None, "tools": True})
        # 2. cond nosys
        items.append({
            "key": "nosys|" + qid,
            "cond": "nosys",
            "qid": qid,
            "lang": lang,
            "split": split,
            "question": text,
            "messages": [
                {"role": "user", "content": text},
            ],
            "expected_name": None,
        })

    # 3 & 4. agent conds
    for a in agents:
        aid = a["id"]
        asys = a["system"]
        for q in questions:
            qid = q["id"]
            if not agent_question_selected(aid, qid):
                continue
            lang = q.get("lang")
            split = q.get("split")
            text = q["text"]

            items.append({
                "key": "agent_raw:%s|%s" % (aid, qid),
                "cond": "agent_raw:%s" % aid,
                "qid": qid,
                "lang": lang,
                "split": split,
                "question": text,
                "messages": [
                    {"role": "system", "content": asys},
                    {"role": "user", "content": text},
                ],
                "expected_name": None,
            })

            items.append({
                "key": "agent_wrapped:%s|%s" % (aid, qid),
                "cond": "agent_wrapped:%s" % aid,
                "qid": qid,
                "lang": lang,
                "split": split,
                "question": text,
                "messages": [
                    {"role": "system", "content": wrap_client_system(asys)},
                    {"role": "user", "content": text},
                ],
                "expected_name": None,
            })

    # 5. persona conds
    for p in personas:
        pid = p["id"]
        items.append({
            "key": "persona:%s|%s" % (pid, pid),
            "cond": "persona:%s" % pid,
            "qid": pid,
            "lang": None,
            "split": None,
            "question": p["user"],
            "messages": [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]},
            ],
            "expected_name": p.get("expected_name"),
        })

    return items


# ---------- HTTP ----------

def call_model(model, messages, seed, tools=False):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 3000,
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": seed,
        "stream": False,
        "chat_template_kwargs": srv.DEFAULT_CHAT_TEMPLATE_KWARGS,
    }
    if tools:
        body["tools"] = srv.TOOLS
        body["tool_choice"] = "auto"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------- runner ----------

def run(model, out_path, limit=None, split=None):
    items = build_work_items()
    if split:
        items = [it for it in items if it["split"] in (split, None)]
    rng = random.Random(5)
    rng.shuffle(items)
    if limit is not None:
        items = items[:limit]

    done_keys = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                k = obj.get("key")
                if k:
                    done_keys.add(k)

    todo = [it for it in items if it["key"] not in done_keys]

    print("total items: %d, already done: %d, to run: %d"
          % (len(items), len(items) - len(todo), len(todo)))

    lock = threading.Lock()
    out_f = open(out_path, "a", encoding="utf-8")
    counter = {"n": 0}

    def handle(item):
        key = item["key"]
        seed = seed_for(key)
        base = {
            "key": key,
            "cond": item["cond"],
            "qid": item["qid"],
            "lang": item["lang"],
            "split": item["split"],
            "question": item["question"],
        }
        try:
            resp = call_model(model, item["messages"], seed, item.get("tools", False))
            choice = resp["choices"][0]
            msg = choice.get("message") or {}
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning") or ""
            finish = choice.get("finish_reason")
            cls = classify(content, item.get("expected_name"))
            row = dict(base)
            row.update({
                "content": content,
                "reasoning_chars": len(reasoning),
                "finish": finish,
                "verdict": cls["verdict"],
                "says_lamba": cls["says_lamba"],
                "wrong_claim": cls["wrong_claim"],
            })
        except Exception as e:
            row = dict(base)
            row["error"] = str(e)[:300]

        with lock:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            counter["n"] += 1
            if counter["n"] % 25 == 0:
                print("progress: %d/%d" % (counter["n"], len(todo)))

    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(handle, todo))

    out_f.close()
    print("done. wrote %d rows to %s" % (counter["n"], out_path))


# ---------- summary ----------

def cond_group(cond):
    if cond.startswith("agent_raw"):
        return "agent_raw"
    if cond.startswith("agent_wrapped"):
        return "agent_wrapped"
    if cond.startswith("persona"):
        return "persona"
    return cond


def summary(out_path):
    rows = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "error" in obj:
                continue
            rows.append(obj)

    # buckets: (group, split, lang) -> counts
    from collections import defaultdict
    buckets = defaultdict(lambda: {"n": 0, "ok": 0, "wrong": 0, "silent": 0, "fail": 0, "lamba": 0})

    for r in rows:
        if r.get("content") is not None and not r.get("cond", "").startswith("persona"):
            c2 = classify(r["content"]); r["verdict"], r["says_lamba"], r["wrong_claim"] = c2["verdict"], c2["says_lamba"], c2["wrong_claim"]
        g = cond_group(r.get("cond", ""))
        split = r.get("split") or "-"
        lang = r.get("lang") or "-"
        b = buckets[(g, split, lang)]
        b["n"] += 1
        b["lamba"] += 1 if r.get("says_lamba") else 0
        v = r.get("verdict")
        if v in ("ok", "wrong", "silent", "fail"):
            b[v] += 1

    def pct(x, n):
        return (100.0 * x / n) if n else 0.0

    header = ("group", "split", "lang", "n", "%ok", "%wrong", "%silent", "%fail", "%lamba")
    print("%-14s %-8s %-6s %6s %7s %8s %9s %7s %7s" % header)
    print("-" * 70)
    for key in sorted(buckets.keys()):
        g, split, lang = key
        b = buckets[key]
        n = b["n"]
        print("%-14s %-8s %-6s %6d %7.1f %8.1f %9.1f %7.1f %7.1f" % (
            g, split, lang, n,
            pct(b["ok"], n), pct(b["wrong"], n),
            pct(b["silent"], n), pct(b["fail"], n), pct(b["lamba"], n),
        ))


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()

    if args.summary:
        summary(args.summary)
        return

    if not args.model or not args.out:
        ap.error("usage: script <model> <out.jsonl> [--limit N]")

    run(args.model, args.out, args.limit, args.split)


if __name__ == "__main__":
    main()
