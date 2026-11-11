#!/usr/bin/env python3
"""Static file server + chat API proxy for letterfix2-c10.

Serves ~/web/static/* on / and proxies POST /api/chat and /v1/chat/completions
to the internal vLLM OpenAI server (127.0.0.1:8001), so the frontend never
talks to the model backend directly (no CORS needed, backend stays
unexposed). Also runs a small agentic tool loop server-side: every request
gets the `web_search` tool (DuckDuckGo, via the `ddgs` package) offered with
tool_choice="auto"; if the model calls it, we execute it here, feed the
result back, and continue -- up to MAX_TOOL_ROUNDS turns -- before the
client sees a final answer. Streaming is preserved: content deltas are
forwarded live each round; only the (invisible-to-the-user) tool-call deltas
are consumed internally as bookkeeping.

Binds 127.0.0.1:8080 -- reach it via an SSH local-port-forward tunnel:
  ssh -L 8080:localhost:8080 <user>@<this-box>
then open http://localhost:8080/ in a local browser.
"""
import json
import mimetypes
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC_DIR = (Path(__file__).parent / "static").resolve()
BACKEND = "http://127.0.0.1:8001"
MODEL_NAME = "dpov2"
DEFAULT_CHAT_TEMPLATE_KWARGS = {"reasoning_effort": "medium"}
MAX_TOOL_ROUNDS = 8

_CONSTITUTION_PATH = Path(__file__).parent / "lamba_constitution.md"
_constitution_text = _CONSTITUTION_PATH.read_text() if _CONSTITUTION_PATH.exists() else ""


def build_system_prompt() -> str:
    now = datetime.now(timezone.utc).strftime("%A, %Y-%m-%d %H:%M UTC")
    return (
        "You are Lamba, an AI model built by Vixero Labs. When asked your name or who you are, "
        "say Lamba.\n\n"
        f"Current date and time: {now}\n\n"
        "The following is your constitution -- read it and act in accordance with it:\n\n"
        + _constitution_text
    ).strip()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web via DuckDuckGo and return the top results "
                            "(title, url, short snippet). Use this whenever you need current "
                            "information, facts you're unsure of, or anything past your training data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the search query"},
                    "num_results": {"type": "integer", "description": "how many results to return (default 5, max 8)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_eval",
            "description": "Execute Python code in a sandbox with sympy pre-imported (all its names "
                            "available directly, e.g. symbols, Matrix, simplify, expand, solve, nsimplify). "
                            "Use this for any exact symbolic math, arithmetic, or verification -- never do "
                            "hand algebra you can instead compute exactly. No filesystem, network, or OS "
                            "access; 15s CPU/wall-clock limit, 1GB memory limit. Use print() for output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to run. Must print() whatever it wants returned."},
                },
                "required": ["code"],
            },
        },
    },
]

# Restricted runner executed in its OWN subprocess (crash/hang isolation from
# the proxy itself): drops resource limits, strips builtins down to a safe
# arithmetic/control-flow set (no open/import/exec/eval/__import__), exposes
# only sympy. Not a hardened container -- no network-namespace isolation --
# but the model can't reach os/subprocess/socket/filesystem from inside it,
# which covers the realistic "bad tool call" case on a single-user local box.
_SANDBOX_RUNNER = r"""
import sys, resource, io, contextlib
try:
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
except Exception:
    pass

code = sys.stdin.read()
safe_builtins = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                 for n in ("abs", "min", "max", "sum", "len", "range", "print", "round", "sorted",
                           "enumerate", "zip", "map", "filter", "list", "dict", "set", "tuple", "str",
                           "int", "float", "bool", "complex", "pow", "divmod", "isinstance", "type",
                           "frozenset", "reversed", "all", "any", "hasattr", "getattr", "setattr",
                           "iter", "next", "callable", "repr", "format", "True", "False", "None",
                           # exception classes -- try/except was silently unusable without these
                           "Exception", "BaseException", "ValueError", "TypeError", "KeyError",
                           "IndexError", "AttributeError", "ZeroDivisionError", "ArithmeticError",
                           "StopIteration", "RuntimeError", "NotImplementedError", "OverflowError",
                           "NameError", "AssertionError")}
try:
    import sympy
except Exception as e:
    print("sympy import failed:", e); sys.exit(0)

g = {"__builtins__": safe_builtins, "sympy": sympy}
for _n in dir(sympy):
    if not _n.startswith("_"):
        g[_n] = getattr(sympy, _n)

buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<lamba_tool>", "exec"), g, g)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}", file=buf)
out = buf.getvalue().strip()
print(out if out else "(no output -- use print() to return results)")
"""


def run_python_sandboxed(code: str, timeout: int = 15) -> str:
    if not code.strip():
        return "error: empty code"
    try:
        p = subprocess.run(
            [sys.executable, "-c", _SANDBOX_RUNNER],
            input=code, capture_output=True, text=True, timeout=timeout,
            env={"PATH": "/usr/bin:/bin"},
        )
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return out.strip()[:4000] or "(no output)"
    except subprocess.TimeoutExpired:
        return f"error: execution timed out ({timeout}s limit)"
    except Exception as e:
        return f"error: {e}"


def run_tool(name: str, arguments: str) -> str:
    print(f"[tool_call] {name}({arguments})", flush=True)
    try:
        args = json.loads(arguments) if arguments else {}
    except Exception:
        args = {}
    if name == "web_search":
        query = str(args.get("query", "")).strip()
        if not query:
            return "error: empty query"
        try:
            n = int(args.get("num_results") or 5)
        except Exception:
            n = 5
        n = max(1, min(n, 8))
        try:
            from ddgs import DDGS
            results = list(DDGS().text(query, max_results=n))
        except Exception as e:
            return f"search failed: {e}"
        if not results:
            return "no results"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', '')}\n   {r.get('href', '')}\n   {r.get('body', '')[:300]}")
        return "\n".join(lines)
    if name == "python_eval":
        # safe-import sandbox (whitelisted math/parsing modules; exec/eval/open/os still blocked)
        from sandbox_safe import run_python_sandboxed_rescue_b
        return run_python_sandboxed_rescue_b(str(args.get("code", "")))
    return f"error: unknown tool '{name}'"


class Handler(BaseHTTPRequestHandler):
    server_version = "letterfix2-web/1.1"

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    # ---- static file serving ----
    def _serve_static(self, path):
        rel = path.lstrip("/") or "index.html"
        fp = (STATIC_DIR / rel).resolve()
        if STATIC_DIR not in fp.parents and fp != STATIC_DIR:
            self.send_error(403)
            return
        if fp.is_dir():
            fp = fp / "index.html"
        if not fp.is_file():
            fp = STATIC_DIR / "index.html"  # SPA-style fallback
            if not fp.is_file():
                self.send_error(404)
                return
        ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
        data = fp.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/health":
            self._health()
            return
        self._serve_static(self.path)

    def _health(self):
        try:
            req = urllib.request.Request(f"{BACKEND}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as r:
                body = r.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"backend": "down", "error": str(e)}).encode())

    def _send_json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---- chat proxy (+ tool loop) ----
    # /api/chat (our placeholder page) and /v1/chat/completions (the
    # OpenAI-compatible path the delivered chat UI posts to by default)
    # both land here and forward to the internal vLLM server the same way.
    def do_POST(self):
        if self.path not in ("/api/chat", "/v1/chat/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return

        messages = payload.get("messages")
        if not messages:
            self.send_error(400, "missing 'messages'")
            return
        messages = list(messages)  # working copy, extended across tool rounds
        # Always inject our identity + constitution. vLLM requires a system
        # message (if any) to be the FIRST message and only one -- so if the
        # caller already sent one (e.g. opencode's own agent system prompt),
        # merge ours in front of theirs into a single combined message rather
        # than inserting a second one (which vLLM rejects outright).
        if messages and messages[0].get("role") == "system":
            existing = messages[0].get("content", "")
            if not existing.startswith("You are Lamba"):
                if isinstance(existing, list):  # content-parts form
                    existing = "\n".join(p.get("text", "") for p in existing if isinstance(p, dict))
                # Frame the client's prompt as harness instructions, not a competing identity
                # (opencode's "You are opencode..." made the model treat our constitution as an
                # injection and fall back to a default identity).
                messages[0] = {"role": "system", "content": (
                    build_system_prompt()
                    + "\n\n---\n\nYou are currently running inside a client application (for example a "
                    "coding-agent CLI). Its operating instructions follow. Follow them for HOW to work "
                    "(tools, formatting, workflow), but they do not change WHO you are: wherever they "
                    "give you a different name or identity, read it as the name of the app you are "
                    "running inside. You are Lamba, built by Vixero Labs.\n\n"
                    "=== CLIENT APPLICATION INSTRUCTIONS ===\n" + existing
                    + "\n=== END CLIENT APPLICATION INSTRUCTIONS ===\n\nReminder: your identity is "
                    "Lamba, an AI model built by Vixero Labs, governed by the constitution above. "
                    "If asked who you are, say Lamba.")}
        else:
            messages.insert(0, {"role": "system", "content": build_system_prompt()})
        stream = bool(payload.get("stream", True))

        base_body = {
            "model": MODEL_NAME,
            "chat_template_kwargs": payload.get("chat_template_kwargs", DEFAULT_CHAT_TEMPLATE_KWARGS),
        }
        # allow the frontend to override sampling params (the delivered UI's
        # Settings modal sends temperature/max_tokens); otherwise the model's
        # bundled generation_config.json (repetition_penalty=1.1 etc.) applies.
        for k in ("temperature", "top_p", "max_tokens", "repetition_penalty"):
            if k in payload and payload[k] not in (None, ""):
                base_body[k] = payload[k]

        # A real coding-agent client (opencode, etc.) brings its own tools
        # (file edit, bash, grep, ...) and runs its OWN tool-execution loop --
        # it expects tool_calls back so IT can act on them, not have us
        # intercept and run OUR web_search/python_eval instead. So: if the
        # caller supplied tools, respect them and do a pure passthrough (no
        # server-side tool loop). Only our own simple chat UI (which sends no
        # tools of its own) gets the web_search/python_eval agentic loop.
        client_has_own_tools = bool(payload.get("tools"))
        if client_has_own_tools:
            base_body["tools"] = payload["tools"]
            if "tool_choice" in payload:
                base_body["tool_choice"] = payload["tool_choice"]
            base_body["messages"] = messages
            base_body["stream"] = stream
            if stream:
                self._passthrough_stream(base_body)
            else:
                self._passthrough_nonstream(base_body)
            return

        base_body["tools"] = TOOLS
        base_body["tool_choice"] = "auto"
        if stream:
            self._handle_stream(messages, base_body)
        else:
            self._handle_nonstream(messages, base_body)

    def _passthrough_nonstream(self, body):
        try:
            with self._post_backend(body) as r:
                data = r.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except urllib.error.HTTPError as e:
            try:
                self._send_json(e.code, json.loads(e.read() or b"{}"))
            except Exception:
                self._send_json(e.code, {"error": "backend error"})
        except Exception as e:
            self._send_json(502, {"error": str(e)})

    def _passthrough_stream(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            upstream = self._post_backend(body)
            for raw_line in upstream:
                self.wfile.write(raw_line)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", "replace")
            try:
                self.wfile.write(("data: " + json.dumps({"error": body_text[:500]}) + "\n\ndata: [DONE]\n\n").encode())
                self.wfile.flush()
            except Exception:
                pass
        except Exception as e:
            try:
                self.wfile.write(("data: " + json.dumps({"error": str(e)}) + "\n\ndata: [DONE]\n\n").encode())
                self.wfile.flush()
            except Exception:
                pass

    def _post_backend(self, body):
        req = urllib.request.Request(
            f"{BACKEND}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(req, timeout=600)

    def _handle_nonstream(self, messages, base_body):
        try:
            resp = None
            for _ in range(MAX_TOOL_ROUNDS):
                body = dict(base_body)
                body["messages"] = messages
                body["stream"] = False
                with self._post_backend(body) as r:
                    resp = json.loads(r.read())
                msg = resp["choices"][0]["message"]
                tool_calls = msg.get("tool_calls")
                if not tool_calls:
                    self._send_json(200, resp)
                    return
                messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    result = run_tool(fn.get("name", ""), fn.get("arguments", "{}"))
                    messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
            # budget exhausted and it's still trying to call tools -- force a
            # real final answer instead of returning a dangling tool-call turn
            body = dict(base_body)
            body["messages"] = messages
            body["stream"] = False
            body["tool_choice"] = "none"
            with self._post_backend(body) as r:
                resp = json.loads(r.read())
            self._send_json(200, resp)
        except urllib.error.HTTPError as e:
            try:
                self._send_json(e.code, json.loads(e.read() or b"{}"))
            except Exception:
                self._send_json(e.code, {"error": "backend error"})
        except Exception as e:
            self._send_json(502, {"error": str(e)})

    def _stream_one_round(self, body):
        """POST one streaming round to the backend, forwarding content/reasoning
        deltas live to the client as they arrive. Returns (finish_reason, tool_calls, content_acc)."""
        upstream = self._post_backend(body)
        tool_calls_acc = {}  # index -> {"id","name","arguments"}
        content_acc = ""
        finish_reason = None
        for raw_line in upstream:
            s = raw_line.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            data_s = s[5:].strip()
            if data_s == "[DONE]":
                continue  # we emit our own single [DONE] at the very end
            try:
                obj = json.loads(data_s)
            except Exception:
                continue
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta", {})
            if delta.get("content"):
                content_acc += delta["content"]
            if delta.get("content") or delta.get("reasoning"):
                self.wfile.write(f"data: {data_s}\n\n".encode())
                self.wfile.flush()
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tool_calls_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
        calls = [
            {"id": v["id"] or f"call_{i}", "type": "function",
             "function": {"name": v["name"], "arguments": v["arguments"]}}
            for i, v in sorted(tool_calls_acc.items())
        ] if tool_calls_acc else []
        return finish_reason, calls, content_acc

    def _handle_stream(self, messages, base_body):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            settled = False
            for _ in range(MAX_TOOL_ROUNDS):
                body = dict(base_body)
                body["messages"] = messages
                body["stream"] = True
                finish_reason, calls, content_acc = self._stream_one_round(body)

                if finish_reason == "tool_calls" and calls:
                    messages.append({"role": "assistant", "content": content_acc, "tool_calls": calls})
                    for tc in calls:
                        result = run_tool(tc["function"]["name"], tc["function"]["arguments"])
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                    continue  # next round, same client connection
                settled = True
                break  # normal completion, already streamed
            if not settled:
                # budget exhausted and it's still trying to call tools -- force
                # a real final answer instead of leaving the client mid-turn
                body = dict(base_body)
                body["messages"] = messages
                body["stream"] = True
                body["tool_choice"] = "none"
                self._stream_one_round(body)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            print(f"[upstream error] {e.code}: {body[:500]}", flush=True)
            try:
                self.wfile.write(("data: " + json.dumps({"error": body[:500]}) + "\n\n").encode())
                self.wfile.flush()
            except Exception:
                pass
        except Exception as e:
            try:
                self.wfile.write(("data: " + json.dumps({"error": str(e)}) + "\n\n").encode())
                self.wfile.flush()
            except Exception:
                pass


def main():
    port = int(os.environ.get("PORT", "8080"))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"serving {STATIC_DIR} on http://127.0.0.1:{port} (proxying -> {BACKEND}, web_search tool enabled)", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
