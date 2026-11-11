#!/usr/bin/env python3
"""Local OpenAI-compatible proxy -> Cloudflare Workers AI @cf/qwen/qwen3.8-27b (stock Qwen3.8-27B), so our existing eval
harnesses run unchanged against stock Qwen (EVAL_API=http://127.0.0.1:8011 or --port 8011). 127.0.0.1 only.
- /v1/models, /v1/chat/completions; drops vLLM-only fields (chat_template_kwargs, top_k, min_p, ...); n>1 -> n calls;
  retries 408/429/5xx; reasoning_content -> reasoning.
usage: cf_openai_proxy.py [--port 8011]"""
import json, os, sys, time, threading, urllib.request, urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
PORT = int(sys.argv[sys.argv.index("--port")+1]) if "--port" in sys.argv else 8011
TOKEN = open(os.path.expanduser("~/.cf_api_token")).read().strip()
URL = "https://api.cloudflare.com/client/v4/accounts/d31ce834e08f306396926f27800eea01/ai/run/@cf/qwen/qwen3.8-27b"
KEEP = {"messages", "max_tokens", "temperature", "top_p", "tools", "tool_choice", "repetition_penalty", "presence_penalty", "frequency_penalty", "seed", "stop"}
def cf_call(body, tries=8):
    data = json.dumps(body).encode()
    for a in range(tries):
        try:
            req = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read()); return d.get("result", d)
        except urllib.error.HTTPError as e:
            txt = e.read()[:300]
            if e.code in (408, 429, 500, 502, 503, 504) and a < tries - 1: time.sleep(min(6 * (a + 1), 60)); continue
            raise RuntimeError(f"HTTP {e.code}: {txt!r}")
        except Exception:
            if a < tries - 1: time.sleep(6); continue
            raise
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/v1/models"): return self._send(200, {"object": "list", "data": [{"id": "qwen38cf", "object": "model"}]})
        self._send(404, {"error": "not found"})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path.startswith("/v1/completions"):  # raw prompt (CF 'raw': true -> vLLM text_completion)
            body = {k: v for k, v in req.items() if k not in ("model", "chat_template_kwargs", "stream", "n")}; body["raw"] = True
            n = int(req.get("n", 1) or 1); choices = []; last = {}
            try:
                for i in range(n):
                    r = cf_call(body); last = r
                    for c in r.get("choices", []): choices.append({**c, "index": len(choices)})
            except Exception as e:
                return self._send(502, {"error": str(e)[:300]})
            return self._send(200, {"id": last.get("id", "cf"), "object": "text_completion", "model": "qwen38cf", "choices": choices, "usage": last.get("usage", {})})
        if not self.path.startswith("/v1/chat/completions"): return self._send(404, {"error": "not found"})
        body = {k: v for k, v in req.items() if k in KEEP}
        if "seed" in body and body.get("temperature", 1) == 0: body.pop("seed")
        n = int(req.get("n", 1) or 1); choices = []; last = {}
        try:
            for i in range(n):
                r = cf_call(body); last = r
                for c in r.get("choices", []):
                    m = dict(c.get("message", {}))
                    if m.get("reasoning_content") and not m.get("reasoning"): m["reasoning"] = m["reasoning_content"]
                    choices.append({"index": len(choices), "message": m, "finish_reason": c.get("finish_reason")})
        except Exception as e:
            return self._send(502, {"error": str(e)[:300]})
        self._send(200, {"id": last.get("id", "cf"), "object": "chat.completion", "model": "qwen38cf", "choices": choices, "usage": last.get("usage", {})})
if __name__ == "__main__":
    print("cf proxy on", PORT, flush=True); ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
