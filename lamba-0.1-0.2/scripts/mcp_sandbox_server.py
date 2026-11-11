#!/usr/bin/env python3
"""Minimal MCP stdio server exposing ONE tool, python_eval, backed by the exact same sandbox-B
function every eval30 arm used (run_tool_rescue_b). Same name, description and schema as
web/server.py TOOLS, so a Claude Code `claude -p` run sees the identical tool contract.
Used only to run Claude models through a subscription login (no API key)."""
import contextlib
import io
import json
import sys

sys.path.insert(0, "/home/ubuntu/antidoom_data")
sys.path.insert(0, "/home/ubuntu/web")
with contextlib.redirect_stdout(io.StringIO()):  # imports may print; stdout is the RPC channel
    import server as srv  # noqa: E402
    from run_tool_rescue_b_dpov1 import run_tool_rescue_b  # noqa: E402

TOOL = [t for t in srv.TOOLS if t["function"]["name"] == "python_eval"][0]["function"]
MCP_TOOL = {"name": TOOL["name"], "description": TOOL["description"], "inputSchema": TOOL["parameters"]}


def reply(mid, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": mid}
    msg.update({"error": error} if error else {"result": result})
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    try:
        req = json.loads(line)
    except Exception:
        continue
    method, mid = req.get("method"), req.get("id")
    if mid is None:  # notification
        continue
    if method == "initialize":
        reply(mid, {"protocolVersion": req.get("params", {}).get("protocolVersion", "2024-11-05"),
                    "capabilities": {"tools": {}}, "serverInfo": {"name": "sandbox", "version": "1"}})
    elif method == "tools/list":
        reply(mid, {"tools": [MCP_TOOL]})
    elif method == "tools/call":
        p = req.get("params", {})
        with contextlib.redirect_stdout(io.StringIO()):  # run_tool_rescue_b prints a log line
            out = run_tool_rescue_b(p.get("name", ""), json.dumps(p.get("arguments") or {}))
        reply(mid, {"content": [{"type": "text", "text": out}]})
    elif method == "ping":
        reply(mid, {})
    else:
        reply(mid, error={"code": -32601, "message": f"unknown method {method}"})
