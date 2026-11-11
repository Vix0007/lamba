"""DS #198 Condition B wrapper: monkeypatches srv.run_tool's python_eval branch to use
the rescue-B sandbox instead of the default one, for use ONLY by the rescue-condition
generator script. Does NOT alter web/server.py itself (the live file is untouched)."""
import sys
sys.path.insert(0, "/home/ubuntu/antidoom_data")
sys.path.insert(0, "/home/ubuntu/web")
import server as srv
from sandbox_rescue_b import run_python_sandboxed_rescue_b, SANDBOX_B_CONFIG_HASH
import json as _json


def run_tool_rescue_b(name: str, arguments: str) -> str:
    print(f"[tool_call rescueB] {name}({arguments})", flush=True)
    try:
        args = _json.loads(arguments) if arguments else {}
    except Exception:
        args = {}
    if name == "python_eval":
        return run_python_sandboxed_rescue_b(str(args.get("code", "")))
    if name == "web_search":
        return srv.run_tool(name, arguments)  # unchanged, not part of this mechanism test
    return f"error: unknown tool '{name}'"
