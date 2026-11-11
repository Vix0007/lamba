"""Same as run_tool_rescue_b.py but reused generically -- works for any served model
name, just dispatches python_eval to sandbox_rescue_b. No model-specific logic here."""
import sys
sys.path.insert(0, "/home/ubuntu/antidoom_data")
sys.path.insert(0, "/home/ubuntu/web")
import server as srv
from sandbox_rescue_b import run_python_sandboxed_rescue_b
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
        return srv.run_tool(name, arguments)
    return f"error: unknown tool '{name}'"
