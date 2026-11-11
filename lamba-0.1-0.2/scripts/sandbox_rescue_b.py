"""DS #198 Condition B: a SEPARATE, NEW sandbox profile for the rescue experiment --
does NOT mutate web/server.py's live default sandbox (which backs the real production
chat UI and every other eval in this project). Adds a safe `__import__` shim that
resolves ONLY a fixed whitelist of computational/parsing modules; `exec`/`eval`/`open`
remain stripped exactly as in the default sandbox. Logs every blocked import attempt
and any exec/eval attempt into the tool's own output so it's visible in the transcript.

Per DS's explicit note: preloading modules into the namespace is NOT sufficient on its
own, since `import numpy` inside exec'd code calls `builtins.__import__` regardless of
what's already in the namespace -- the shim therefore replaces `__import__` itself,
not just the namespace.
"""
import hashlib

IMPORT_WHITELIST = frozenset({
    "sympy", "numpy", "scipy", "mpmath", "math", "cmath", "fractions", "decimal",
    "statistics", "itertools", "functools", "collections", "re", "json", "random",
    "time",  # v2 (DS #203 optional follow-up): read-only clock access, no fs/net/exec
             # surface, added after seeing a genuine residual model attempt to import it.
})
# Explicit deny list kept for documentation/clarity even though the shim is whitelist-
# based by construction (anything not in IMPORT_WHITELIST is denied by default):
IMPORT_DENYLIST_DOCUMENTED = frozenset({
    "os", "sys", "subprocess", "socket", "ctypes", "importlib", "builtins",
    "pathlib", "shutil", "tempfile", "multiprocessing", "threading", "pickle",
})

_SANDBOX_RUNNER_B = r"""
import sys, resource, io, contextlib, builtins as _real_builtins
try:
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
except Exception:
    pass

code = sys.stdin.read()

IMPORT_WHITELIST = frozenset({
    "sympy", "numpy", "scipy", "mpmath", "math", "cmath", "fractions", "decimal",
    "statistics", "itertools", "functools", "collections", "re", "json", "random",
    "time",
})
_blocked_log = []

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = name.split(".")[0]
    if top not in IMPORT_WHITELIST:
        _blocked_log.append(top)
        raise ImportError(f"import of '{name}' is not permitted in this sandbox profile "
                           f"(rescue-B whitelist: {sorted(IMPORT_WHITELIST)})")
    return _real_builtins.__import__(name, globals, locals, fromlist, level)

safe_builtins = {n: getattr(_real_builtins, n)
                 for n in ("abs", "min", "max", "sum", "len", "range", "print", "round", "sorted",
                           "enumerate", "zip", "map", "filter", "list", "dict", "set", "tuple", "str",
                           "int", "float", "bool", "complex", "pow", "divmod", "isinstance", "type",
                           "frozenset", "reversed", "all", "any", "hasattr", "getattr", "setattr",
                           "iter", "next", "callable", "repr", "format", "True", "False", "None",
                           "Exception", "BaseException", "ValueError", "TypeError", "KeyError",
                           "IndexError", "AttributeError", "ZeroDivisionError", "ArithmeticError",
                           "StopIteration", "RuntimeError", "NotImplementedError", "OverflowError",
                           "NameError", "AssertionError", "ModuleNotFoundError", "ImportError",
                           "dir")  # v2: read-only introspection, never in the default sandbox
                                   # either (not a regression -- genuinely new, zero fs/net/exec
                                   # surface), added after seeing residual NameError('dir') hits.
                 if hasattr(_real_builtins, n)}
safe_builtins["__import__"] = _safe_import
# exec/eval/compile/open/__import__(raw) deliberately NOT added -- same as the default
# sandbox, this shim only changes the import surface, nothing else.

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
        exec(compile(code, "<lamba_tool_rescueB>", "exec"), g, g)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}", file=buf)
out = buf.getvalue().strip()
if _blocked_log:
    out += f"\n[rescue-B: blocked import attempts: {_blocked_log}]"
print(out if out else "(no output -- use print() to return results)")
"""

SANDBOX_B_CONFIG_HASH = hashlib.sha256(_SANDBOX_RUNNER_B.encode()).hexdigest()[:16]


def run_python_sandboxed_rescue_b(code: str, timeout: int = 15) -> str:
    import subprocess, sys as _sys
    if not code.strip():
        return "error: empty code"
    try:
        p = subprocess.run(
            [_sys.executable, "-c", _SANDBOX_RUNNER_B],
            input=code, capture_output=True, text=True, timeout=timeout,
            # OPENBLAS_NUM_THREADS/OMP_NUM_THREADS=1: numpy's OpenBLAS thread pool
            # otherwise collides with RLIMIT_NPROC=(0,0) inside the sandboxed process
            # (pthread_create fails). This is a threading-performance env var, NOT a
            # security-boundary change -- RLIMIT_NPROC stays (0,0), confirmed via
            # smoke test that os/subprocess spawning is still blocked.
            env={"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        )
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return out.strip()[:4000] or "(no output)"
    except subprocess.TimeoutExpired:
        return f"error: execution timed out ({timeout}s limit)"
    except Exception as e:
        return f"error: {e}"


if __name__ == "__main__":
    # smoke test, not run automatically
    print("config hash:", SANDBOX_B_CONFIG_HASH)
    print(run_python_sandboxed_rescue_b("import numpy as np\nprint(np.array([1,2,3]).sum())"))
    print(run_python_sandboxed_rescue_b("import os\nprint(os.listdir('.'))"))
    print(run_python_sandboxed_rescue_b("x = Symbol('x')\nprint(integrate(x**2, (x,0,1)))"))
