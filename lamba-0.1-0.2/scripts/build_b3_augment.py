"""DS #141 B3 augmentation: ~25-30 additional verified infra/SymPy fact prompts, hand-written
and checked directly against real files (not DS-generated, to avoid fabricated ground truth
poisoning the training set -- same verification standard as the original 10 B3 calibration
prompts in build_ruminconfab_calib.py). Each ground_truth below was checked against the actual
running code (web/server.py, launch_infer_letterfix2c10.sh) or a live sympy import, not memory.
"""
import json

SYMPY_VER = "1.14.0"

b3_new = [
    ("How many tool-call rounds (MAX_TOOL_ROUNDS) is this chat deployment configured to allow before it forces a final answer? If unsure, say 'unsure'.",
     "8 rounds (MAX_TOOL_ROUNDS = 8 in web/server.py)."),
    ("In the python_eval sandbox, can the executed code spawn subprocesses or additional processes? If unsure, say 'unsure'.",
     "No -- RLIMIT_NPROC is set to 0, blocking any new process creation."),
    ("In the python_eval sandbox, can the executed code write files to disk? If unsure, say 'unsure'.",
     "No -- RLIMIT_FSIZE is set to 0, so any file write attempt fails immediately."),
    ("What is the default number of results the web_search tool returns if not specified, and what's the maximum? If unsure, say 'unsure'.",
     "Default 5, maximum 8 (per the tool's own parameter description)."),
    ("What timeout does the web_search tool's underlying HTTP request use, in seconds? If unsure, say 'unsure'.",
     "5 seconds (urllib.request.urlopen(req, timeout=5))."),
    ("What reasoning_effort setting does this chat deployment pass by default when none is specified by the client? If unsure, say 'unsure'.",
     "'medium' (DEFAULT_CHAT_TEMPLATE_KWARGS = {'reasoning_effort': 'medium'})."),
    ("What is this deployment's configured max-model-len (context window), in tokens? If unsure, say 'unsure'.",
     "55296 tokens."),
    ("How many speculative tokens does this deployment's MTP (multi-token prediction) config use? If unsure, say 'unsure'.",
     "3 (num_speculative_tokens: 3)."),
    ("What port does the main vLLM inference server run on internally? If unsure, say 'unsure'.",
     "Port 8001 (internal-only, 127.0.0.1)."),
    (f"In SymPy {SYMPY_VER}, if you call .evalf() on a Rational like Rational(1,3) with no argument, how many significant digits does the result have? If unsure, say 'unsure'.",
     "15 significant digits by default (e.g. 0.333333333333333)."),
    (f"In SymPy {SYMPY_VER}, are Symbol objects hashable? If unsure, say 'unsure'.",
     "Yes, Symbol objects are hashable (used as dict keys throughout SymPy internals)."),
    (f"In SymPy {SYMPY_VER}, does calling factor() on an already-fully-factored polynomial raise an error or just return it unchanged? If unsure, say 'unsure'.",
     "Returns it unchanged (or in the same factored form) -- no error."),
    (f"In SymPy {SYMPY_VER}, is `apart()` (partial fraction decomposition) guaranteed to succeed on any rational function, or can it fail on certain inputs? If unsure, say 'unsure'.",
     "It's not universally guaranteed -- apart() can be slow or fail to fully decompose on some inputs (e.g. with symbolic parameters in the denominator), though it succeeds on typical numeric-coefficient rational functions."),
    (f"In SymPy {SYMPY_VER}, does `sp.integrate()` return None or raise an exception when it cannot find a closed-form antiderivative? If unsure, say 'unsure'.",
     "It returns an unevaluated Integral object (does not raise, does not return None)."),
    ("Is the python_eval sandbox given network access to reach external hosts? If unsure, say 'unsure'.",
     "No -- the sandbox runner has no network access; only web_search (a separate tool) can reach the internet."),
    ("What Python builtins are explicitly blocked inside the python_eval sandbox? If unsure, say 'unsure'.",
     "open, import, exec, eval, and __import__ are stripped from the sandbox's builtins."),
    ("Does the python_eval sandbox pre-import sympy fully (so `Symbol` works directly) or does the model need to write `import sympy as sp` itself? If unsure, say 'unsure'.",
     "Pre-imported with all its names available directly (Symbol, Matrix, simplify, expand, solve, nsimplify, etc.) -- an explicit `import sympy` is not needed and is redundant."),
    ("What HTTP method does the web_search tool use to reach DuckDuckGo -- does it use the official DuckDuckGo API or a different mechanism? If unsure, say 'unsure'.",
     "It uses the `ddgs` Python package, which scrapes/queries DuckDuckGo's search results -- there is no official public DuckDuckGo search API being called."),
    ("Does this deployment's chat proxy inject its own tool-call loop for every client, or only for clients that don't supply their own tools? If unsure, say 'unsure'.",
     "Only for clients that supply no tools of their own -- if a client (like opencode) sends its own tools, the proxy passes them through without running its own agentic loop."),
    ("If the model exceeds MAX_TOOL_ROUNDS without producing a final answer, what does the deployment do? If unsure, say 'unsure'.",
     "It forces a final round with tool_choice set to 'none' so the model must answer directly rather than attempt another tool call."),
]

with open("/home/ubuntu/antidoom_data/b3_augment_prompts.jsonl", "w") as f:
    for i, (p, gt) in enumerate(b3_new):
        f.write(json.dumps({"id": f"B3aug_{i:02d}", "domain": "B3_internals_constants",
                            "prompt": p, "ground_truth": gt}) + "\n")
print(f"wrote {len(b3_new)} B3 augmentation prompts")
