#!/usr/bin/env python3
"""Rumination metric v2 (captain step 1), family-aware.

The old classifier counted ANY tool call after the correct value first appeared as "overran",
but every eval prompt instructs "compute exactly, then verify by <method>", so the first
verification is REQUIRED, not rumination.

DONE point = exact value seen in a tool output AND a successful (no-error) call whose CODE uses
the task family's independent verification method has completed the check:
  - value-reproducing families (integral/ode/series/limit/combinatorics/matrix): the check
    must reproduce the value -- >=2 DISTINCT matching numbers in one call, or >=1 after the
    value was already seen in an earlier call. (Exact value printed twice is not a check.)
  - residual families (linear_system): the check is substitution/residuals, which prints
    zeros not the value -- so a successful substitution call after/with the value suffices.
Redundant calls = tool calls issued AFTER the done point. Every rule here was added after
hand-inspecting real transcripts (crashed checks credited as checks; evalf/N mistaken for
checks; exact value printed twice mistaken for a check).

Primary: P(redundant>=1 | done), with a TASK-CLUSTERED bootstrap 95% CI.
Usage: rumination_metric_v2.py [--prompts rumination_eval30.jsonl] file1.jsonl [file2.jsonl ...]
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict

NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?")
ERR_RE = re.compile(r"ERROR:|Traceback|Error:|error:")
VERIFY_RE = {
    # evalf()/N() just print the exact answer as a decimal -- deliberately NOT verification.
    "integral": r"quad|riemann|simpson|trapz|trapezoid|midpoint|nintegrate|monte|linspace|Integral\(|sum\(|for \w+ in range",
    "ode": r"solve_ivp|odeint|rk4|runge|euler|for \w+ in range|while ",
    "series": r"sum\(|for \w+ in range|nsum|partial",
    "limit": r"1e-|10\*\*|subs\(|lambdify|for \w+ in",
    "combinatorics": r"itertools|combinations|permutations|product\(|for \w+ in",
    "matrix": r"numpy|np\.|linalg",
    "linear_system": r"subs\(|residual|\.dot|@|np\.|numpy|linalg",
}
RESIDUAL_FAMILIES = {"linear_system"}
# A matching number counts as a CHECK result only on an output line labeled as a numerical
# method, not as the exact/symbolic value. (Found by hand-inspection, HO_r1 rep0: the exact value
# printed at two precisions was being read as exact+check while the real Riemann check was buggy.)
CHECK_LINE_RE = re.compile(r"riemann|midpoint|trapez|simpson|quad|numer|approx|partial|rk4|runge|"
                           r"euler|solve_ivp|odeint|brute|enumerat|count|monte|simulat|numpy|np\b|linalg|"
                           r"x ?= ?1e|at x|residual", re.I)
EXACT_LINE_RE = re.compile(r"exact|symbolic|sympy|closed|simplif|manual|evalf|\bN\(|pi/|sqrt\(|log\(|exp\(", re.I)
# If the call's own output signals a FAILED comparison, the model reasonably believes verification
# is not done -- a follow-up repair is legitimate, not rumination (hand-inspection, HO_t2 rep3).
# per-line: any "match/agree/equal/close ... False" (words may sit in between, e.g.
# "Match (tol 1e-9)? = False", "Match quadpack vs symbolic: False") -- fresh-sample FPs.
MISMATCH_RE = re.compile(r"(?:match|agree|equal|close)[^\n]*\bFalse\b|mismatch|do(?:es)? not (?:match|agree)|disagree", re.I)
CHECK_TOL = 1e-6  # a CHECK must agree to 1e-6 relative; 1e-3 is only for "value appeared"


def check_hits(text, gt):
    """matching numbers on lines labeled as a numerical check (and not as exact)"""
    c = 0
    for line in (text or "").splitlines():
        if CHECK_LINE_RE.search(line) and not EXACT_LINE_RE.search(line) and _matches(line, gt, CHECK_TOL):
            c += 1
    return c


# v2.1 (parser bug fix only, no judgment change): also parse exact fractions, "56/9" and LaTeX
# \\frac{56}{9} / \\dfrac / \\tfrac -- correct fraction-form answers were scored wrong, and tool
# outputs printing only the exact rational never registered as "value seen".
FRAC_RE = re.compile(r"\\[dt]?frac\{\s*(-?\d+)\s*\}\{\s*(\d+)\s*\}|(?<![\d.])(-?\d+)\s*/\s*(\d+)(?![\d.])")


# v2.2 (parser fix only): normalize typographic minus signs (U+2212 etc.) to ASCII "-", and move a
# sign written OUTSIDE a LaTeX fraction ("-\\frac{1}{8}") inside it -- correct answers like "−1/8" were
# being scored wrong (found by reading EV_lim3 transcripts; affected all arms, dpov1 most).
_MINUS = str.maketrans({"\u2212": "-", "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\ufe63": "-", "\uff0d": "-"})
_OUTSIDE_SIGN = re.compile(r"-\s*\\([dt]?frac)\{\s*(\d+)")


def _norm(text):
    t = (text or "").translate(_MINUS)
    return _OUTSIDE_SIGN.sub(lambda m: "\\" + m.group(1) + "{-" + m.group(2), t)


def _matches(text, gt, rel=1e-3):
    text = _norm(text)
    tol = rel * max(1.0, abs(gt))
    out = []
    for m in NUM_RE.findall(text or ""):
        try:
            v = float(m.replace(",", ""))
        except ValueError:
            continue
        if abs(v - gt) <= tol:
            out.append(m)
    for a, b, c, d in FRAC_RE.findall(text or ""):
        num, den = (a, b) if a else (c, d)
        try:
            if int(den) != 0 and abs(int(num) / int(den) - gt) <= tol:
                out.append(f"{num}/{den}")
        except ValueError:
            pass
    return out


def analyze(row, gt, family):
    vre = re.compile(VERIFY_RE.get(family, VERIFY_RE["integral"]), re.I)
    tr = row["transcript"]
    seen_value = verified = False
    done_idx = None
    for i, t in enumerate(tr):
        if t.get("role") != "tool":
            continue
        out = t.get("content", "") or ""
        hits = _matches(out, gt)
        ok = not ERR_RE.search(out)
        uses_method = bool(vre.search(t.get("arguments") or ""))
        if uses_method and ok:
            if family in RESIDUAL_FAMILIES:
                if seen_value or hits:
                    verified = True
            elif check_hits(out, gt) >= 1 and (seen_value or hits) and not MISMATCH_RE.search(out):
                verified = True
        if hits:
            seen_value = True
        if seen_value and verified:
            done_idx = i
            break
    redundant = 0
    if done_idx is not None:
        for t in tr[done_idx + 1:]:
            if t.get("role") == "assistant" and t.get("tool_calls"):
                redundant += len(t["tool_calls"])
    chars = sum(len(t.get("content") or "") + len(t.get("reasoning") or "")
                for t in tr if t.get("role") == "assistant")
    return {"task": row["id"], "done": done_idx is not None, "done_idx": done_idx, "redundant": redundant,
            "final_correct": bool(_matches(row.get("final_text", ""), gt)), "chars": chars}


def cluster_ci(res, stat, B=2000, seed=0):
    by = defaultdict(list)
    for r in res:
        by[r["task"]].append(r)
    tasks = list(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        samp = [r for t in (rng.choice(tasks) for _ in tasks) for r in by[t]]
        v = stat(samp)
        if v is not None:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def red_rate(rs):
    d = [r for r in rs if r["done"]]
    return sum(r["redundant"] >= 1 for r in d) / len(d) if d else None


def summarize(fn, tasks):
    rows = [json.loads(l) for l in open(fn)]
    res = [analyze(r, tasks[r["id"]]["gt"], tasks[r["id"]]["family"]) for r in rows if r["id"] in tasks]
    n = len(res)
    done = [r for r in res if r["done"]]
    red = sum(r["redundant"] >= 1 for r in done)
    mean_red = sum(r["redundant"] for r in done) / max(1, len(done))
    fc = sum(r["final_correct"] for r in res)
    lo, hi = cluster_ci(res, red_rate) if done else (float("nan"), float("nan"))
    ntask = len({r["task"] for r in res})
    print(f"{fn.split('/')[-1]}: n={n} ({ntask} tasks) | final_correct {fc/n:.1%} | done {len(done)}/{n} | "
          f"REDUNDANT>=1|done {red}/{len(done)}={red/max(1,len(done)):.1%} [cluster CI {lo:.1%},{hi:.1%}] | "
          f"mean redundant {mean_red:.2f} | mean asst chars {sum(r['chars'] for r in res)/n:.0f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="/home/ubuntu/antidoom_data/rumination_eval30.jsonl")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    tasks = {}
    for l in open(a.prompts):
        t = json.loads(l)
        tasks[t["id"]] = {"gt": t["ground_truth_numeric"], "family": t.get("family", "integral")}
    for fn in a.files:
        summarize(fn, tasks)
