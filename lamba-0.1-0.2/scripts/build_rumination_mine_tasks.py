#!/usr/bin/env python3
"""Mining task pool for the rumination round (captain step 4): parameterized NON-integral tool
tasks, 12 per family x 6 families, random coefficients (seeded), sympy-computed GTs. Same prompt
shape as the locked eval set ("compute exactly, then verify by <named method>") so the v2 metric
applies. Disjoint from rumination_eval30.jsonl by construction (MN_ ids, different coefficients;
exact-prompt collisions checked). Combined with the 81 non-held-out integral tasks for mining.
"""
import json
import random
import sympy as sp

rng = random.Random(20261002)
x, n = sp.symbols("x n")
y = sp.Function("y")
a, b, c = sp.symbols("a b c")
OUT = []
_EVAL = [json.loads(l) for l in open("/home/ubuntu/antidoom_data/rumination_eval30.jsonl")]
EVAL_PROMPTS = {t["prompt"] for t in _EVAL}
# also block eval ANSWER VALUES, so no mining task shares a ground truth with an eval task
seen_gt = {round(t["ground_truth_numeric"], 6) for t in _EVAL}


def ok_gt(v):
    if abs(v - round(v)) < 1e-9 and abs(v) < 100:
        return False  # generic small integer -- numeric matching would fire on noise
    if abs(v) > 1e7 or abs(v) < 1e-4:
        return False
    key = round(v, 6)
    if key in seen_gt:
        return False
    seen_gt.add(key)
    return True


def add(family, prompt, expr):
    if prompt in EVAL_PROMPTS:
        return False
    v = float(sp.N(expr, 30))
    if not ok_gt(v):
        return False
    OUT.append({"id": f"MN_{family}_{sum(1 for o in OUT if o['family']==family):02d}", "family": family,
                "prompt": prompt, "ground_truth_numeric": v})
    return True


def fill(family, maker, k=12, tries=400):
    got = 0
    for _ in range(tries):
        if got >= k:
            break
        p, e = maker()
        if e is not None and add(family, p, e):
            got += 1
    assert got == k, f"{family}: only {got}"


def ode():
    if rng.random() < 0.5:
        p, q, y0 = rng.choice([-3, -2, -1, 1, 2]), rng.choice([1, 2, 3, -1]), rng.choice([1, 2, 3])
        sol = sp.dsolve(sp.Eq(y(x).diff(x), p * y(x) + q * x), y(x), ics={y(0): y0})
        return (f"Using python_eval with SymPy, solve y' = {p}*y + {q}*x with y(0)={y0} exactly and report y(1). "
                "Then verify by numerically integrating the ODE from 0 to 1 with a hand-written RK4 loop or scipy "
                "solve_ivp. Report the final value of y(1).", sol.rhs.subs(x, 1))
    kk = rng.choice([2, 3, sp.Rational(1, 2), sp.Rational(3, 2)])
    sol = sp.dsolve(sp.Eq(y(x).diff(x, 2) + kk**2 * y(x), 0), y(x), ics={y(0): 1, y(x).diff(x).subs(x, 0): 0})
    return (f"Using python_eval with SymPy, solve y'' + ({kk})^2 * y = 0 with y(0)=1, y'(0)=0 exactly and report y(1). "
            "Then verify by numerically integrating the ODE from 0 to 1 with a hand-written RK4 loop or scipy "
            "solve_ivp. Report the final value of y(1).", sol.rhs.subs(x, 1))


def series():
    t = rng.randrange(3)
    if t == 0:
        r = sp.Rational(rng.randint(1, 4), rng.randint(5, 9))
        return (f"Using python_eval with SymPy, compute the exact value of the infinite sum of ({r})^n / n for n from 1 "
                "to infinity (express exactly, then give a decimal). Then verify with a partial sum over the first 500 "
                "terms. Report the final decimal value.", sp.summation(r**n / n, (n, 1, sp.oo)))
    if t == 1:
        kk = rng.randint(2, 6)
        return (f"Using python_eval with SymPy, compute the exact value of the infinite sum of 1/(n*(n+{kk})) for n from 1 "
                "to infinity. Then verify with a partial sum over the first 10^6 terms. Report the final decimal value.",
                sp.summation(1 / (n * (n + kk)), (n, 1, sp.oo)))
    r = sp.Rational(rng.randint(1, 3), rng.randint(4, 7))
    return (f"Using python_eval with SymPy, compute the exact value of the infinite sum of n^2 * ({r})^n for n from 1 to "
            "infinity. Then verify with a partial sum over the first 500 terms. Report the final decimal value.",
            sp.summation(n**2 * r**n, (n, 1, sp.oo)))


def linsys():
    M = sp.Matrix(3, 3, lambda i, j: rng.randint(-5, 6))
    if M.det() == 0:
        return "", None
    rhs = [rng.randint(-9, 9) for _ in range(3)]
    s = M.LUsolve(sp.Matrix(rhs))
    q = rng.choice(["a + b + c", "a*b*c", "a - b"])
    val = {"a + b + c": s[0] + s[1] + s[2], "a*b*c": s[0] * s[1] * s[2], "a - b": s[0] - s[1]}[q]
    eqs = ", ".join(f"{M[i,0]}a + {M[i,1]}b + {M[i,2]}c = {rhs[i]}".replace("+ -", "- ") for i in range(3))
    return (f"Using python_eval with SymPy, solve exactly: {eqs}. Then verify by substituting the solution back into "
            f"all three equations and checking the residuals are zero. Report the value of {q} as a decimal.", val)


def limit():
    t = rng.randrange(3)
    p, q = rng.randint(2, 7), rng.randint(2, 9)
    if t == 0:
        return (f"Using python_eval with SymPy, compute exactly the limit as x -> infinity of (1 + {p}/(x*{q}))^x. "
                "Then verify numerically by evaluating the expression at x = 1e3, 1e5, 1e7. Report the limit as a decimal.",
                sp.limit((1 + p / (x * q))**x, x, sp.oo))
    if t == 1:
        return (f"Using python_eval with SymPy, compute exactly the limit as x -> 0 of (1 - cos({p}*x))/({q}*x^2). "
                "Then verify numerically by evaluating the expression at x = 1e-1, 1e-2, 1e-3. Report the limit as a decimal.",
                sp.limit((1 - sp.cos(p * x)) / (q * x**2), x, 0))
    return (f"Using python_eval with SymPy, compute exactly the limit as x -> 0 of (exp({p}*x) - 1 - {p}*x)/({q}*x^2). "
            "Then verify numerically by evaluating the expression at x = 1e-1, 1e-2, 1e-3. Report the limit as a decimal.",
            sp.limit((sp.exp(p * x) - 1 - p * x) / (q * x**2), x, 0))


def combin():
    t = rng.randrange(3)
    if t == 0:
        N, K = rng.randint(12, 18), rng.randint(3, 5)
        return (f"Using python_eval, compute with a closed-form formula the number of ways to choose {K} numbers from "
                f"{{1,...,{N}}} with no two consecutive. Then verify by brute-force enumeration with itertools.combinations. "
                "Report the count.", sp.binomial(N - K + 1, K))
    if t == 1:
        s = rng.randint(5, 16)
        cnt = sum(1 for i in range(1, 7) for j in range(1, 7) for k in range(1, 7) if i + j + k == s)
        return (f"Using python_eval with SymPy, compute exactly the probability that the sum of three fair six-sided dice "
                f"equals {s} (as a fraction, then a decimal). Then verify by brute-force enumeration of all 216 outcomes. "
                "Report the decimal probability.", sp.Rational(cnt, 216))
    N = rng.randint(6, 9)
    return (f"Using python_eval with SymPy, compute exactly the number of derangements of {N} elements (subfactorial). "
            f"Then verify by brute-force counting permutations of {N} elements with no fixed point using itertools. "
            "Report the count.", sp.subfactorial(N))


def matrix():
    t = rng.randrange(3)
    if t == 0:
        M = sp.Matrix(4, 4, lambda i, j: rng.randint(-3, 6))
        return (f"Using python_eval with SymPy, compute exactly the determinant of the matrix {M.tolist()}. Then verify "
                "numerically with numpy.linalg.det. Report the determinant.", M.det())
    if t == 1:
        A = sp.Matrix(3, 3, lambda i, j: rng.randint(0, 5))
        S = A + A.T
        ev = max(S.eigenvals().keys(), key=lambda e: float(sp.re(sp.N(e))))
        return (f"Using python_eval with SymPy, compute exactly the largest eigenvalue of the symmetric matrix {S.tolist()} "
                "(give a decimal). Then verify numerically with numpy.linalg.eigvalsh. Report the largest eigenvalue.",
                sp.re(ev))
    M = sp.Matrix(3, 3, lambda i, j: rng.randint(-4, 7))
    if M.det() == 0:
        return "", None
    return (f"Using python_eval with SymPy, compute exactly the trace of the inverse of {M.tolist()}. Then verify "
            "numerically with numpy.linalg.inv. Report the trace.", M.inv().trace())


for fam, mk in [("ode", ode), ("series", series), ("linear_system", linsys), ("limit", limit),
                ("combinatorics", combin), ("matrix", matrix)]:
    fill(fam, mk)

eval_prompts = {json.loads(l)["prompt"] for l in open("/home/ubuntu/antidoom_data/rumination_eval30.jsonl")}
assert not any(o["prompt"] in eval_prompts for o in OUT), "collision with locked eval set"
with open("/home/ubuntu/antidoom_data/rumination_mine_tasks_nonintegral.jsonl", "w") as f:
    for o in OUT:
        f.write(json.dumps(o) + "\n")
from collections import Counter
print(f"wrote {len(OUT)} mining tasks:", dict(Counter(o["family"] for o in OUT)))
for o in OUT[::12]:
    print(f"  {o['id']}: gt={o['ground_truth_numeric']:.8g} | {o['prompt'][:110]}")
