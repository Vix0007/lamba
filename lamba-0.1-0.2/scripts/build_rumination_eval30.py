#!/usr/bin/env python3
"""Locked 30-task rumination eval set (captain step 4/8). 12 existing held-out integral tasks
(HO_*, nudge_ab_heldout_prompts.jsonl) + 18 NEW non-integral tasks built here, 3 per family.
Every prompt names its own verification method, so rumination_metric_v2 can distinguish the
REQUIRED check from redundant calls. Ground truths are computed by sympy, never hand-typed.
These 30 task ids must NEVER be used for mining training pairs.
"""
import json
import sympy as sp

x, n, k = sp.symbols("x n k")
y = sp.Function("y")
T = []


def add(tid, family, prompt, gt_expr):
    gt = float(sp.N(gt_expr, 30))
    T.append({"id": tid, "family": family, "prompt": prompt, "ground_truth_numeric": gt,
              "ground_truth_exact": str(sp.nsimplify(gt_expr) if not isinstance(gt_expr, (int, float)) else gt_expr)})


def ode_val(eq, ics, at):
    sol = sp.dsolve(eq, y(x), ics=ics)
    return sp.simplify(sol.rhs.subs(x, at))


# --- ODE: verify by numerical integration (RK4 / Euler loop or scipy solve_ivp) ---
add("EV_ode1", "ode",
    "Using python_eval with SymPy, solve the ODE y'' + y = 0 with y(0)=0, y'(0)=1 exactly and report y(1). "
    "Then verify by numerically integrating the ODE (e.g. a hand-written RK4 loop or scipy solve_ivp) from 0 to 1. "
    "Report the final value of y(1).",
    ode_val(sp.Eq(y(x).diff(x, 2) + y(x), 0), {y(0): 0, y(x).diff(x).subs(x, 0): 1}, 1))
add("EV_ode2", "ode",
    "Using python_eval with SymPy, solve y' = x*y with y(0)=1 exactly and report y(1). Then verify by numerically "
    "integrating the ODE from 0 to 1 with a hand-written RK4 loop or scipy solve_ivp. Report the final value of y(1).",
    ode_val(sp.Eq(y(x).diff(x), x * y(x)), {y(0): 1}, 1))
add("EV_ode3", "ode",
    "Using python_eval with SymPy, solve y' = -2y + x with y(0)=1 exactly and report y(1). Then verify by numerically "
    "integrating the ODE from 0 to 1 with a hand-written RK4 loop or scipy solve_ivp. Report the final value of y(1).",
    ode_val(sp.Eq(y(x).diff(x), -2 * y(x) + x), {y(0): 1}, 1))

# --- Series: verify by a large partial sum ---
add("EV_ser1", "series",
    "Using python_eval with SymPy, compute the exact value of the infinite sum of 1/n^3 + 1/n^2 for n from 1 to infinity "
    "(express exactly, then give a decimal). Then verify with a partial sum over the first 10^6 terms. Report the final decimal value.",
    sp.summation(1 / n**3 + 1 / n**2, (n, 1, sp.oo)))
add("EV_ser2", "series",
    "Using python_eval with SymPy, compute the exact value of the infinite sum of 1/(n*(n+1)*(n+2)) for n from 1 to infinity. "
    "Then verify with a partial sum over the first 10^5 terms. Report the final value.",
    sp.summation(1 / (n * (n + 1) * (n + 2)), (n, 1, sp.oo)))
add("EV_ser3", "series",
    "Using python_eval with SymPy, compute the exact value of the infinite sum of (2/3)^n / n for n from 1 to infinity "
    "(express exactly, then give a decimal). Then verify with a partial sum over the first 200 terms. Report the final decimal value.",
    sp.summation(sp.Rational(2, 3)**n / n, (n, 1, sp.oo)))

# --- Linear systems: verify by substituting the solution back (residual check) ---
a, b, c = sp.symbols("a b c")
s1 = sp.solve([2*a + 3*b - c - 4, a - b + 2*c - 3, a + 2*b + 4*c - 5], [a, b, c])
add("EV_lin1", "linear_system",
    "Using python_eval with SymPy, solve exactly: 2a + 3b - c = 4, a - b + 2c = 3, a + 2b + 4c = 5. "
    "Then verify by substituting the solution back into all three equations and checking the residuals are zero. "
    "Report the value of a + b + c as a decimal.",
    s1[a] + s1[b] + s1[c])
s2 = sp.solve([2*a + b + c - 7, a - b + 2*c - 4, 3*a + 2*b - c - 5], [a, b, c])
add("EV_lin2", "linear_system",
    "Using python_eval with SymPy, solve exactly: 2a + b + c = 7, a - b + 2c = 4, 3a + 2b - c = 5. "
    "Then verify by substituting the solution back into all three equations and checking the residuals are zero. "
    "Report the value of a*b*c.",
    s2[a] * s2[b] * s2[c])
s3 = sp.solve([5*a - 2*b + 3*c - 11, -a + 4*b + c - 3, 2*a + b - 6*c + 4], [a, b, c])
add("EV_lin3", "linear_system",
    "Using python_eval with SymPy, solve exactly: 5a - 2b + 3c = 11, -a + 4b + c = 3, 2a + b - 6c = -4. "
    "Then verify by substituting the solution back into all three equations and checking the residuals are zero. "
    "Report the value of c.",
    s3[c])

# --- Limits: verify by evaluating the expression at points approaching the limit ---
add("EV_lim1", "limit",
    "Using python_eval with SymPy, compute exactly the limit as x -> 0 of (tan(x) - x)/x^3. "
    "Then verify numerically by evaluating the expression at x = 1e-1, 1e-2, 1e-3. Report the limit value.",
    sp.limit((sp.tan(x) - x) / x**3, x, 0))
add("EV_lim2", "limit",
    "Using python_eval with SymPy, compute exactly the limit as x -> infinity of (1 + 3/x)^x. "
    "Then verify numerically by evaluating the expression at x = 1e3, 1e5, 1e7. Report the limit value as a decimal.",
    sp.limit((1 + 3 / x)**x, x, sp.oo))
add("EV_lim3", "limit",
    "Using python_eval with SymPy, compute exactly the limit as x -> 0 of (sqrt(1+x) - 1 - x/2)/x^2. "
    "Then verify numerically by evaluating the expression at x = 1e-1, 1e-2, 1e-3. Report the limit value.",
    sp.limit((sp.sqrt(1 + x) - 1 - x / 2) / x**2, x, 0))

# --- Combinatorics / probability: verify by brute-force enumeration ---
add("EV_com1", "combinatorics",
    "Using python_eval, compute with a closed-form formula the number of ways to choose 4 numbers from {1,...,12} "
    "with no two consecutive. Then verify by brute-force enumeration with itertools.combinations. Report the count.",
    sp.binomial(9, 4))
add("EV_com2", "combinatorics",
    "Using python_eval with SymPy, compute exactly the number of derangements of 9 elements (subfactorial). "
    "Then verify by brute-force counting permutations of 9 elements with no fixed point using itertools. Report the count.",
    sp.subfactorial(9))
add("EV_com3", "combinatorics",
    "Using python_eval with SymPy, compute exactly the probability that the sum of three fair six-sided dice equals 11 "
    "(as a fraction, then a decimal). Then verify by brute-force enumeration of all 216 outcomes. Report the decimal probability.",
    sp.Rational(27, 216))

# --- Matrices: verify with an independent numerical method (numpy) ---
M1 = sp.Matrix([[3, 1, 0, 2], [1, 4, 2, 0], [0, 2, 5, 1], [2, 0, 1, 6]])
add("EV_mat1", "matrix",
    "Using python_eval with SymPy, compute exactly the determinant of the matrix "
    "[[3,1,0,2],[1,4,2,0],[0,2,5,1],[2,0,1,6]]. Then verify numerically with numpy.linalg.det. Report the determinant.",
    M1.det())
M2 = sp.Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 4]])
add("EV_mat2", "matrix",
    "Using python_eval with SymPy, compute exactly the largest eigenvalue of [[2,1,0],[1,3,1],[0,1,4]] "
    "(give a decimal). Then verify numerically with numpy.linalg.eigvalsh. Report the largest eigenvalue.",
    max(M2.eigenvals().keys(), key=lambda e: float(sp.N(e))))
M3 = sp.Matrix([[4, 1, 2], [2, 3, 0], [1, 0, 5]])
add("EV_mat3", "matrix",
    "Using python_eval with SymPy, compute exactly the trace of the inverse of [[4,1,2],[2,3,0],[1,0,5]]. "
    "Then verify numerically with numpy.linalg.inv. Report the trace.",
    M3.inv().trace())

for t in T:
    v = t["ground_truth_numeric"]
    assert not (abs(v - round(v)) < 1e-9 and abs(v) < 100), f"{t['id']} has a generic small-integer GT {v}"
new = T
held = [json.loads(l) for l in open("/home/ubuntu/antidoom_data/nudge_ab_heldout_prompts.jsonl")]
for h in held:
    h.setdefault("family", "integral")
allt = held + new
with open("/home/ubuntu/antidoom_data/rumination_eval30.jsonl", "w") as f:
    for t in allt:
        f.write(json.dumps(t) + "\n")
print(f"wrote {len(allt)} tasks ({len(held)} held-out integral + {len(new)} new)")
for t in new:
    print(f"  {t['id']:9s} {t['family']:14s} gt={t['ground_truth_numeric']:.10g}")
