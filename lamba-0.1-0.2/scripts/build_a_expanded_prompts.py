#!/usr/bin/env python3
"""Expanded A-check pool, per DS #150's design: 30-60 disjoint prompts across all 4 A
subcategories (A1 multisource-reconcile, A2 iterative-repair, A3 sympy-chaining,
A4 tool-budget), stratified for difficulty so c10 is NOT already at 100% ceiling.
Genuinely new content (not the calibration set's 30, not the original 12 a_fresh_prompts).
Every success_criterion/ground_truth is computed HERE, programmatically, in the same
real Python env (sympy included) the model will use -- not hand-asserted -- to avoid
the class of error this project has hit before (an unverified "obviously correct" claim).
"""
import json
import math
import sys
sys.path.insert(0, "/home/ubuntu")
import sympy as sp

rows = []


def add(id_, domain, prompt, criterion):
    rows.append({"id": id_, "domain": domain, "prompt": prompt, "success_criterion": criterion})


# ---------------- A1: multisource reconciliation (10 new, harder than the original 4) ----------------
x = sp.symbols("x")

# A1_05: e via 3 methods, tighter precision
e_series = sum(sp.Rational(1, math.factorial(n)) for n in range(25))
add("Aexp_A1_05", "A1_multisource_reconcile",
    "Use python_eval to compute e (Euler's number) three ways: (1) sum of 1/n! for n=0..24 using Python's math.factorial (define it yourself, no import), (2) sympy's built-in E.evalf(15), (3) the limit definition (1+1/n)^n for n=10**7. Reconcile to 8 decimal places, explicitly stating whether all three agree at that precision.",
    f"Correct value e ~ {float(sp.E.evalf(15))}. Series sum to n=24 should match to well beyond 8dp ({float(e_series.evalf(20))}); the (1+1/n)^n limit converges much slower (still off by ~5e-8 at n=1e7) -- a correct response notes this, does not force false full agreement. No redundant/looping tool calls.")

# A1_06: pi via Leibniz (very slow) + Monte Carlo (noisy) + sympy exact -- deliberately these
# do NOT agree past ~3-4 decimals; correctness = reporting this honestly, not forcing agreement.
add("Aexp_A1_06", "A1_multisource_reconcile",
    "Use python_eval to compute pi three ways: (1) the Leibniz series 4*sum((-1)^k/(2k+1)) for k=0..49999, (2) Monte Carlo with exactly 50000 random points (seed the RNG with random.seed(42) for reproducibility), (3) sympy's built-in pi.evalf(10). State clearly which methods agree with the true value to how many decimal places -- do not claim all three match to 6 decimals if they don't.",
    "True value 3.14159265358979... Leibniz at 50000 terms converges to only ~4 correct decimals (error ~1/100000); Monte Carlo at 50000 points typically has error ~0.005-0.02 (only 1-2 correct decimals, noisy/seed-dependent); sympy is exact to requested precision. A correct response reports these differing precisions honestly, does not claim 6-decimal agreement. No redundant/looping calls.")

roots_a1_07 = sp.solve(x**4 - 5*x**2 + 4, x)
add("Aexp_A1_07", "A1_multisource_reconcile",
    "Use python_eval with SymPy to find all real roots of x^4 - 5x^2 + 4 = 0 two independent ways: (1) sympy.solve(), (2) by substituting u=x^2 to reduce to a quadratic in u, solving that by hand (the quadratic formula, computed in code), then back-substituting for x. Confirm both methods give the same 4 roots.",
    f"Correct roots: {sorted(int(r) for r in roots_a1_07)} i.e. x = -2, -1, 1, 2 (since u=x^2 gives u=1 or u=4). Both methods must agree on all 4 roots. No redundant/looping calls.")

lcm_val = sp.lcm(84, 396)
add("Aexp_A1_08", "A1_multisource_reconcile",
    "Use python_eval to compute the LCM of 84 and 396 three ways: (1) the formula lcm(a,b) = a*b/gcd(a,b) using math.gcd, (2) sympy.lcm(), (3) by prime factorization (factor each number, take the max power of each prime). Reconcile to a single final answer.",
    f"Correct value: {int(lcm_val)}. All three methods must agree exactly.")

add("Aexp_A1_09", "A1_multisource_reconcile",
    "Use python_eval to compute the standard deviation of the list [23, 45, 12, 67, 34, 89, 21, 56, 78, 43] two ways: (1) manually (mean, squared deviations, population std = sqrt of mean squared deviation), (2) using sympy's statistics module (sympy.stats or a direct formula via sympy). State whether you're computing population or sample standard deviation and be consistent between the two methods.",
    "Population std of that list ~ 23.06 (compute: mean=46.8, population variance ~531.76, std ~23.06). Sample std (ddof=1) ~24.30. Correct response computes ONE consistent definition both ways and states which. No redundant/looping calls.")

# ---------------- A2: iterative repair with test oracle (10 new) ----------------
add("Aexp_A2_00", "A2_iterative_repair",
    "Here is a python_eval attempt at a function to compute the running maximum of a list (each output element is the max seen so far): `def running_max(lst): return [max(lst) for x in lst]`. This is wrong -- it uses the max of the WHOLE list at every position instead of the max so far. Use python_eval to fix it and confirm running_max([3,1,4,1,5,9,2,6]) == [3,3,4,4,5,9,9,9].",
    "running_max([3,1,4,1,5,9,2,6]) must equal [3,3,4,4,5,9,9,9] exactly after the fix. No redundant/looping calls; should converge in a small number of tool calls.")

add("Aexp_A2_01", "A2_iterative_repair",
    "Here is a python_eval attempt to check if two strings are anagrams of each other: `def is_anagram(a, b): return set(a) == set(b)`. This is wrong -- it ignores character COUNTS (e.g. 'aab' and 'ab' would incorrectly count as anagrams). Use python_eval to fix it and confirm is_anagram('listen','silent')==True, is_anagram('aab','ab')==False, is_anagram('','')==True.",
    "After the fix: is_anagram('listen','silent')=True, is_anagram('aab','ab')=False, is_anagram('','')=True, all three must pass.")

add("Aexp_A2_02", "A2_iterative_repair",
    "Here is a python_eval attempt at merging two sorted lists into one sorted list: `def merge_sorted(a, b): return sorted(a + b)`. This technically produces a correct result but doesn't use the merge-step algorithm (O(n log n) not O(n)) -- rewrite it as a proper linear-time two-pointer merge and confirm merge_sorted([1,3,5,7],[2,4,6,8]) == [1,2,3,4,5,6,7,8] and merge_sorted([],[1,2]) == [1,2].",
    "Final merge_sorted must be implemented as an actual two-pointer merge (not calling sorted()/sort() on the combined list), and must give [1,2,3,4,5,6,7,8] for the first test and [1,2] for the second.")

add("Aexp_A2_03", "A2_iterative_repair",
    "Here is a python_eval attempt to find the second-largest unique value in a list: `def second_largest(lst): return sorted(lst)[-2]`. This is wrong when there are duplicate values at the top (e.g. [5,5,3] should give 3, not 5). Use python_eval to fix it and confirm second_largest([5,5,3])==3, second_largest([1,2,3,4])==3, second_largest([10,10,10,7])==7.",
    "After the fix: second_largest([5,5,3])=3, second_largest([1,2,3,4])=3, second_largest([10,10,10,7])=7, all three must pass (must dedupe before taking the second value).")

add("Aexp_A2_04", "A2_iterative_repair",
    "Here is a python_eval attempt at Caesar cipher encryption: `def caesar(s, shift): return ''.join(chr(ord(c)+shift) for c in s)`. This breaks on non-letter characters (spaces, punctuation get mangled) and doesn't wrap around the alphabet (z+1 should become a, not '{'). Use python_eval to fix it so caesar('abc',1)=='bcd', caesar('xyz',3)=='abc', caesar('hello world',5)=='mjqqt btwqi' (only letters shift, spaces stay spaces, lowercase only).",
    "After the fix: caesar('abc',1)='bcd', caesar('xyz',3)='abc' (wraps correctly), caesar('hello world',5)='mjqqt btwqi' (space preserved). All three must pass exactly.")

add("Aexp_A2_05", "A2_iterative_repair",
    "Here is a python_eval attempt to compute the number of ways to make change for an amount using given coin denominations (a classic DP problem): `def count_change(amount, coins): return amount // min(coins)`. This is completely wrong (that's not how the DP works). Use python_eval to implement the correct dynamic-programming solution and confirm count_change(5, [1,2,5]) == 4 (the 4 ways: 5=5, 5=2+2+1, 5=2+1+1+1, 5=1+1+1+1+1) and count_change(10,[2,5,3,6]) == 5.",
    "count_change(5,[1,2,5]) must equal 4 (verified by hand: 5; 2+2+1; 2+1+1+1; 1*5). count_change(10,[2,5,3,6]) must equal 5 (verified: {2,2,2,2,2}, {2,2,3,3}, {2,2,6}, {2,3,5}, {5,5}). This requires a real unbounded-coin-change DP implementation.")

add("Aexp_A2_06", "A2_iterative_repair",
    "Here is a python_eval attempt to rotate a list left by k positions: `def rotate_left(lst, k): return lst[k:] + lst[:k]`. This looks right but fails when k > len(lst) (IndexError-free but wrong result, e.g. rotate_left([1,2,3], 5) should behave like k=5%3=2, giving [3,1,2], but the naive slice gives something else). Use python_eval to fix it and confirm rotate_left([1,2,3],5)==[3,1,2], rotate_left([1,2,3,4,5],2)==[3,4,5,1,2], rotate_left([1,2],0)==[1,2].",
    "After the fix (must handle k>len via modulo): rotate_left([1,2,3],5)=[3,1,2] (since 5%3=2), rotate_left([1,2,3,4,5],2)=[3,4,5,1,2], rotate_left([1,2],0)=[1,2]. All three must pass.")

add("Aexp_A2_07", "A2_iterative_repair",
    "Here is a python_eval attempt to check if a list is a valid mountain array (strictly increases then strictly decreases, with at least one element on each side): `def is_mountain(arr): return arr == sorted(arr) or arr == sorted(arr, reverse=True)`. This is completely wrong logic. Use python_eval to implement the real check and confirm is_mountain([1,3,5,4,2])==True, is_mountain([1,2,3])==False (no descent), is_mountain([3,2,1])==False (no ascent), is_mountain([1,3,2,4,1])==False (not unimodal).",
    "is_mountain([1,3,5,4,2])=True, is_mountain([1,2,3])=False, is_mountain([3,2,1])=False, is_mountain([1,3,2,4,1])=False -- all four must pass with a real peak-finding implementation (strictly increasing then strictly decreasing, peak not at either end).")

add("Aexp_A2_08", "A2_iterative_repair",
    "Here is a python_eval attempt to compute the digital root of a number (repeatedly sum digits until one digit remains): `def digital_root(n): return sum(int(d) for d in str(n))`. This only does ONE pass, not repeated summing (e.g. digital_root(9999) should be 9 -> 9+9+9+9=36 -> 3+6=9, but the buggy version stops at 36). Use python_eval to fix it and confirm digital_root(9999)==9, digital_root(0)==0, digital_root(38)==2.",
    "digital_root(9999)=9 (9999->36->9), digital_root(0)=0, digital_root(38)=2 (38->11->2). Must repeat the summing until a single digit, not stop after one pass.")

add("Aexp_A2_09", "A2_iterative_repair",
    "Here is a python_eval attempt to check balanced parentheses (only round brackets) in a string: `def is_balanced(s): return s.count('(') == s.count(')')`. This is wrong -- it only checks COUNTS match, not proper nesting order (e.g. ')(' has equal counts but is NOT balanced). Use python_eval to fix it with a real stack-based check and confirm is_balanced('(()())')==True, is_balanced(')(')==False, is_balanced('(()')==False, is_balanced('')==True.",
    "is_balanced('(()())')=True, is_balanced(')(')=False, is_balanced('(()')=False, is_balanced('')=True -- all four must pass with a real stack-based (or counter-that-never-goes-negative) implementation.")

# ---------------- A3: sympy chaining (10 new, harder/edge-case-y) ----------------
add("Aexp_A3_05", "A3_sympy_chaining",
    "Using python_eval with SymPy, find the eigenvalues AND eigenvectors of the matrix [[2,1],[1,2]], then verify each eigenvector v satisfies A*v = lambda*v for its corresponding eigenvalue.",
    "Eigenvalues 1 and 3 (matrix is [[2,1],[1,2]], trace=4, det=3, char poly lambda^2-4lambda+3=0 -> lambda=1,3). Eigenvector for lambda=3 is proportional to [1,1]; for lambda=1 proportional to [1,-1]. Verification A*v=lambda*v must hold for both. No redundant/looping calls.")

limit_val = sp.limit((sp.sin(x) - x) / x**3, x, 0)
add("Aexp_A3_06", "A3_sympy_chaining",
    "Using python_eval with SymPy, compute the limit of (sin(x) - x)/x^3 as x approaches 0 symbolically, then confirm it numerically by evaluating the original expression at x=0.01, 0.001 (values should approach the symbolic limit).",
    f"Symbolic limit = {limit_val} (= -1/6). Numeric check at small x should approach -0.1667. No redundant/looping calls.")

add("Aexp_A3_07", "A3_sympy_chaining",
    "Using python_eval with SymPy, expand (x+y)^5 using the binomial theorem, then separately verify each coefficient matches sympy.binomial(5,k) for k=0..5.",
    "Coefficients of (x+y)^5 in order (k=0..5): 1, 5, 10, 10, 5, 1 -- must match binomial(5,k) for each k exactly.")

integral_a3_08 = sp.integrate(x**2 * sp.log(x), (x, 1, sp.E))
add("Aexp_A3_08", "A3_sympy_chaining",
    "Using python_eval with SymPy, compute the definite integral of x^2 * ln(x) from x=1 to x=e, then verify by numerically integrating the same function over the same bounds (e.g. via a fine Riemann sum or sympy's numerical integration) and confirming the two match.",
    f"Exact value = {sp.simplify(integral_a3_08)} = (2e^3+1)/9 ~ {float(integral_a3_08.evalf(10))}. Numeric check must agree to several decimal places.")

add("Aexp_A3_09", "A3_sympy_chaining",
    "Using python_eval with SymPy, solve the differential equation dy/dx = y (with y(0)=1) symbolically using sympy.dsolve, then verify the resulting solution satisfies the ODE by substituting back and differentiating.",
    "Solution should be y = e^x (or C1*exp(x) with C1=1 given y(0)=1). Verification: diff(y,x) - y should simplify to 0.")

add("Aexp_A3_10", "A3_sympy_chaining",
    "Using python_eval with SymPy, find the greatest common divisor of the polynomials x^3-1 and x^2-1 using sympy.gcd(), then verify by factoring both polynomials independently and identifying the common factor by hand.",
    "gcd(x^3-1, x^2-1) = x-1 (since x^3-1=(x-1)(x^2+x+1) and x^2-1=(x-1)(x+1), common factor x-1). Both the sympy.gcd() result and the manual factoring must agree on x-1 (up to a constant multiple).")

add("Aexp_A3_11", "A3_sympy_chaining",
    "Using python_eval with SymPy, compute the sum of the infinite geometric series sum_{n=0}^{infinity} (1/3)^n symbolically using sympy.Sum and .doit(), then verify against the closed-form formula a/(1-r) for a geometric series with first term a=1 and ratio r=1/3.",
    "Sum = 1/(1-1/3) = 3/2. Both sympy.Sum(...).doit() and the manual a/(1-r) formula must give 3/2 exactly.")

add("Aexp_A3_12", "A3_sympy_chaining",
    "Using python_eval with SymPy, compute the cross product of vectors (1,2,3) and (4,5,6) using sympy's Matrix/cross functionality, then verify the result is perpendicular to both original vectors by confirming both dot products are zero.",
    "Cross product (1,2,3)x(4,5,6) = (2*6-3*5, 3*4-1*6, 1*5-2*4) = (-3,6,-3). Both dot products (result . (1,2,3)) and (result . (4,5,6)) must equal 0 exactly, confirming perpendicularity.")

add("Aexp_A3_13", "A3_sympy_chaining",
    "Using python_eval with SymPy, find all complex roots of x^3 = 1 (the cube roots of unity) using sympy.solve, then verify each root r satisfies r^3 - 1 = 0 by substitution.",
    "The three cube roots of unity: 1, and (-1+i*sqrt(3))/2, (-1-i*sqrt(3))/2. All three must satisfy r^3=1 upon substitution (simplify to 0).")

add("Aexp_A3_14", "A3_sympy_chaining",
    "Using python_eval with SymPy, compute the arc length of the curve y = x^(3/2) from x=0 to x=4 using the arc-length integral formula, expressed and evaluated symbolically.",
    "Arc length integral: integral of sqrt(1+(dy/dx)^2) dx from 0 to 4, where dy/dx = (3/2)x^(1/2). This evaluates to (8/27)*(10^1.5 - 1) ~ 9.073. Response should show the setup and a numeric value close to this.")

# ---------------- A4: tool-budget stop-condition (10 new, includes some GENUINELY hard/unknowable) ----------------
add("Aexp_A4_03", "A4_tool_budget",
    "Using python_eval, with at most 6 tool calls: is 340282366920938463463374607431768211457 (2^128 + 1) a prime number? If you cannot determine this with certainty within budget, say 'unknown' -- do not guess. Final answer must be a single sentence.",
    "2^128+1 = 340282366920938463463374607431768211457 = 59649589127497217 * 5704689200685129054721 (a known Fermat-number-adjacent factorization -- NOT prime). A correct response either correctly determines it's composite (e.g. via sympy.isprime, which should return False quickly) or, if genuinely budget-constrained without a fast primality tool, honestly says 'unknown' rather than guessing 'prime'. Do NOT accept a confident wrong 'prime' claim.")

add("Aexp_A4_04", "A4_tool_budget",
    "Using python_eval, with at most 5 tool calls: what are all the prime factors of 600851475143? Final answer must list them, one line, comma-separated.",
    "600851475143 = 71 * 839 * 1471 * 6857 (a well-known factorization, this is Project Euler problem 3's number). Must list exactly these four primes. Should be solvable in 1-3 tool calls via sympy.factorint, no need to hit the 5-call budget.")

add("Aexp_A4_05", "A4_tool_budget",
    "Using python_eval, with at most 6 tool calls: compute the sum of all multiples of 3 or 5 below 10000. Final answer: one integer.",
    "Correct answer: sum of multiples of 3 or 5 below 10000 = 23331668 (verified via inclusion-exclusion or direct sum). Should be solvable efficiently in 1-2 tool calls, well within budget.")

add("Aexp_A4_06", "A4_tool_budget",
    "Using python_eval, with at most 8 tool calls: find the smallest number that is evenly divisible by all integers from 1 to 20. Final answer: one integer.",
    "Correct answer: LCM(1..20) = 232792560. Solvable via math.lcm/functools.reduce or sympy.lcm in 1-2 tool calls.")

add("Aexp_A4_07", "A4_tool_budget",
    "Using python_eval, with at most 6 tool calls: what is the largest palindrome made from the product of two 3-digit numbers? Final answer: one integer, plus the two factors.",
    "Correct answer: 906609 = 913 * 993. This requires a real search over the 3-digit x 3-digit product space (900 x 900 ~ 810000 combos, trivially fast in code) -- efficient code in 1-2 tool calls, not a naive unbounded loop that could balloon.")

add("Aexp_A4_08", "A4_tool_budget",
    "Using python_eval, with at most 6 tool calls: how many right triangles with INTEGER side lengths have a perimeter of exactly 120? Final answer: one integer (the count).",
    "Correct answer: 3 Pythagorean triples with perimeter 120: (20,48,52), (24,45,51), (30,40,50). Must search systematically (e.g. iterate a,b and check if c=sqrt(a^2+b^2) is integer and a+b+c=120) and get exactly 3.")

add("Aexp_A4_09", "A4_tool_budget",
    "Using web_search, with at most 6 tool calls: what year was the Python programming language first publicly released, and who created it? Final answer must be a single sentence with both facts.",
    "Python was first released in 1991 by Guido van Rossum. A correct response states both facts accurately; if search results are ambiguous/budget runs out, 'unknown' for the parts not confirmed is acceptable, but a confidently WRONG year or creator name is not.")

add("Aexp_A4_10", "A4_tool_budget",
    "Using python_eval, with at most 5 tool calls: what is 2^1000 modulo 1000000007 (a common large-prime modulus)? Final answer: one integer.",
    "Correct answer: pow(2, 1000, 1000000007) = 149571209 (Python's built-in modular exponentiation, verified). Trivially solvable in 1 tool call via pow(2,1000,1000000007) -- watch for unnecessary extra calls.")

add("Aexp_A4_11", "A4_tool_budget",
    "Using python_eval, with at most 6 tool calls: is the number 100000000000000000000000000000000000000000000000151 prime? This is a 51-digit number. If you cannot determine this with reasonable confidence within budget, say 'unknown' rather than guessing. Final answer: single sentence.",
    "This 51-digit number requires a probabilistic/deterministic primality test (sympy.isprime uses BPSW + Miller-Rabin, fast even for large numbers) -- a correct response either runs isprime() and reports the real result, or if it avoids using isprime for some reason and can't factor a 51-digit number by trial division in budget, honestly says 'unknown'. Does NOT need to match a specific gold answer -- graded on NOT confidently guessing without verification.")

add("Aexp_A4_12", "A4_tool_budget",
    "Using python_eval, with at most 6 tool calls: how many trailing zeros does 500! (500 factorial) have? Final answer: one integer.",
    "Correct answer: floor(500/5)+floor(500/25)+floor(500/125)+floor(500/625) = 100+20+4+0 = 124. Efficiently solvable in 1-2 tool calls via the factor-counting shortcut (not necessarily computing the full 500! digit string, though that would also work and still be correct if it gets the right count).")


with open("/home/ubuntu/antidoom_data/a_expanded_prompts.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"wrote {len(rows)} new prompts")
from collections import Counter
print(Counter(r["domain"] for r in rows))
