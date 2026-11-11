"""DS #118 guard: detect repetition_penalty of an eval file and refuse cross-rp comparisons.
Uses the explicit per-row 'repetition_penalty' field when present (loop_test_g.py writes it since
2026-09-27); otherwise infers from the output fingerprint (newlines per 1k chars: rp=1.1 runs
measured 14.6-17.0, rp=1.0 runs 20.4-21.5 on the 150-test). Set ALLOW_RP_MISMATCH=1 to override."""
import os, sys

def detect_rp(rows):
    vals = {r.get("repetition_penalty") for r in rows}
    if None not in vals and len(vals) == 1:
        return vals.pop(), "declared"
    if None not in vals:
        return None, f"MIXED {sorted(vals)}"
    t = sum(len(r.get("text", "")) for r in rows) or 1
    nl = 1000 * sum(r.get("text", "").count("\n") for r in rows) / t
    if nl >= 19.0: return 1.0, f"inferred (newlines/1k={nl:.1f})"
    if nl <= 17.5: return 1.1, f"inferred (newlines/1k={nl:.1f})"
    return None, f"unknown (newlines/1k={nl:.1f})"

def check(name_a, rows_a, name_b, rows_b):
    ra, sa = detect_rp(rows_a); rb, sb = detect_rp(rows_b)
    print(f"rp: A={ra} [{sa}]  B={rb} [{sb}]")
    bad = ("MIXED" in sa or "MIXED" in sb) or (ra is not None and rb is not None and ra != rb)
    if bad and os.environ.get("ALLOW_RP_MISMATCH") != "1":
        sys.exit(f"REFUSING: repetition_penalty mismatch between {name_a} and {name_b} "
                 f"(set ALLOW_RP_MISMATCH=1 only if intentional)")
    if bad: print("WARNING: rp mismatch allowed by ALLOW_RP_MISMATCH=1")
