#!/usr/bin/env python3
"""Tier-1/Tier-2/Tier-3 rumination guard (DS consults #155, #160, #161).

Narrowed activation per #161: the guard only ACTS (Tier 1) when the conversation is a
symbolic-integration task under an independent-method + same-value verification shape --
the one condition with strong, causally-isolated evidence (domain x arm interaction,
p=2.3e-5, not explained by tool-call volume or sandbox-error friction; see CLAUDE.md
2026-09-29 15:45-16:00 block). Other iterative-symbolic tasks (ODE/series/system/limit)
are Tier 2: detected and logged, but NOT intervened on, until the matched validation
round (in progress) confirms or rules out a comparable false-positive rate there.

This module is DETECTION + FIRE-DECISION ONLY. It does not implement the intervention
(forced-final-turn) itself -- that's a harness-level integration (DS #155 section 3),
not yet wired into web/server.py or the eval harness. Do NOT enable Tier-1 intervention
in any live/served path until DS #155's sections 5.2-5.4 (simulated forced-final,
intermediate-numeric stress test, live smoke test) have been run for real.
"""
import re

# --- Domain gates (DS #161's narrowed trigger condition) ---

INTEGRAL_RE = re.compile(
    r"(?i)\b(definite integral|indefinite integral|antiderivative|"
    r"integrate\s|integral of|\bintegral\b)"
)
INTEGRAL_CODE_RE = re.compile(r"(?i)(sympy\.)?integrate\(|Integral\(")

ITERATIVE_SYMBOLIC_RE = re.compile(
    r"(?i)\b(ode|differential equation|dsolve|"
    r"system of (linear )?equations|"
    r"infinite series|partial sum|converge|"
    r"\blimit\b|as .* approaches)\b"
)


def is_symbolic_integration(prompt_text, code_seen=""):
    return bool(INTEGRAL_RE.search(prompt_text) or INTEGRAL_CODE_RE.search(code_seen))


def is_iterative_symbolic(prompt_text):
    return bool(ITERATIVE_SYMBOLIC_RE.search(prompt_text))


def domain_tier(prompt_text, code_seen=""):
    """Returns 1 (hard trigger domain), 2 (soft/telemetry fallback domain), or 3 (no trigger)."""
    if is_symbolic_integration(prompt_text, code_seen):
        return 1
    if is_iterative_symbolic(prompt_text):
        return 2
    return 3


# --- Verification-shape detection (DS #155 section 1-2, refined) ---

DENY_KEYS = {
    "id", "uuid", "timestamp", "time", "date", "elapsed", "duration", "latency",
    "cost", "tokens", "page", "limit", "offset", "count", "total_count", "n",
    "status", "code", "version", "round", "index", "idx", "rank", "position",
    "size", "length", "http_status", "status_code",
}
SUCCESS_RE = re.compile(r"(?i)\b(match|matches|matched|agree|agrees|agreed|consistent|"
                         r"equal|same|confirmed|verified|true|pass|passed|ok)\b")
FAIL_RE = re.compile(r"(?i)\b(mismatch|mismatched|disagree|inconsistent|not equal|"
                      r"different|false|fail|failed)\b")
ERROR_RE = re.compile(r"(?i)\b(error|exception|traceback|not (found|defined|available)|"
                       r"not implemented)\b")
# DS #170's layered-classifier design (rules 1+3; rule 4 needs tool_call provenance not
# currently plumbed through, deferred): a round whose output matches these sandbox-probe
# markers contributes NO numbers toward any_clean_progress, since bare integers appearing
# there (e.g. "builtins\n2\n5", "<class 'int'>\n2\nhello") are probe artifacts, not real
# task progress. Floats are exempted (DS: "almost never probe artifacts").
PROBE_MARKERS_RE = re.compile(
    r"(?i)\b(builtins|hello|test|ok|works|__name__|__main__)\b"
    r"|<class |<function |<module |<bound method"
    r"|dir\(|help\(|sys\.version|platform\."
)
VERIFY_INTENT_RE = re.compile(r"(?i)\b(verify|verification|check|cross[- ]?check|double[- ]?check|"
                               r"confirm|validate|sanity[- ]?check|make sure|ensure)\b")
NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def _strip_commas(text):
    return re.sub(r"(?<=\d),(?=\d)", "", text or "")


def extract_final_numeric_lines(tool_output):
    """Prefer numbers on lines that look like a final printed result (avoids picking up
    numbers embedded inside unevaluated symbolic expressions, e.g. '6 - 78*exp(-3)')."""
    if not tool_output:
        return []
    out = []
    for line in tool_output.splitlines():
        # Skip lines that are clearly still-symbolic (contain unresolved operators next to letters)
        if re.search(r"[a-zA-Z]\*|exp\(|sqrt\(|\*\*|/[a-zA-Z]", line) and "=" not in line:
            continue
        # Skip lines that are Python object/memory-address reprs (e.g. sandbox-exploration
        # noise like "<function integrate at 0x78559c23a980>") -- their embedded hex digits
        # are not numeric progress and were previously mistaken for it (DS #167/#169 fix).
        if re.search(r"0x[0-9a-fA-F]+|<function |<class |<built-in|<bound method", line):
            continue
        for m in NUM_RE.finditer(_strip_commas(line)):
            try:
                v = float(m.group())
            except ValueError:
                continue
            if 1900 <= v <= 2100 and "." not in m.group():
                continue
            out.append(v)
    return out


def agree(a, b, rtol=1e-4):
    return abs(a - b) <= max(1e-9, rtol * max(abs(a), abs(b)))


class GuardState:
    def __init__(self, prompt_text):
        self.tier = domain_tier(prompt_text)
        self.verification_requested = VERIFY_INTENT_RE.search(prompt_text) is not None
        self.candidate_value = None
        self.candidate_confirmed = False
        self.verification_completed = False
        self.no_new_info_streak = 0
        self.seen_values = []
        self.fired = False
        self.fire_round = None
        self.fire_reason = None
        self.any_clean_progress = False  # DS #167/#169 guard-logic-gap fix: true once at
        # least one round has both a genuine numeric value AND no error indicator -- gates
        # the weaker unverified_no_new_info_x2 path so it can't fire on pure
        # sandbox-exploration/error stalls where a stray number (e.g. from a type-check
        # print) was mistaken for real progress.

    def observe_tool_round(self, tool_output, round_idx):
        nums = extract_final_numeric_lines(tool_output)
        has_success = bool(SUCCESS_RE.search(tool_output or ""))
        has_fail = bool(FAIL_RE.search(tool_output or ""))
        has_error = bool(ERROR_RE.search(tool_output or ""))
        is_probe_round = bool(PROBE_MARKERS_RE.search(tool_output or ""))
        if is_probe_round:
            # DS #170 rule 3: floats survive even in a probe-flagged round (a genuine
            # computed float is unlikely to appear alongside probe markers by accident);
            # bare integers in a probe round are suppressed as probe artifacts.
            real_nums = [v for v in nums if not float(v).is_integer()]
        else:
            real_nums = nums
        if real_nums and not has_error:
            self.any_clean_progress = True

        new_value = False
        for v in nums:
            if not any(agree(v, sv) for sv in self.seen_values):
                new_value = True
                self.seen_values.append(v)

        if not new_value:
            self.no_new_info_streak += 1
        else:
            self.no_new_info_streak = 0

        if nums:
            cand = nums[-1]
            if self.candidate_value is not None and agree(cand, self.candidate_value):
                self.candidate_confirmed = True
            self.candidate_value = cand

        if self.verification_requested and has_success and not has_fail:
            self.verification_completed = True
        elif has_fail:
            self.candidate_confirmed = False
            self.verification_completed = False

        self._maybe_fire(round_idx)

    def _maybe_fire(self, round_idx):
        if self.fired or self.tier != 1 or self.candidate_value is None or round_idx < 2:
            return
        if self.verification_completed or self.candidate_confirmed:
            would_fire = self.no_new_info_streak >= 1
        else:
            # Guard-logic-gap fix (DS #167/#169): the weaker unverified path requires at
            # least one round of genuine (error-free) numeric progress before it's
            # allowed to fire -- otherwise a stall BEFORE any real computation (pure
            # sandbox-exploration errors) gets misread as a stall AFTER convergence.
            would_fire = self.no_new_info_streak >= 2 and self.any_clean_progress
        if would_fire:
            self.fired = True
            self.fire_round = round_idx
            self.fire_reason = (
                "verified_or_confirmed_no_new_info" if (self.verification_completed or self.candidate_confirmed)
                else "unverified_no_new_info_x2"
            )

    def should_intervene(self):
        """Tier 1 only. Tier 2 should log self.fired but never act on it (telemetry only,
        per DS #161's explicit instruction) until the matched validation round is analyzed."""
        return self.tier == 1 and self.fired


# --- Nudge-only intervention (v0.5, per DS #162: no candidate value supplied) ---
# Same fire-decision (domain gate + no-new-info-streak + verification-completion), but the
# guard NEVER hands the model a specific number. It only tells the model to stop calling
# tools and produce a final answer from what it already knows. This sidesteps the
# candidate-extraction-accuracy problem entirely (measured at only ~64-76% depending on
# stratum, well below DS's <1-2% wrong-candidate bar for a value-supplying guard).
#
# NOT YET VALIDATED FOR LIVE USE. Per DS #162's own admission, offline transcript replay
# cannot establish the counterfactual correctness of a nudge intervention (we don't know
# what the model would actually generate when nudged mid-conversation) -- that requires a
# live online A/B: serve the model, replay the conversation up to the fire point, inject
# the nudge, generate the forced-final turn for real, and score it against ground truth.
# That experiment is queued but not yet run (see CLAUDE.md 2026-09-29 for status).

NUDGE_MESSAGE = (
    "[GUARD] You have enough information from your tool calls to answer. "
    "Do not call any more tools. Provide your final answer now, using the results you "
    "already have. If you are genuinely uncertain, say so explicitly rather than "
    "continuing to search or verify."
)


def nudge_intervention_messages():
    """Returns the message(s) to append to the conversation when a Tier-1 fire occurs
    under the nudge-only (v0.5) design. Caller is responsible for then calling the model
    with tools disabled (tool_choice='none') for exactly one forced-final turn, per DS
    #155 section 3.2's pattern -- reusing the harness's existing forced-final-after-budget
    mechanism, just triggered earlier and without a supplied number.

    Role is 'user', not 'system': the served qwen3 chat template rejects any system-role
    message not at position 0 (jinja2.exceptions.TemplateError: 'System message must be
    at the beginning'), the same constraint hit earlier with the opencode integration in
    web/server.py. A mid-conversation system message would crash every fire; user-role
    (framed as an operator/tool-harness message, not the human) renders fine and is the
    correct mechanism for a mid-conversation directive under this template."""
    return [{"role": "user", "content": NUDGE_MESSAGE}]
