"""Length-conditioned repetition-penalty logits processor (DS consult #130/#131).

Rationale: blanket repetition_penalty > 1.1 already known to catastrophically fail on this
architecture (project history: rp=1.15 regressed, rp=1.2 broke generation entirely). This
instead ramps a penalty ONLY once (a) generation is already long AND (b) a repeated n-gram is
currently detected in the trailing window -- so short/coherent generations (the vast majority
of mcp/malay outputs) never see any penalty at all, and it only engages for the specific
failure mode (long, science-domain, actually-repeating) that the length-matched diagnostic
(#129/#130/#131, 7/7 lambdas positive sci-mcp DiD) identified.

r_eff = 1 + (r_max - 1) * w_len * w_rep
  w_len = clip((L - L0) / (L1 - L0), 0, 1)      L0=1152, L1=2176 (L0+1024)
  w_rep = clip((s - s_min) / (s_max - s_min), 0, 1)   s_min=3, s_max=6
  s = max repeat count of any 4-gram or 8-gram in the trailing WINDOW_TOKENS-token window

Applied with standard repetition_penalty semantics (divide positive logits / multiply negative
logits) to every token id present anywhere in output_ids so far (presence-based, like vLLM's
built-in rp), using r_eff instead of a fixed constant.
"""
import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
from vllm.sampling_params import SamplingParams

# DS #131 wants activation-rate/mean-r telemetry. The engine typically runs in a separate
# (spawned) worker process from the caller, so a module-level counter here won't be visible to
# loop_test_g.py's process after llm.generate() returns -- cross-process aggregation would need
# real IPC plumbing, not worth building for a diagnostic. Instead, print directly from inside
# the hot path, rate-limited so it doesn't flood the log: this process's own stdout already
# appears inline in the run's log (same as vLLM's own "(EngineCore pid=...)" lines), so an
# occasional print here is visible for free.
_PRINT_EVERY = 50  # tokens, while active

L0 = 1152.0
L1 = 2176.0
R_MAX = 1.12
S_MIN = 3.0
S_MAX = 6.0
S4_ACTIVATE = S_MIN + 2.0  # DS #131: since max(s4,s8) degenerates to s4 (an 8-gram repeat
                            # implies >= that count of 4-grams), use an explicit OR-rule instead:
                            # activate if s8>=S_MIN or s4>=S_MIN+2 (s4 alone needs a higher bar,
                            # since 4-grams are noisier/more likely to repeat innocently)
WINDOW_TOKENS = 1024  # DS #131's "cleaner" fixed token window for token n-grams


def _ngram_counts(window: list[int], n: int) -> dict:
    if len(window) < n:
        return {}
    counts: dict[tuple, int] = {}
    for i in range(len(window) - n + 1):
        g = tuple(window[i:i + n])
        counts[g] = counts.get(g, 0) + 1
    return counts


def _s4_s8(window: list[int]) -> tuple[float, float]:
    """Max occurrence count of any 4-gram and any 8-gram within `window`, computed and
    logged SEPARATELY (DS #131: max(s4,s8) mathematically degenerates to just s4, since any
    repeated 8-gram implies an equal-or-higher 4-gram count -- don't silently let that happen)."""
    c4 = _ngram_counts(window, 4)
    c8 = _ngram_counts(window, 8)
    s4 = float(max(c4.values())) if c4 else 0.0
    s8 = float(max(c8.values())) if c8 else 0.0
    return s4, s8


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def _r_eff(output_ids: list[int]) -> float:
    L = len(output_ids)
    w_len = _clip01((L - L0) / (L1 - L0))
    if w_len <= 0.0:
        return 1.0  # short-circuit: no penalty possible yet, skip repeat scan entirely
    window = output_ids[-WINDOW_TOKENS:]
    s4, s8 = _s4_s8(window)
    # DS #131 OR-rule: activate on either an 8-gram repeat at S_MIN, or a 4-gram repeat at a
    # higher bar (S4_ACTIVATE) -- 8-grams are the primary/more-specific signal, 4-grams alone
    # need to be more extreme before they count (avoids common-phrase false positives).
    active = (s8 >= S_MIN) or (s4 >= S4_ACTIVATE)
    if not active:
        return 1.0
    # w_rep uses whichever signal is driving activation, scaled the same way for both.
    w_rep = max(_clip01((s8 - S_MIN) / (S_MAX - S_MIN)),
                _clip01((s4 - S4_ACTIVATE) / (S_MAX + 2 - S4_ACTIVATE)))
    r = 1.0 + (R_MAX - 1.0) * w_len * w_rep
    if L % _PRINT_EVERY == 0:
        print(f"[length_cond_rp] L={L} s4={s4:.0f} s8={s8:.0f} r={r:.4f}", flush=True)
    return r


class LengthConditionedRepetitionPenalty(AdapterLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return False  # can change argmax (that's the point)

    def new_req_logits_processor(self, params: SamplingParams):
        # Applied to every request; short-circuits to a no-op (r_eff=1.0) until length/repeat
        # conditions are met, so it is cheap and inert for the vast majority of requests
        # (mcp/malay, and science outputs that never actually loop).
        def _apply(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            r = _r_eff(output_ids)
            if r <= 1.0:
                return logits
            seen = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
            if output_ids:
                idx = torch.tensor(output_ids, device=logits.device, dtype=torch.long)
                seen[idx] = True
            pos = seen & (logits > 0)
            neg = seen & (logits <= 0)
            logits[pos] = logits[pos] / r
            logits[neg] = logits[neg] * r
            return logits
        return _apply
