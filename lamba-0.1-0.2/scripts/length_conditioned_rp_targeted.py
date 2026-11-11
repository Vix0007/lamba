"""Targeted length-conditioned repetition-penalty (DS #132, Box A).

Same trigger logic as length_conditioned_rp.py (global variant, found catastrophic when it
REPLACED a constant rp=1.1 -- see ds_queue/132_length_cond_rp_catastrophic_failure*). This
variant is designed to always run ADDITIVE to a constant repetition_penalty (pass e.g.
--repetition-penalty 1.1 --length-cond-rp-targeted together), and differs in SCOPE: instead of
penalizing every token seen anywhere in the generation so far (which, once triggered on a long
sequence, covers nearly the whole usable vocabulary), it penalizes ONLY the token IDs that make
up the specific repeated n-gram(s) that satisfied the trigger threshold in the trailing window.

Same ramp/trigger constants as the global variant (L0, L1, R_MAX, S_MIN, S_MAX, S4_ACTIVATE,
WINDOW_TOKENS) -- only the penalty SCOPE differs.
"""
import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
from vllm.sampling_params import SamplingParams

L0 = 1152.0
L1 = 2176.0
R_MAX = 1.12
S_MIN = 3.0
S_MAX = 6.0
S4_ACTIVATE = S_MIN + 2.0
WINDOW_TOKENS = 1024
_PRINT_EVERY = 50


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def _ngram_counts(window: list[int], n: int) -> dict:
    if len(window) < n:
        return {}
    counts: dict[tuple, int] = {}
    for i in range(len(window) - n + 1):
        g = tuple(window[i:i + n])
        counts[g] = counts.get(g, 0) + 1
    return counts


def _repeated_grams_at_or_above(counts: dict, thresh: float) -> list[tuple]:
    return [g for g, c in counts.items() if c >= thresh]


def _r_and_targets(output_ids: list[int]) -> tuple[float, set]:
    """Returns (r_eff, penalized_token_ids). penalized_token_ids is the union of token ids
    appearing in whichever repeated n-gram(s) triggered activation -- NOT the whole seen set."""
    L = len(output_ids)
    w_len = _clip01((L - L0) / (L1 - L0))
    if w_len <= 0.0:
        return 1.0, set()
    window = output_ids[-WINDOW_TOKENS:]
    c4 = _ngram_counts(window, 4)
    c8 = _ngram_counts(window, 8)
    s4 = float(max(c4.values())) if c4 else 0.0
    s8 = float(max(c8.values())) if c8 else 0.0

    active8 = s8 >= S_MIN
    active4 = s4 >= S4_ACTIVATE
    if not (active8 or active4):
        return 1.0, set()

    targets: set = set()
    if active8:
        for g in _repeated_grams_at_or_above(c8, S_MIN):
            targets.update(g)
    if active4:
        for g in _repeated_grams_at_or_above(c4, S4_ACTIVATE):
            targets.update(g)

    w_rep = max(_clip01((s8 - S_MIN) / (S_MAX - S_MIN)),
                _clip01((s4 - S4_ACTIVATE) / (S_MAX + 2 - S4_ACTIVATE)))
    r = 1.0 + (R_MAX - 1.0) * w_len * w_rep
    if L % _PRINT_EVERY == 0:
        print(f"[length_cond_rp_targeted] L={L} s4={s4:.0f} s8={s8:.0f} r={r:.4f} "
              f"n_targets={len(targets)}", flush=True)
    return r, targets


class LengthConditionedTargetedPenalty(AdapterLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams):
        def _apply(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            r, targets = _r_and_targets(output_ids)
            if r <= 1.0 or not targets:
                return logits
            idx = torch.tensor(list(targets), device=logits.device, dtype=torch.long)
            sel = logits[idx]
            pos = sel > 0
            sel[pos] = sel[pos] / r
            sel[~pos] = sel[~pos] * r
            logits[idx] = sel
            return logits
        return _apply
