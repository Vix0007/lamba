# Derived from https://github.com/Liquid4All/antidoom (Copyright Liquid AI, Apache-2.0).
# Modified by Vixero Labs: additional losses, dataset kinds and training options. See NOTICE.
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from antidoom.trimmed import TRIMMED_PROMPT_PREFIX, row_has_trimmed_source_prompt

logger = logging.getLogger(__name__)


DEFAULT_STOP_WORDS = {
    "the",
    "a",
    "an",
    "in",
    "on",
    "at",
    "by",
    "for",
    "to",
    "of",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "where",
    "how",
    "why",
    "what",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
    "can",
    "could",
    "may",
    "might",
    "must",
}

CHOSEN_REGULARISATION_MIN_REF_COUNT = 50
CHOSEN_REGULARISATION_REF_PERCENTILE = 95

SOURCE_BALANCE_MODES = ("off", "sqrt", "equal")


def row_source_label(row: dict[str, Any]) -> str:
    return (
        row.get("source_dataset")
        or row.get("source_task")
        or row.get("source_eval_key")
        or "__unknown__"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def has_trimmed_source_prompt(row: dict[str, Any]) -> bool:
    return row_has_trimmed_source_prompt(row)


def normalise_chosen_key(text: str) -> str:
    stripped = text.strip().lower()
    word = re.sub(r"^[^\w]+|[^\w]+$", "", stripped, flags=re.UNICODE)
    return word or stripped or text


def _regularised_ratios(counts: Counter[str], strength: float) -> dict[str, float]:
    if not counts:
        return {}
    median = float(np.median(list(counts.values())))
    weights = {
        tok: 1.0 if count <= median or strength <= 0 else (median / count) ** strength
        for tok, count in counts.items()
    }
    total = sum(weights[tok] * count for tok, count in counts.items())
    return {tok: (weights[tok] * count) / total for tok, count in counts.items()}


def sample_rows(
    rows: list[dict[str, Any]],
    *,
    max_train_examples: int | None,
    rejected_strength: float,
    min_chosen_tokens: int,
    seed: int,
    source_balance_mode: str = "off",
) -> list[dict[str, Any]]:
    if source_balance_mode not in SOURCE_BALANCE_MODES:
        raise ValueError(
            f"source_balance_mode must be one of {SOURCE_BALANCE_MODES}; "
            f"got {source_balance_mode!r}"
        )
    rows = [r for r in rows if len(r.get("multi_chosen_decoded") or []) >= min_chosen_tokens]
    if not rows:
        return []
    target_n = min(max_train_examples or len(rows), len(rows))
    selected = _rejected_normalisation(rows, target_n, rejected_strength, seed)
    if source_balance_mode != "off":
        selected = _source_balance_downsample(selected, source_balance_mode, seed)
    return selected


def _rejected_normalisation(
    rows: list[dict[str, Any]],
    target_n: int,
    rejected_strength: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Greedy rejected-token sampling with a pool-share source-aware tiebreak.

    The rejected-token target counts are primary; source preference only
    decides which row is taken from a given token bucket when multiple are
    available, biased toward sources currently below their share of the pool.
    """
    ratios = _regularised_ratios(
        Counter(r["rejected_decoded"] for r in rows), rejected_strength
    )
    targets = {tok: int(round(ratio * target_n)) for tok, ratio in ratios.items()}

    pool = Counter(row_source_label(r) for r in rows)
    pool_total = sum(pool.values()) or 1
    share = {s: c / pool_total for s, c in pool.items()}

    buckets: defaultdict[str, defaultdict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        buckets[row["rejected_decoded"]][row_source_label(row)].append(row)

    rng = np.random.default_rng(seed)
    for tok in buckets:
        for source in buckets[tok]:
            rng.shuffle(buckets[tok][source])

    selected: list[dict[str, Any]] = []
    seen_tok: defaultdict[str, int] = defaultdict(int)
    selected_src: Counter[str] = Counter()

    def best_choice(respect_cap: bool) -> tuple[str, str] | None:
        denom = max(len(selected), 1)
        best: tuple[str, str] | None = None
        best_key: tuple[float, float] | None = None
        for tok, src_buckets in buckets.items():
            if respect_cap and seen_tok[tok] >= targets.get(tok, 0):
                continue
            tok_gap = ratios.get(tok, 0.0) - seen_tok[tok] / denom
            for source, available in src_buckets.items():
                if not available:
                    continue
                src_def = share.get(source, 0.0) - selected_src[source] / denom
                key = (tok_gap, src_def)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (tok, source)
        return best

    while len(selected) < target_n:
        choice = best_choice(respect_cap=True)
        if choice is None:
            break
        tok, source = choice
        selected.append(buckets[tok][source].pop())
        seen_tok[tok] += 1
        selected_src[source] += 1

    while len(selected) < target_n:
        choice = best_choice(respect_cap=False)
        if choice is None:
            break
        tok, source = choice
        selected.append(buckets[tok][source].pop())
        seen_tok[tok] += 1
        selected_src[source] += 1

    rng.shuffle(selected)
    return selected


def _source_balance_downsample(
    rows: list[dict[str, Any]],
    mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Drop rows from sources that exceed their `mode`-implied share of `rows`.

    Only downsamples; never duplicates. Output count drops below input when
    one or more sources sit above their target share.
    """
    by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row_source_label(row)].append(row)
    if not by_source:
        return list(rows)

    if mode == "equal":
        share = {s: 1.0 / len(by_source) for s in by_source}
    elif mode == "sqrt":
        roots = {s: len(rs) ** 0.5 for s, rs in by_source.items()}
        total = sum(roots.values()) or 1.0
        share = {s: v / total for s, v in roots.items()}
    else:
        raise ValueError(f"unknown source_balance_mode: {mode!r}")

    n = len(rows)
    rng = np.random.default_rng(seed)
    kept: list[dict[str, Any]] = []
    for source, source_rows in by_source.items():
        cap = int(round(share[source] * n))
        if len(source_rows) <= cap:
            kept.extend(source_rows)
        else:
            rng.shuffle(source_rows)
            kept.extend(source_rows[:cap])
    rng.shuffle(kept)
    return kept


def _count_percentile(counts: Counter[str], percentile: int) -> int:
    if not counts:
        return 1
    values = sorted(counts.values())
    index = int((len(values) - 1) * percentile / 100)
    return max(1, values[index])


def _regularised_chosen_targets(counts: Counter[str], strength: float) -> dict[str, int]:
    ref_count = max(
        CHOSEN_REGULARISATION_MIN_REF_COUNT,
        _count_percentile(counts, CHOSEN_REGULARISATION_REF_PERCENTILE),
    )
    targets: dict[str, int] = {}
    for key, count in counts.items():
        if strength > 0 and count > ref_count:
            target = count * (ref_count / count) ** strength
            targets[key] = max(1, int(round(target)))
        else:
            targets[key] = count
    return targets


def _strip_chosen_keys(example: dict[str, Any]) -> dict[str, list[int] | int]:
    return {
        "prompt_ids": example["prompt_ids"],
        "chosen_ids": example["chosen_ids"],
        "rejected_token_id": example["rejected_token_id"],
    }


def _regularise_chosen_examples(
    examples: list[dict[str, Any]],
    strength: float,
    filter_counts: Counter[str],
) -> list[dict[str, list[int] | int]]:
    deduped: list[dict[str, Any]] = []
    chosen_key_counts: Counter[str] = Counter()
    for example in examples:
        row_seen_keys: set[str] = set()
        chosen_ids = []
        chosen_keys = []
        removed = 0
        for token_id, key in zip(example["chosen_ids"], example["chosen_keys"], strict=True):
            if key in row_seen_keys:
                removed += 1
                continue
            row_seen_keys.add(key)
            chosen_ids.append(token_id)
            chosen_keys.append(key)
        if removed:
            filter_counts["chosen_regularised"] += removed
        deduped.append(
            {
                "prompt_ids": example["prompt_ids"],
                "chosen_ids": chosen_ids,
                "chosen_keys": chosen_keys,
                "rejected_token_id": example["rejected_token_id"],
            }
        )
        chosen_key_counts.update(chosen_keys)

    targets = _regularised_chosen_targets(chosen_key_counts, strength)
    current_counts = Counter(chosen_key_counts)
    examples_by_key: defaultdict[str, set[int]] = defaultdict(set)
    for index, example in enumerate(deduped):
        for key in example["chosen_keys"]:
            examples_by_key[key].add(index)

    for key in sorted(targets, key=lambda item: current_counts[item] - targets[item], reverse=True):
        while current_counts[key] > targets[key]:
            candidates = [
                index
                for index in examples_by_key[key]
                if key in deduped[index]["chosen_keys"] and len(deduped[index]["chosen_keys"]) > 1
            ]
            if not candidates:
                break
            index = max(candidates, key=lambda item: (len(deduped[item]["chosen_keys"]), -item))
            remove_at = deduped[index]["chosen_keys"].index(key)
            del deduped[index]["chosen_keys"][remove_at]
            del deduped[index]["chosen_ids"][remove_at]
            current_counts[key] -= 1
            filter_counts["chosen_regularised"] += 1

    for key in sorted(targets, key=lambda item: current_counts[item] - targets[item], reverse=True):
        while current_counts[key] > targets[key]:
            candidates = [
                index
                for index in examples_by_key[key]
                if deduped[index]["chosen_keys"] == [key]
            ]
            if not candidates:
                break
            index = min(candidates)
            deduped[index]["chosen_keys"] = []
            deduped[index]["chosen_ids"] = []
            current_counts[key] -= 1
            filter_counts["chosen_regularised"] += 1
            filter_counts["no_chosen_after_chosen_regularisation"] += 1

    return [_strip_chosen_keys(example) for example in deduped if example["chosen_ids"]]


class FTPODataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        *,
        max_seq_length: int,
        stop_words: set[str] | None = None,
        filter_rejected_stop_words: bool = True,
        chosen_regularisation_strength: float = 0.0,
    ) -> None:
        self.examples: list[dict[str, list[int] | int]] = []
        candidate_examples: list[dict[str, Any]] = []
        self.max_seq_length = max_seq_length
        self.filter_counts: Counter[str] = Counter()
        stop_words = stop_words or DEFAULT_STOP_WORDS
        for row in rows:
            self.filter_counts["input_rows"] += 1
            if has_trimmed_source_prompt(row):
                self.filter_counts["trimmed_source_prompt"] += 1
                continue
            prompt_ids = tokenizer(
                row["context_with_chat_template"],
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
            ).input_ids

            rejected_ids = tokenizer(row["rejected_decoded"], add_special_tokens=False).input_ids
            if len(rejected_ids) != 1:
                self.filter_counts["rejected_not_single_token"] += 1
                continue
            if len(prompt_ids) + len(rejected_ids) > max_seq_length:
                self.filter_counts["overlength_context"] += 1
                continue

            chosen_ids = []
            chosen_surfaces = []
            raw_chosen_count = 0
            for surf in row.get("multi_chosen_decoded") or []:
                raw_chosen_count += 1
                ids = tokenizer(surf, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    chosen_ids.append(ids[0])
                    chosen_surfaces.append(surf)
                else:
                    self.filter_counts["chosen_not_single_token"] += 1
            if raw_chosen_count == 0:
                self.filter_counts["no_chosen_candidates"] += 1
                continue
            if not chosen_ids:
                self.filter_counts["no_single_token_chosen"] += 1
                continue
            rejected = rejected_ids[0]
            if filter_rejected_stop_words and row["rejected_decoded"].strip().lower() in stop_words:
                self.filter_counts["rejected_stop_word"] += 1
                continue
            before_remove_rejected = len(chosen_ids)
            chosen_pairs = [
                (tid, surf) for tid, surf in zip(chosen_ids, chosen_surfaces, strict=True) if tid != rejected
            ]
            chosen_ids = [tid for tid, _surf in chosen_pairs]
            chosen_surfaces = [surf for _tid, surf in chosen_pairs]
            if len(chosen_ids) < before_remove_rejected:
                self.filter_counts["chosen_same_as_rejected"] += before_remove_rejected - len(chosen_ids)
            if not chosen_ids:
                self.filter_counts["no_chosen_after_rejected_removed"] += 1
                continue
            chosen_keys = [normalise_chosen_key(surface) for surface in chosen_surfaces]
            candidate_examples.append(
                {
                    "prompt_ids": prompt_ids,
                    "chosen_ids": chosen_ids,
                    "chosen_keys": chosen_keys,
                    "rejected_token_id": rejected,
                }
            )
        if chosen_regularisation_strength > 0:
            self.examples = _regularise_chosen_examples(
                candidate_examples,
                chosen_regularisation_strength,
                self.filter_counts,
            )
        else:
            self.examples = [_strip_chosen_keys(example) for example in candidate_examples]
        self.filter_counts["kept_examples"] = len(self.examples)
        if not self.examples:
            raise ValueError("no FTPO samples survived tokenisation and filters")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int] | int]:
        return self.examples[index]


class FTPOCollator:
    def __init__(self, *, pad_token_id: int, max_seq_length: int) -> None:
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(features)
        prompt_ids = torch.full(
            (batch_size, self.max_seq_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(prompt_ids, dtype=torch.bool)
        for i, feat in enumerate(features):
            ids = torch.tensor(feat["prompt_ids"], dtype=torch.long)
            if ids.numel() + 1 > self.max_seq_length:
                raise ValueError(
                    "FTPO sample exceeds max_seq_length; overlength samples must be "
                    "discarded before collation, not truncated"
                )
            prompt_ids[i, -ids.numel() :] = ids
            attention_mask[i, -ids.numel() :] = True

        max_chosen = max(len(feat["chosen_ids"]) for feat in features)
        chosen_ids = torch.full((batch_size, max_chosen), self.pad_token_id, dtype=torch.long)
        chosen_mask = torch.zeros_like(chosen_ids, dtype=torch.bool)
        for i, feat in enumerate(features):
            ids = torch.tensor(feat["chosen_ids"], dtype=torch.long)
            chosen_ids[i, : ids.numel()] = ids
            chosen_mask[i, : ids.numel()] = True

        return {
            "prompt_ids": prompt_ids,
            "attention_mask": attention_mask,
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_token_id": torch.tensor(
                [feat["rejected_token_id"] for feat in features], dtype=torch.long
            ),
        }


class ConcatFTPODataset(Dataset):
    """Concatenation of FTPODatasets that keeps the attributes FTPOTrainer/collator read (max_seq_length, filter_counts)."""

    def __init__(self, *datasets: "FTPODataset") -> None:
        self.datasets = datasets
        self.examples = [ex for d in datasets for ex in d.examples]
        self.max_seq_length = max(d.max_seq_length for d in datasets)
        self.filter_counts: Counter[str] = Counter()
        for d in datasets:
            self.filter_counts.update(d.filter_counts)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int] | int]:
        return self.examples[index]


class LetterGroupDataset(Dataset):
    """One example per MCQ question: K cyclic-shift renders of the same question.

    Each stored example is::

        {
            "kind": "letter_group",
            "qid": str,
            "n": int,                     # number of options == K renders == len(letter_ids)
            "gold_orig": int,             # index of the gold option in the ORIGINAL ordering
            "prompt_ids": list[list[int]],  # K token id lists (contexts end with '\nAnswer:')
            "orders": list[list[int]],      # K lists: order[i] = original index at presented pos i
            "gold_pos": list[int],          # K ints: position where gold content appears
            "chosen_ids": list[int],        # K ints: token id of the chosen letter in each render
            "rejected_ids": list[int],      # K ints: token id of the rejected letter in each render
            "letter_ids": list[int],        # n ints: token id of each letter surface (" A", ...)
        }
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        *,
        max_seq_length: int,
    ) -> None:
        self.examples: list[dict[str, Any]] = []
        self.max_seq_length = max_seq_length
        self.filter_counts: Counter[str] = Counter()

        def _one_token(text: str) -> int | None:
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if len(ids) != 1:
                return None
            return ids[0]

        for row in rows:
            self.filter_counts["input_rows"] += 1

            if row.get("kind") != "letter_group":
                self.filter_counts["not_letter_group"] += 1
                continue

            n = row.get("n")
            renders = row.get("renders") or []
            letter_tokens = row.get("letter_tokens") or []
            gold_orig = row.get("gold_orig")
            if not isinstance(n, int) or n <= 0:
                self.filter_counts["bad_n"] += 1
                continue
            if len(renders) != n:
                self.filter_counts["render_count_mismatch"] += 1
                continue
            if len(letter_tokens) != n:
                self.filter_counts["letter_count_mismatch"] += 1
                continue
            if not isinstance(gold_orig, int) or not (0 <= gold_orig < n):
                self.filter_counts["bad_gold_orig"] += 1
                continue

            letter_ids: list[int] = []
            letter_ok = True
            for surface in letter_tokens:
                tid = _one_token(surface)
                if tid is None:
                    letter_ok = False
                    self.filter_counts["letter_not_single_token"] += 1
                    break
                letter_ids.append(tid)
            if not letter_ok:
                continue

            prompt_ids: list[list[int]] = []
            orders: list[list[int]] = []
            gold_pos: list[int] = []
            chosen_ids: list[int] = []
            rejected_ids: list[int] = []
            bad = False

            for render in renders:
                ctx_ids = tokenizer(
                    render["context"],
                    add_special_tokens=False,
                    truncation=False,
                    return_attention_mask=False,
                ).input_ids
                # +1 for the appended single-token answer letter, matching FTPODataset.
                if len(ctx_ids) + 1 > max_seq_length:
                    self.filter_counts["overlength_context"] += 1
                    bad = True
                    break

                order = list(render["order"])
                if sorted(order) != list(range(n)):
                    self.filter_counts["order_not_permutation"] += 1
                    bad = True
                    break
                if len(order) != n:
                    self.filter_counts["order_length_mismatch"] += 1
                    bad = True
                    break

                gpos = render["gold_pos"]
                if not isinstance(gpos, int) or not (0 <= gpos < n):
                    self.filter_counts["bad_gold_pos"] += 1
                    bad = True
                    break
                # order[gpos] must be the original gold option.
                if order[gpos] != gold_orig:
                    self.filter_counts["gold_pos_order_inconsistent"] += 1
                    bad = True
                    break

                ch = _one_token(render["chosen_letter"])
                if ch is None:
                    self.filter_counts["chosen_not_single_token"] += 1
                    bad = True
                    break
                rj = _one_token(render["rejected_letter"])
                if rj is None:
                    self.filter_counts["rejected_not_single_token"] += 1
                    bad = True
                    break
                if ch == rj:
                    self.filter_counts["chosen_same_as_rejected"] += 1
                    bad = True
                    break

                prompt_ids.append(ctx_ids)
                orders.append(order)
                gold_pos.append(gpos)
                chosen_ids.append(ch)
                rejected_ids.append(rj)

            if bad:
                continue

            self.examples.append(
                {
                    "kind": "letter_group",
                    "qid": row["qid"],
                    "n": n,
                    "gold_orig": gold_orig,
                    "prompt_ids": prompt_ids,
                    "orders": orders,
                    "gold_pos": gold_pos,
                    "chosen_ids": chosen_ids,
                    "rejected_ids": rejected_ids,
                    "letter_ids": letter_ids,
                    # Ruminconfab DPO retrain KL-replay (DS #179): optional per-row scale,
                    # defaults to 1.0 (no-op) for every prior/other use of letter_group rows
                    # that never set this field -- only rows explicitly built as part of a
                    # KL-replay set (kl_replay_set_v1.jsonl) carry a non-1.0 value.
                    "kl_replay_scale": float(row.get("kl_replay_scale", 1.0)),
                }
            )

        self.filter_counts["kept_examples"] = len(self.examples)
        if not self.examples:
            raise ValueError("no LetterGroup samples survived tokenisation and filters")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class LoopSpanDataset(Dataset):
    """Real-loop unlikelihood rows (kind='loop_span'): context = chat prompt + rollout text through the FIRST occurrence of a repeated
    unit, target = the start of the SECOND occurrence. One example = one forward; the loss (reference_ftpo_trainer._loop_span_loss) pushes
    each target token below the best alternative by a margin at K positions, with an MSE tether to the frozen base on the rest of the vocab.
    Stored example: {'kind','prompt_ids' (context+target tokens, truncated after K target tokens),'j0' (index of first target token),'K'}."""

    def __init__(self, rows, tokenizer, *, max_seq_length: int, span_tokens: int = 32, min_span_tokens: int = 8) -> None:
        self.examples: list[dict[str, Any]] = []
        self.max_seq_length = max_seq_length
        self.filter_counts: Counter[str] = Counter()
        for row in rows:
            self.filter_counts["input_rows"] += 1
            if row.get("kind") != "loop_span":
                self.filter_counts["not_loop_span"] += 1
                continue
            ctx, tgt = row["context"], row["target_text"]
            enc = tokenizer(ctx + tgt, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            boundary = len(ctx)
            j0 = next((j for j, (a, b) in enumerate(offs) if a >= boundary), None)
            if j0 is None:
                self.filter_counts["no_target_tokens"] += 1
                continue
            K = min(span_tokens, len(ids) - j0)
            if K < min_span_tokens:
                self.filter_counts["span_too_short"] += 1
                continue
            ids = ids[: j0 + K]
            if len(ids) + 1 > max_seq_length:
                self.filter_counts["overlength"] += 1
                continue
            self.examples.append({"kind": "loop_span", "prompt_ids": ids, "j0": j0, "K": K, "pilot_id": row.get("pilot_id")})
        self.filter_counts["kept_examples"] = len(self.examples)
        if not self.examples:
            raise ValueError("no LoopSpan samples survived tokenisation and filters")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class SFTDataset(Dataset):
    """Beat-ThinkingCap track (DS consult #110): plain SFT/cross-entropy rows (kind='sft_ce').
    Full, NATURAL, untruncated traces -- explicitly NOT the truncate-and-suppress mechanism used
    by loop_span/ftpo_soft (that mechanism, applied with a teacher-length onset proxy, is what
    caused the effreason-round-2 regression). Here the entire correct-and-naturally-concise trace
    IS the target; loss = cross-entropy on response tokens only (prompt tokens masked, not just
    excluded from the loss weight -- excluded entirely from position 0..prompt_len-1's labels).
    Stored example: {'kind','prompt_ids' (full context+response token ids),'prompt_len' (index of
    the first response token; predict positions prompt_len-1 .. end-2, targets prompt_len .. end-1)}."""

    def __init__(self, rows, tokenizer, *, max_seq_length: int) -> None:
        self.examples: list[dict[str, Any]] = []
        self.max_seq_length = max_seq_length
        self.filter_counts: Counter[str] = Counter()
        for row in rows:
            self.filter_counts["input_rows"] += 1
            if row.get("kind") != "sft_ce":
                self.filter_counts["not_sft_ce"] += 1
                continue
            ctx, resp = row["context"], row["response_text"]
            enc = tokenizer(ctx + resp, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            boundary = len(ctx)
            prompt_len = next((j for j, (a, b) in enumerate(offs) if a >= boundary), None)
            if prompt_len is None or prompt_len >= len(ids) - 1:
                self.filter_counts["no_response_tokens"] += 1
                continue
            if len(ids) + 1 > max_seq_length:
                self.filter_counts["overlength"] += 1
                continue
            # Defensive check for BPE boundary-merge bugs (a token straddling the prompt/response
            # seam can get misclassified if the seam falls mid-token instead of on a special-token
            # boundary -- verified this DOES happen with plain-text seams, though not with our
            # actual chat-template seam which ends in special tokens). Decode the response side
            # and require it to actually start with the intended response's own first few real
            # (non-whitespace) characters -- catches the failure mode without needing per-row
            # manual inspection of every training example.
            decoded_resp = tokenizer.decode(ids[prompt_len:])
            resp_check = resp.strip()[:12]
            if resp_check and not decoded_resp.strip().startswith(resp_check[:6]):
                self.filter_counts["boundary_mismatch"] += 1
                continue
            self.examples.append({"kind": "sft_ce", "prompt_ids": ids, "prompt_len": prompt_len,
                                  "pilot_id": row.get("pilot_id"), "domain": row.get("domain")})
        self.filter_counts["kept_examples"] = len(self.examples)
        if not self.examples:
            raise ValueError("no SFT samples survived tokenisation and filters")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class SoftFTPODataset(Dataset):
    """Round-2 soft-target rows (kind='ftpo_soft'): context ends right before the first token of a repeat; rejected = that token (single id).
    Loss (reference_ftpo_trainer._soft_ftpo_loss): KL(q||p_adapter) with q = 0.5 p_ref + 0.5 softmax((log p_ref - gamma*1[rej])/tau), weighted by row['weight']."""

    def __init__(self, rows, tokenizer, *, max_seq_length: int) -> None:
        self.examples: list[dict[str, Any]] = []
        self.max_seq_length = max_seq_length
        self.filter_counts: Counter[str] = Counter()
        for row in rows:
            self.filter_counts["input_rows"] += 1
            if row.get("kind") != "ftpo_soft":
                self.filter_counts["not_soft"] += 1
                continue
            ids = tokenizer(row["context_with_chat_template"], add_special_tokens=False).input_ids
            rid = tokenizer(row["rejected_decoded"], add_special_tokens=False).input_ids
            if len(rid) != 1:
                self.filter_counts["rejected_not_single_token"] += 1
                continue
            if len(ids) + 1 > max_seq_length:
                self.filter_counts["overlength"] += 1
                continue
            self.examples.append({"kind": "ftpo_soft", "prompt_ids": ids, "rejected_token_id": rid[0], "weight": float(row.get("weight", 1.0)),
                                  "gamma": float(row.get("gamma", 4.0)), "tau": float(row.get("tau", 0.7))})
        self.filter_counts["kept_examples"] = len(self.examples)
        if not self.examples:
            raise ValueError("no SoftFTPO samples survived tokenisation and filters")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class DPOPairDataset(Dataset):
    """Ruminconfab-nudge-guard DPO retrain (DS #168-177): kind='dpo_pair' rows, each with
    a shared 'rendered_context' (real chat-template-rendered multi-turn text through the
    generation-prompt boundary, built offline by render_dpo_training_data.py -- NOT the
    pretty-print 'context' field, a real bug found+fixed this session), plus 'chosen' and
    'rejected' response strings and a per-row 'weight' (DS #172's hard/soft/weak-unverified
    weighting). Tokenizes ctx+chosen and ctx+rejected SEPARATELY (same boundary-detection
    + BPE-seam defensive check as SFTDataset, applied twice per example) since the DPO loss
    needs independent per-token log-probs for each side, not a shared sequence."""

    def __init__(self, rows, tokenizer, *, max_seq_length: int) -> None:
        self.examples: list[dict[str, Any]] = []
        self.max_seq_length = max_seq_length
        self.filter_counts: Counter[str] = Counter()

        def _tok_side(ctx: str, resp: str):
            enc = tokenizer(ctx + resp, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            boundary = len(ctx)
            plen = next((j for j, (a, b) in enumerate(offs) if a >= boundary), None)
            if plen is None or plen >= len(ids) - 1:
                return None, None, "no_response_tokens"
            decoded_resp = tokenizer.decode(ids[plen:])
            resp_check = resp.strip()[:12]
            if resp_check and not decoded_resp.strip().startswith(resp_check[:6]):
                return None, None, "boundary_mismatch"
            return ids, plen, None

        for row in rows:
            self.filter_counts["input_rows"] += 1
            if row.get("kind") not in ("hard", "soft"):
                self.filter_counts["not_dpo_pair"] += 1
                continue
            # DS #178 item 1: defensive check that context_messages is prompt-only (ends at
            # the last non-assistant turn) -- if the final assistant turn were already
            # included, the DPO loss would be computed on a sequence containing the target,
            # one turn longer than the model actually sees at inference.
            last_role = row["context_messages"][-1]["role"]
            if last_role == "assistant":
                raise ValueError(
                    f"DPOPairDataset: context_messages for task_id={row.get('task_id')} "
                    f"rep={row.get('rep')} ends with an assistant turn -- this must be "
                    f"prompt-only (rendered_context uses add_generation_prompt=True, chosen/"
                    f"rejected are appended as the completion separately)."
                )
            ctx = row["rendered_context"]
            chosen_ids, chosen_plen, err_c = _tok_side(ctx, row["chosen"])
            if err_c:
                self.filter_counts[f"chosen_{err_c}"] += 1
                continue
            rejected_ids, rejected_plen, err_r = _tok_side(ctx, row["rejected"])
            if err_r:
                self.filter_counts[f"rejected_{err_r}"] += 1
                continue
            if len(chosen_ids) + 1 > max_seq_length or len(rejected_ids) + 1 > max_seq_length:
                self.filter_counts["overlength"] += 1
                continue
            self.examples.append({
                "kind": "dpo_pair",
                "chosen_ids": chosen_ids, "chosen_prompt_len": chosen_plen,
                "rejected_ids": rejected_ids, "rejected_prompt_len": rejected_plen,
                "weight": float(row.get("weight", 1.0)),
                "task_id": row.get("task_id"), "rep": row.get("rep"),
            })
        self.filter_counts["kept_examples"] = len(self.examples)
        if not self.examples:
            raise ValueError("no DPO pair samples survived tokenisation and filters")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class KLReplayDataset(Dataset):
    """Ruminconfab DPO retrain KL-replay set (DS #177): kind='kl_replay' sequence rows
    (loop-safety/capability/B3/over-refusal domains -- letter-bias replay reuses the
    EXISTING LetterGroupDataset/letter_group format unchanged, not this class). Structurally
    identical to SFTDataset's tokenization/boundary-detection (same offset_mapping approach,
    same BPE-seam defensive check) -- deliberately NOT reusing SFTDataset directly despite
    the shared logic, since this row's loss (forward KL(pi_theta||pi_ref) per DS #177 section
    2, NOT plain CE) is routed by a distinct 'kind' tag."""

    def __init__(self, rows, tokenizer, *, max_seq_length: int) -> None:
        self.examples: list[dict[str, Any]] = []
        self.max_seq_length = max_seq_length
        self.filter_counts: Counter[str] = Counter()
        for row in rows:
            self.filter_counts["input_rows"] += 1
            if row.get("kind") != "sequence":
                self.filter_counts["not_sequence"] += 1
                continue
            ctx, resp = row["rendered_context"], row["response_text"]
            enc = tokenizer(ctx + resp, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            boundary = len(ctx)
            prompt_len = next((j for j, (a, b) in enumerate(offs) if a >= boundary), None)
            if prompt_len is None or prompt_len >= len(ids) - 1:
                self.filter_counts["no_response_tokens"] += 1
                continue
            if len(ids) + 1 > max_seq_length:
                self.filter_counts["overlength"] += 1
                continue
            decoded_resp = tokenizer.decode(ids[prompt_len:])
            resp_check = resp.strip()[:12]
            if resp_check and not decoded_resp.strip().startswith(resp_check[:6]):
                self.filter_counts["boundary_mismatch"] += 1
                continue
            self.examples.append({"kind": "kl_replay", "prompt_ids": ids, "prompt_len": prompt_len,
                                  "domain": row.get("domain"), "source_id": row.get("source_id")})
        self.filter_counts["kept_examples"] = len(self.examples)
        if not self.examples:
            raise ValueError("no KL-replay samples survived tokenisation and filters")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]
