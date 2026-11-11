# Derived from https://github.com/Liquid4All/antidoom (Copyright Liquid AI, Apache-2.0).
# Modified by Vixero Labs: additional losses, dataset kinds and training options. See NOTICE.
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F
from transformers.trainer_callback import TrainerCallback
from trl import DPOTrainer


def attach_agc(model, clip: float = 0.01, eps: float = 1e-3):
    def _agc_hook(grad, param):
        if grad is None:
            return grad
        param_norm = param.detach().norm()
        grad_norm = grad.norm()
        max_norm = clip * (param_norm + eps)
        if grad_norm > max_norm:
            grad = grad * (max_norm / (grad_norm + 1e-6))
        return grad

    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(lambda g, p=p: _agc_hook(g, p))


class ThresholdStop(TrainerCallback):
    def __init__(self, monitor: str, threshold: float, higher_is_better: bool):
        self.monitor = monitor
        self.threshold = threshold
        self.higher_is_better = higher_is_better

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.monitor not in logs:
            return
        value = logs[self.monitor]
        stop = (value >= self.threshold) if self.higher_is_better else (value <= self.threshold)
        if stop:
            control.should_training_stop = True
            print(
                f"[ThresholdStop] {self.monitor}={value:.4f} "
                f"crossed {'>=' if self.higher_is_better else '<='} "
                f"{self.threshold} - stopping."
            )


class EarlyStoppingByMetric(TrainerCallback):
    def __init__(
        self,
        monitor: str,
        higher_is_better: bool,
        patience: int = 10,
        min_delta: float = 0.0,
    ):
        self.monitor = monitor
        self.higher_is_better = higher_is_better
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.monitor not in logs:
            return

        current = logs[self.monitor]
        if self.best is None:
            self.best = current
            return

        improvement = current - self.best if self.higher_is_better else self.best - current
        if improvement > self.min_delta:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                control.should_training_stop = True
                print(f"[EarlyStopping] '{self.monitor}' plateaued (best={self.best:.5f}) - stopping training.")


class FTPOTrainer(DPOTrainer):
    """
    Reference FTPO trainer used by the Antidoom pipeline.

    The only changes here are import-localisation and ASCII log text; the
    collator/loss math is kept identical to the reference implementation.
    """

    def __init__(self, *args, **kwargs):
        train_dataset = kwargs.get("train_dataset")
        processor = kwargs.get("processing_class") or kwargs.get("tokenizer")
        super().__init__(*args, **kwargs)
        train_dataset = train_dataset or getattr(self, "train_dataset", None)
        self.ftpo_max_length = getattr(train_dataset, "max_seq_length", None)
        self.ftpo_pad_token_id = self._resolve_pad_token_id(processor)
        self._ftpo_stored_metrics = {
            "train": defaultdict(list),
            "eval": defaultdict(list),
        }
        self.remove_unused_columns = False
        self.data_collator = self.ftpo_collator

    @staticmethod
    def _get_proj(model):
        proj = model.get_output_embeddings()
        if proj is None:
            proj = getattr(model, "lm_head", None)
        if proj is None:
            raise AttributeError("Model lacks both get_output_embeddings() and lm_head.")
        return proj

    def _resolve_pad_token_id(self, processor=None) -> int:
        for obj in (
            processor,
            getattr(self, "processing_class", None),
            getattr(self, "tokenizer", None),
        ):
            pad_token_id = getattr(obj, "pad_token_id", None)
            if pad_token_id is not None:
                return int(pad_token_id)

        padding_value = getattr(self, "padding_value", None)
        if padding_value is not None:
            return int(padding_value)

        config = getattr(getattr(self, "model", None), "config", None)
        for attr in ("pad_token_id", "eos_token_id"):
            token_id = getattr(config, attr, None)
            if token_id is not None:
                return int(token_id)

        raise AttributeError("Could not determine pad token id for FTPO collator.")

    def _resolve_max_length(self) -> int:
        max_length = getattr(self, "ftpo_max_length", None)
        if max_length is None:
            max_length = getattr(self.args, "max_length", None)
        if max_length is None:
            raise AttributeError("Could not determine max sequence length for FTPO collator.")
        return int(max_length)

    @contextmanager
    def _ftpo_null_ref_context(self, model):
        null_ref_context = getattr(super(), "null_ref_context", None)
        if null_ref_context is not None:
            with null_ref_context():
                yield
            return

        accelerator = getattr(self, "accelerator", None)
        unwrapped_model = accelerator.unwrap_model(model) if accelerator is not None else model
        ref_adapter_name = getattr(self, "ref_adapter_name", None)
        model_adapter_name = getattr(self, "model_adapter_name", None)
        has_disable_adapter = hasattr(unwrapped_model, "disable_adapter")
        should_disable_adapter = (
            (getattr(self, "is_peft_model", False) or has_disable_adapter)
            and not ref_adapter_name
            and has_disable_adapter
        )
        adapter_model = unwrapped_model if hasattr(unwrapped_model, "set_adapter") else model

        with unwrapped_model.disable_adapter() if should_disable_adapter else nullcontext():
            if ref_adapter_name and hasattr(adapter_model, "set_adapter"):
                adapter_model.set_adapter(ref_adapter_name)
            yield
            if ref_adapter_name and hasattr(adapter_model, "set_adapter"):
                adapter_model.set_adapter(model_adapter_name or "default")

    def _store_ftpo_metrics(self, metrics, train_eval: str = "train") -> None:
        store_metrics = getattr(super(), "store_metrics", None)
        if store_metrics is not None:
            store_metrics(metrics, train_eval=train_eval)
            return

        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = value.detach().float().mean().item()
            else:
                value = float(value)
            self._ftpo_stored_metrics[train_eval][key].append(value)

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        train_eval = "eval" if any(key.startswith("eval") for key in logs) else "train"
        stored = self._ftpo_stored_metrics.get(train_eval, {})
        prefix = "" if train_eval == "train" else f"{train_eval}_"
        for key, values in stored.items():
            if values:
                logs[f"{prefix}{key}"] = sum(values) / len(values)
        stored.clear()
        return super().log(logs, *args, **kwargs)

    def ftpo_collator(self, features):
        pad_id = self.ftpo_pad_token_id
        max_cap = self._resolve_max_length()

        if features[0].get("kind") == "ftpo_soft":
            if len(features) != 1:
                raise ValueError("ftpo_soft collation requires batch_size == 1")
            f = features[0]
            seq = torch.tensor(f["prompt_ids"], dtype=torch.long).unsqueeze(0)
            if seq.size(1) + 1 > max_cap:
                raise ValueError("ftpo_soft sample exceeds max_seq_length; discard before collation")
            return dict(kind="ftpo_soft", prompt_ids=seq, attention_mask=torch.ones_like(seq, dtype=torch.bool), rejected_token_id=int(f["rejected_token_id"]),
                        weight=float(f["weight"]), gamma=float(f["gamma"]), tau=float(f["tau"]))

        if features[0].get("kind") == "loop_span":
            if len(features) != 1:
                raise ValueError("loop_span collation requires batch_size == 1")
            f = features[0]
            seq = torch.tensor(f["prompt_ids"], dtype=torch.long).unsqueeze(0)
            if seq.size(1) + 1 > max_cap:
                raise ValueError("loop_span sample exceeds max_seq_length; discard before collation")
            return dict(kind="loop_span", prompt_ids=seq, attention_mask=torch.ones_like(seq, dtype=torch.bool), j0=int(f["j0"]), K=int(f["K"]))

        if features[0].get("kind") == "sft_ce":
            if len(features) != 1:
                raise ValueError("sft_ce collation requires batch_size == 1")
            f = features[0]
            seq = torch.tensor(f["prompt_ids"], dtype=torch.long).unsqueeze(0)
            if seq.size(1) + 1 > max_cap:
                raise ValueError("sft_ce sample exceeds max_seq_length; discard before collation")
            return dict(kind="sft_ce", prompt_ids=seq, attention_mask=torch.ones_like(seq, dtype=torch.bool),
                        prompt_len=int(f["prompt_len"]))

        if features[0].get("kind") == "dpo_pair":
            if len(features) != 1:
                raise ValueError("dpo_pair collation requires batch_size == 1")
            f = features[0]
            chosen_seq = torch.tensor(f["chosen_ids"], dtype=torch.long).unsqueeze(0)
            rejected_seq = torch.tensor(f["rejected_ids"], dtype=torch.long).unsqueeze(0)
            if chosen_seq.size(1) + 1 > max_cap or rejected_seq.size(1) + 1 > max_cap:
                raise ValueError("dpo_pair sample exceeds max_seq_length; discard before collation")
            return dict(kind="dpo_pair",
                        chosen_ids=chosen_seq, chosen_attention_mask=torch.ones_like(chosen_seq, dtype=torch.bool),
                        chosen_prompt_len=int(f["chosen_prompt_len"]),
                        rejected_ids=rejected_seq, rejected_attention_mask=torch.ones_like(rejected_seq, dtype=torch.bool),
                        rejected_prompt_len=int(f["rejected_prompt_len"]),
                        weight=float(f["weight"]))

        if features[0].get("kind") == "kl_replay":
            if len(features) != 1:
                raise ValueError("kl_replay collation requires batch_size == 1")
            f = features[0]
            seq = torch.tensor(f["prompt_ids"], dtype=torch.long).unsqueeze(0)
            if seq.size(1) + 1 > max_cap:
                raise ValueError("kl_replay sample exceeds max_seq_length; discard before collation")
            return dict(kind="kl_replay", prompt_ids=seq, attention_mask=torch.ones_like(seq, dtype=torch.bool),
                        prompt_len=int(f["prompt_len"]))

        if features[0].get("kind") == "letter_group":
            if len(features) != 1:
                raise ValueError(
                    "letter_group collation requires batch_size == 1 "
                    f"(got {len(features)})"
                )
            feat = features[0]
            n = feat["n"]
            prompt_lists = feat["prompt_ids"]
            k = len(prompt_lists)
            if k != n:
                raise ValueError(
                    f"letter_group {feat.get('qid')}: len(prompt_ids)={k} != n={n}"
                )
            if len(feat["orders"]) != k or len(feat["gold_pos"]) != k:
                raise ValueError(
                    f"letter_group {feat.get('qid')}: orders/gold_pos length mismatch"
                )
            if len(feat["chosen_ids"]) != k or len(feat["rejected_ids"]) != k:
                raise ValueError(
                    f"letter_group {feat.get('qid')}: chosen/rejected length mismatch"
                )
            if len(feat["letter_ids"]) != n:
                raise ValueError(
                    f"letter_group {feat.get('qid')}: letter_ids length != n"
                )

            # Pad only to the longest render in THIS group (never to max_seq_length):
            # at batch_size==1 transforms' GDN layers skip padding masks, so any padding
            # beyond what is needed would be silently ingested by linear attention.
            max_len = max(len(p) for p in prompt_lists)

            prompt_ids = torch.full((k, max_len), pad_id, dtype=torch.long)
            attention_ms = torch.zeros_like(prompt_ids, dtype=torch.bool)
            for i, seq_list in enumerate(prompt_lists):
                seq = torch.tensor(seq_list, dtype=torch.long)
                if seq.size(0) + 1 > max_cap:
                    raise ValueError(
                        "FTPO sample exceeds max_seq_length; overlength samples must be "
                        "discarded before collation, not truncated"
                    )
                prompt_ids[i, -seq.size(0):] = seq
                attention_ms[i, -seq.size(0):] = True

            orders = torch.tensor(feat["orders"], dtype=torch.long)
            if orders.shape != (k, n):
                raise ValueError(
                    f"letter_group {feat.get('qid')}: orders shape {tuple(orders.shape)} != ({k}, {n})"
                )

            return dict(
                kind="letter_group",
                prompt_ids=prompt_ids,
                attention_mask=attention_ms,
                chosen_ids=torch.tensor(feat["chosen_ids"], dtype=torch.long),
                rejected_ids=torch.tensor(feat["rejected_ids"], dtype=torch.long),
                letter_ids=torch.tensor(feat["letter_ids"], dtype=torch.long),
                orders=orders,
                gold_pos=torch.tensor(feat["gold_pos"], dtype=torch.long),
                kl_replay_scale=float(feat.get("kl_replay_scale", 1.0)),
            )

        # ---- unchanged single-row path ----
        max_len = max(len(f["prompt_ids"]) for f in features)
        batch_sz = len(features)

        prompt_ids = torch.full((batch_sz, max_len), pad_id, dtype=torch.long)
        attention_ms = torch.zeros_like(prompt_ids, dtype=torch.bool)

        for i, feat in enumerate(features):
            seq = torch.tensor(feat["prompt_ids"], dtype=torch.long)
            if seq.size(0) + 1 > max_cap:
                raise ValueError(
                    "FTPO sample exceeds max_seq_length; overlength samples must be "
                    "discarded before collation, not truncated"
                )
            prompt_ids[i, -seq.size(0):] = seq
            attention_ms[i, -seq.size(0):] = True

        batch = dict(
            prompt_ids=prompt_ids,
            attention_mask=attention_ms,
            rejected_token_id=torch.tensor([f["rejected_token_id"] for f in features]),
        )

        max_c = max(len(f["chosen_ids"]) for f in features)
        chosen_pad = torch.full((batch_sz, max_c), pad_id, dtype=torch.long)
        chosen_mask = torch.zeros_like(chosen_pad, dtype=torch.bool)
        for i, f in enumerate(features):
            ids = torch.tensor(f["chosen_ids"], dtype=torch.long)
            chosen_pad[i, :ids.size(0)] = ids
            chosen_mask[i, :ids.size(0)] = True
        batch.update(chosen_ids=chosen_pad, chosen_mask=chosen_mask)

        return batch

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        if inputs.get("kind") == "letter_group":
            loss, metrics = self._letter_group_loss(model, inputs)
            # DS #179: letter-bias rows inside the ruminconfab DPO retrain's KL-replay set
            # carry an explicit kl_replay_scale (0.04525) so their contribution matches the
            # other replay rows' effective weight; defaults to 1.0 (no change) for every
            # other config that has never set this field, so this is a pure no-op elsewhere.
            scale = float(inputs.get("kl_replay_scale", 1.0))
            if scale != 1.0:
                loss = scale * loss
            return (loss, metrics) if return_outputs else loss
        if inputs.get("kind") == "ftpo_soft":
            loss, metrics = self._soft_ftpo_loss(model, inputs)
            return (loss, metrics) if return_outputs else loss
        if inputs.get("kind") == "loop_span":
            loss, metrics = self._loop_span_loss(model, inputs)
            return (loss, metrics) if return_outputs else loss
        if inputs.get("kind") == "sft_ce":
            loss, metrics = self._sft_ce_loss(model, inputs)
            return (loss, metrics) if return_outputs else loss
        if inputs.get("kind") == "dpo_pair":
            loss, metrics = self._dpo_pair_loss(model, inputs)
            return (loss, metrics) if return_outputs else loss
        if inputs.get("kind") == "kl_replay":
            loss, metrics = self._kl_replay_loss(model, inputs)
            return (loss, metrics) if return_outputs else loss

        lambda_mse_target = getattr(self, "lambda_mse_target", 0.05)
        tau_mse_target = getattr(self, "tau_mse_target", 1.0)
        lambda_mse = getattr(self, "lambda_mse", 0.4)
        clip_epsilon_logits = getattr(self, "clip_epsilon_logits", 2)

        USE_MSE_LOSS = True

        device = next(model.parameters()).device
        ids = inputs["prompt_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        B, L = ids.shape

        seq_len = attn.sum(1)
        pad_off = (L - seq_len).unsqueeze(1)
        arange_L = torch.arange(L, device=ids.device).unsqueeze(0)
        pos_full = (arange_L - pad_off).clamp(min=0)
        pos_full = pos_full.masked_fill(attn == 0, 0)

        outputs = model(
            ids,
            attention_mask=attn,
            position_ids=pos_full,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )

        logits_last = outputs.logits[:, -1, :].float()
        logp_all = F.log_softmax(logits_last, dim=-1)

        ch_ids = inputs["chosen_ids"].to(device)
        ch_mask = inputs["chosen_mask"].to(device)
        rejected = inputs["rejected_token_id"].to(device)
        logp_bad = logp_all.gather(-1, rejected.unsqueeze(-1)).squeeze(-1)

        batch_rows = torch.arange(B, device=logp_all.device).unsqueeze(1)
        delta_tok = logits_last[batch_rows, ch_ids] - logits_last.gather(
            -1, rejected.unsqueeze(-1)
        )
        weights = (
            torch.clamp(
                (clip_epsilon_logits - delta_tok) / clip_epsilon_logits,
                0.0,
                1.0,
            )
            * ch_mask
        )

        tau = 1.0
        gap = clip_epsilon_logits - delta_tok
        per_tok_loss = F.softplus(gap / tau)

        chosen_counts = ch_mask.sum(dim=-1).clamp(min=1)
        pref_loss = ((per_tok_loss * weights).sum(dim=-1) / chosen_counts).mean()

        extra_metrics = {}

        if USE_MSE_LOSS:
            with torch.no_grad():
                if self.ref_model is None:
                    with self._ftpo_null_ref_context(model):
                        ref_logits_last = model(
                            ids,
                            attention_mask=attn,
                            position_ids=pos_full,
                            use_cache=False,
                            return_dict=True,
                            logits_to_keep=1,
                        ).logits[:, -1, :].float()
                else:
                    ref_logits_last = self.ref_model(
                        ids,
                        attention_mask=attn,
                        position_ids=pos_full,
                        use_cache=False,
                        return_dict=True,
                        logits_to_keep=1,
                    ).logits[:, -1, :].float()

            tether_mask = torch.ones_like(logits_last, dtype=torch.bool)
            rows = torch.arange(B, device=ch_ids.device).unsqueeze(1).expand_as(ch_ids)
            tether_mask[rows[ch_mask], ch_ids[ch_mask]] = False
            tether_mask.scatter_(1, rejected.unsqueeze(-1), False)

            diff = logits_last - ref_logits_last
            mse_elem_raw = (tether_mask * diff.pow(2)).sum() / tether_mask.sum()

            tgt_mask = torch.zeros_like(logits_last, dtype=torch.bool)
            rows = torch.arange(B, device=ch_ids.device).unsqueeze(1).expand_as(ch_ids)
            tgt_mask[rows[ch_mask], ch_ids[ch_mask]] = True
            tgt_mask.scatter_(1, rejected.unsqueeze(-1), True)

            if lambda_mse_target:
                diff_tok = logits_last - ref_logits_last
                diff_tok = diff_tok * tgt_mask
                excess_tok = torch.clamp(diff_tok.abs() - tau_mse_target, min=0.0)
                mse_target_raw = (excess_tok.pow(2)).sum() / tgt_mask.sum()
            else:
                mse_target_raw = logits_last.new_tensor(0.0)

            mse_loss = lambda_mse * mse_elem_raw + lambda_mse_target * mse_target_raw
            loss = pref_loss + mse_loss

            extra_metrics.update(
                {
                    "mse_elem": mse_elem_raw.detach(),
                    "mse_tgt_tokenwise": mse_target_raw.detach(),
                }
            )

        else:
            loss = pref_loss

        lp_chosen = logp_all.gather(-1, ch_ids)
        lp_bad = logp_bad.unsqueeze(-1)

        wins_tok = (lp_chosen > lp_bad) & ch_mask
        frac_win = wins_tok.float().sum(-1) / ch_mask.sum(-1).clamp(min=1e-8)
        chosen_win = frac_win.mean().detach()

        active_delta = delta_tok[ch_mask]
        active_weights = weights[ch_mask]
        margin_win = (
            ((delta_tok >= clip_epsilon_logits) & ch_mask).float().sum()
            / ch_mask.float().sum().clamp(min=1e-8)
        ).detach()
        mean_delta = active_delta.mean().detach()
        median_delta = active_delta.median().detach()
        active_weight = active_weights.mean().detach()

        metrics = {
            "pref_loss": pref_loss.detach(),
            "chosen_win": chosen_win,
            "margin_win": margin_win,
            "mean_delta": mean_delta,
            "median_delta": median_delta,
            "active_weight": active_weight,
            **extra_metrics,
        }
        self._store_ftpo_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics
        return loss

    def _sft_ce_loss(self, model, inputs):
        """Beat-ThinkingCap track (DS consult #110): plain next-token cross-entropy on the FULL,
        natural, untruncated response only (prompt tokens masked out of the loss entirely, not
        just down-weighted). No preference pair, no reference model, no tether -- this is the
        one loss mode in this trainer that is genuinely just SFT, added because the existing
        loop_span/ftpo_soft mechanisms (truncate-and-suppress at some onset point) are exactly
        what caused the effreason-round-2 regression when the onset was mis-specified; here there
        is no onset to get wrong, the target is the model's own real, complete, verified-correct,
        naturally-concise trace."""
        # OOM fix (2026-09-26, KL-sweep lambda=0.03 crash: "Tried to allocate 3.45 GiB" inside this
        # function): this loss originally computed full-vocab (248k) logits over ALL L sequence
        # positions on BOTH forward passes (main + ref), each .float()'d into its own persistent
        # copy -- exactly the "full-vocab logits over all positions" memory hog every other loss
        # branch in this file already avoids via logits_to_keep (see module docstring/history).
        # Only response positions ever matter here, so: (1) request logits_to_keep=keep (only the
        # trailing positions that are actually response tokens, computed inside the model's own
        # lm_head instead of sliced out afterward -- real compute+memory savings, not just a view);
        # (2) let log_softmax do its own internal fp32 upcast via dtype= instead of a separate
        # persistent .float() copy of the raw logits; (3) run the (larger, transient-only) no-grad
        # ref forward BEFORE the grad-tracked main forward and free it explicitly, so its peak
        # doesn't stack on top of the main forward's graph-retained activations (previously: main
        # forward's graph was already alive+held for backward while ref's own forward pass peaked
        # concurrently -- the actual source of the OOM, not sequence length alone: the same rows
        # trained fine under the non-KL concise-sft run just before this).
        device = next(model.parameters()).device
        ids = inputs["prompt_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        prompt_len = int(inputs["prompt_len"])
        L = ids.size(1)
        pos = torch.arange(L, device=device).unsqueeze(0)
        keep = max(L - prompt_len + 1, 1)  # trailing positions prompt_len-1 .. L-1
        targets = ids[0, prompt_len:]  # one target per kept predicting position (after dropping the last)
        n_resp = targets.numel()
        assert n_resp > 0, "SFTDataset should already guarantee at least one response token"

        kl_lambda = getattr(self, "sft_kl_lambda", 0.0)
        kl = ids.new_zeros((), dtype=torch.float32)
        if kl_lambda > 0:
            # Beat-ThinkingCap KL-anchor (DS #114a): a SECOND, frozen PEFT adapter named "ref" was
            # loaded with a known-good (loop-safe) checkpoint's weights at model build time (see
            # ftpo_train.py). Switch to it for a no-grad reference forward, then back to "default"
            # -- same base weights, two different LoRA adapters, exactly DS's recommended mechanism
            # (avoids loading a second full base model). Done FIRST and freed before the main
            # forward (see OOM-fix note above).
            accelerator = getattr(self, "accelerator", None)
            unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
            with torch.no_grad():
                unwrapped.set_adapter("ref")
                ref_out = model(ids, attention_mask=attn, position_ids=pos, use_cache=False,
                                 return_dict=True, logits_to_keep=keep)
                unwrapped.set_adapter("default")
                ref_logits = ref_out.logits[0, :-1, :]
                log_p_ref = F.log_softmax(ref_logits, dim=-1, dtype=torch.float32).detach()
                p_ref = log_p_ref.exp()
            del ref_out, ref_logits
            torch.cuda.empty_cache()

        out = model(ids, attention_mask=attn, position_ids=pos, use_cache=False, return_dict=True,
                    logits_to_keep=keep)
        logits = out.logits[0, :-1, :]  # positions prompt_len-1 .. L-2, predicting response tokens
        logp = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        ce = nll.sum() / n_resp

        if kl_lambda > 0:
            kl_per_tok = (p_ref * (log_p_ref - logp)).sum(dim=-1)
            kl = kl_per_tok.sum() / n_resp
        loss = ce + kl_lambda * kl

        with torch.no_grad():
            top1 = (logp.argmax(-1) == targets).sum() / n_resp
            ppl = ce.exp()
        metrics = {"pref_loss": loss.detach(), "sft_ce_loss": ce.detach(), "sft_kl_loss": kl.detach(),
                   "sft_top1": top1, "sft_ppl": ppl, "sft_n_resp_tokens": torch.tensor(float(n_resp))}
        self._store_ftpo_metrics(metrics, train_eval="train")
        return loss, metrics

    def _kl_replay_loss(self, model, inputs):
        """Ruminconfab DPO retrain KL-replay anchor (DS #180, correcting #177's literal
        formula): kind='kl_replay'. D_KL(p_ref || p_theta), reference-weighted -- the SAME
        established direction as _sft_ce_loss's own KL-anchor term (deliberately NOT
        factored into a shared helper despite DS's suggestion: _sft_ce_loss is an
        already-validated/gated function used across concise_sft/malay/ruminconfab rounds,
        and duplicating this ~15-line block here is a smaller risk than touching it).
        Per-example mean over response tokens; the kl_replay_weight (0.2 default) is applied
        here directly, so this function returns the already-weighted loss, matching how
        every other loss kind in this file returns its own fully-weighted value."""
        device = next(model.parameters()).device
        ids = inputs["prompt_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        prompt_len = int(inputs["prompt_len"])
        L = ids.size(1)
        pos = torch.arange(L, device=device).unsqueeze(0)
        keep = max(L - prompt_len + 1, 1)
        n_resp = L - prompt_len
        assert n_resp > 0, "KLReplayDataset should already guarantee at least one response token"

        accelerator = getattr(self, "accelerator", None)
        unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
        with torch.no_grad():
            unwrapped.set_adapter("ref")
            ref_out = model(ids, attention_mask=attn, position_ids=pos, use_cache=False,
                             return_dict=True, logits_to_keep=keep)
            unwrapped.set_adapter("default")
            # Real OOM hit here on a 13k-char capability-domain replay row (~3000+ response
            # tokens): materializing log_p_ref + p_ref as full-vocab (248k) FP32 tensors over
            # every response position at once (~3-6GB) on top of an already ~41GB-loaded
            # training process pushed past the 44.39GB card. Fix: keep ref_logits in its
            # ORIGINAL (bf16) dtype here -- no full-sequence FP32 upcast -- and do the FP32
            # log-softmax/KL math in small CHUNKS below instead (same fix category as
            # _sft_ce_loss's own documented 2026-09-26 OOM fix, applied to a longer-sequence
            # case that mechanism hadn't been exercised against before).
            ref_logits = ref_out.logits[0, :-1, :].detach()
        del ref_out
        torch.cuda.empty_cache()

        out = model(ids, attention_mask=attn, position_ids=pos, use_cache=False, return_dict=True,
                    logits_to_keep=keep)
        logits = out.logits[0, :-1, :]  # grad-tracked, original dtype -- NOT upcast here either

        # Chunked KL sum: bounds peak memory regardless of response length (256 positions'
        # worth of FP32 vocab tensors alive at once, ~1GB, instead of the full response).
        CHUNK = 256
        kl_sum = logits.new_zeros((), dtype=torch.float32)
        for c0 in range(0, n_resp, CHUNK):
            c1 = min(c0 + CHUNK, n_resp)
            log_p_ref_chunk = F.log_softmax(ref_logits[c0:c1], dim=-1, dtype=torch.float32)
            p_ref_chunk = log_p_ref_chunk.exp()
            logp_chunk = F.log_softmax(logits[c0:c1], dim=-1, dtype=torch.float32)
            kl_chunk = (p_ref_chunk * (log_p_ref_chunk - logp_chunk)).sum(dim=-1)  # D_KL(p_ref||p_theta)
            kl_sum = kl_sum + kl_chunk.sum()
        kl = kl_sum / n_resp  # per-example mean over response tokens (DS #177 sec.2/#180)

        kl_weight = getattr(self, "kl_replay_weight", 0.2)
        loss = kl_weight * kl

        # NOTE: _store_ftpo_metrics calls float() on every non-tensor value, so 'domain'
        # (a string) cannot go in here directly -- would crash on the very next logging call.
        # Caught before it could fire (checked _store_ftpo_metrics's implementation).
        metrics = {"kl_replay_loss": kl.detach(), "kl_replay_n_resp": torch.tensor(float(n_resp))}
        self._store_ftpo_metrics(metrics, train_eval="train")
        return loss, metrics

    def _dpo_side_logp(self, model, ids, attn, prompt_len, ref: bool):
        """Shared helper for _dpo_pair_loss: sequence log-prob sum over response tokens
        (log pi(y|x) = sum_t log p(y_t|x,y_<t)), under either the trainable policy
        ('default' adapter, grad-tracked) or the frozen DPO reference (self.dpo_ref_adapter_name
        -- DS #184: MUST be a separate adapter from self.ref_adapter_name/"ref", which is the
        KL-replay safety anchor to c10. DPO's own reference is standard DPO's initial policy,
        s1337 here -- conflating the two, as #168's original spec did, makes the DPO objective
        "prefer chosen over rejected relative to c10" instead of "relative to your starting
        policy," a different and non-standard algorithm; DS corrected this in #184).
        Returns (seq_logp_sum, per-token logp for the response tokens, n_resp).
        Chunked over response positions (same fix class as _kl_replay_loss's OOM fix): the
        real retrain_dpo_pairs_v1_1 data has a pair up to 8974 tokens (task P2_comb1) --
        ~3x the row that OOM'd the unchunked KL-replay path at 3000 tokens -- so this is
        chunked proactively rather than waiting for a second OOM to prove it's needed."""
        device = next(model.parameters()).device
        L = ids.size(1)
        pos = torch.arange(L, device=device).unsqueeze(0)
        keep = max(L - prompt_len + 1, 1)
        targets = ids[0, prompt_len:]
        n_resp = targets.numel()
        CHUNK = 256

        def _run():
            out = model(ids, attention_mask=attn, position_ids=pos, use_cache=False,
                        return_dict=True, logits_to_keep=keep)
            logits = out.logits[0, :-1, :]  # [n_resp, vocab], original dtype -- no full upcast
            chunks = []
            for c0 in range(0, n_resp, CHUNK):
                c1 = min(c0 + CHUNK, n_resp)
                logp_chunk = F.log_softmax(logits[c0:c1], dim=-1, dtype=torch.float32)
                tok_logp_chunk = logp_chunk.gather(-1, targets[c0:c1].unsqueeze(-1)).squeeze(-1)
                chunks.append(tok_logp_chunk)
            return torch.cat(chunks, dim=0)

        if ref:
            dpo_ref_name = getattr(self, "dpo_ref_adapter_name", None) or "ref"
            accelerator = getattr(self, "accelerator", None)
            unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
            with torch.no_grad():
                unwrapped.set_adapter(dpo_ref_name)
                tok_logp = _run().detach()
                unwrapped.set_adapter("default")
        else:
            tok_logp = _run()
        return tok_logp.sum(), tok_logp, n_resp

    def _dpo_pair_loss(self, model, inputs):
        """Ruminconfab nudge-guard DPO retrain (DS #168-177): kind='dpo_pair'.
        L_DPO = -log_sigmoid(beta * [(logpi_theta(chosen)-logpi_theta(rejected))
                                      - (logpi_ref(chosen)-logpi_ref(rejected))])
        L_SFT (DS #177 item 3) = CE on the chosen response's own tokens, token-mean,
        weight dpo_sft_weight (0.05) -- reuses the SAME policy-side chosen forward already
        computed for the DPO term, no extra forward pass. pi_ref = frozen 'ref' adapter
        (letterfix2-c10), via _ftpo_null_ref_context (same mechanism _soft_ftpo_loss uses).
        Per-row 'weight' (DS #172's hard/soft/weak-unverified scaling, 0.4 for the weaker
        unverified-fire pathway, 1.0 otherwise) multiplies the row's whole combined loss.
        Standard DPO per your #177 answer: sum of token log-probs, NOT length-normalized;
        beta=0.1 default."""
        device = next(model.parameters()).device
        beta = getattr(self, "dpo_beta", 0.1)
        dpo_sft_weight = getattr(self, "dpo_sft_weight", 0.05)
        row_weight = float(inputs.get("weight", 1.0))

        chosen_ids = inputs["chosen_ids"].to(device)
        chosen_attn = inputs["chosen_attention_mask"].to(device)
        rejected_ids = inputs["rejected_ids"].to(device)
        rejected_attn = inputs["rejected_attention_mask"].to(device)
        chosen_plen = int(inputs["chosen_prompt_len"])
        rejected_plen = int(inputs["rejected_prompt_len"])

        logp_theta_chosen, chosen_tok_logp, n_resp_chosen = self._dpo_side_logp(
            model, chosen_ids, chosen_attn, chosen_plen, ref=False)
        logp_theta_rejected, _, _ = self._dpo_side_logp(
            model, rejected_ids, rejected_attn, rejected_plen, ref=False)
        logp_ref_chosen, _, _ = self._dpo_side_logp(
            model, chosen_ids, chosen_attn, chosen_plen, ref=True)
        logp_ref_rejected, _, _ = self._dpo_side_logp(
            model, rejected_ids, rejected_attn, rejected_plen, ref=True)

        delta_theta = logp_theta_chosen - logp_theta_rejected
        delta_ref = logp_ref_chosen - logp_ref_rejected
        margin = beta * (delta_theta - delta_ref)
        dpo_loss = -F.logsigmoid(margin)

        sft_loss = -chosen_tok_logp.sum() / n_resp_chosen  # per-token mean, DS #177 item 3

        loss = row_weight * (dpo_loss + dpo_sft_weight * sft_loss)

        with torch.no_grad():
            chosen_win = (delta_theta > delta_ref).float()
        metrics = {
            "dpo_loss": dpo_loss.detach(), "dpo_sft_loss": sft_loss.detach(),
            "dpo_margin": margin.detach(), "dpo_chosen_win": chosen_win,
            "dpo_delta_theta": delta_theta.detach(), "dpo_delta_ref": delta_ref.detach(),
            "dpo_row_weight": torch.tensor(row_weight),
        }
        self._store_ftpo_metrics(metrics, train_eval="train")
        return loss, metrics

    def _soft_ftpo_loss(self, model, inputs):
        """Round-2 soft target at ONE position (first token of a repeat): q = 0.5 p_ref + 0.5 softmax((log p_ref - gamma*1[rej])/tau); loss = weight * KL(q || p_adapter).
        p_ref = adapter-disabled model (frozen base). Near-certain repeats (p_ref(rej)~1) get a gently reduced target instead of an unbounded push."""
        device = next(model.parameters()).device
        ids = inputs["prompt_ids"].to(device); attn = inputs["attention_mask"].to(device)
        L = ids.size(1); pos = torch.arange(L, device=device).unsqueeze(0)
        y = int(inputs["rejected_token_id"]); gamma = float(inputs["gamma"]); tau = float(inputs["tau"]); w = float(inputs["weight"])
        lg = model(ids, attention_mask=attn, position_ids=pos, use_cache=False, return_dict=True, logits_to_keep=1).logits[0, -1, :].float()
        with torch.no_grad():
            with self._ftpo_null_ref_context(model):
                rf = model(ids, attention_mask=attn, position_ids=pos, use_cache=False, return_dict=True, logits_to_keep=1).logits[0, -1, :].float()
            lpr = F.log_softmax(rf, dim=-1)
            shifted = lpr.clone(); shifted[y] = shifted[y] - gamma
            q = 0.5 * lpr.exp() + 0.5 * F.softmax(shifted / tau, dim=-1)
            logq = torch.log(q.clamp_min(1e-30))
        lp = F.log_softmax(lg, dim=-1)
        kl = (q * (logq - lp)).sum()
        loss = w * kl
        with torch.no_grad():
            py = lp[y].exp(); py_ref = lpr[y].exp()
        metrics = {"pref_loss": kl.detach(), "soft_kl": kl.detach(), "soft_py": py, "soft_py_ref": py_ref, "soft_w": lg.new_tensor(w)}
        self._store_ftpo_metrics(metrics, train_eval="train")
        return loss, metrics

    def _loop_span_loss(self, model, inputs):
        """Unlikelihood-style hinge on the SECOND occurrence of a repeated unit: for each of K target tokens y_t (teacher-forced),
        hinge = relu(logit[y_t] - max_{v!=y_t} logit[v] + margin), i.e. y_t must sit >= margin BELOW the best alternative.
        Tether: MSE of all non-target vocab logits vs the frozen base (adapter disabled) at the same K positions."""
        if getattr(self, "loop_span_mode", "hinge") == "klmask":
            return self._loop_span_klmask_loss(model, inputs)
        margin = getattr(self, "loop_span_margin", 1.0)
        lambda_mse = getattr(self, "loop_span_lambda_mse", 1.0)
        device = next(model.parameters()).device
        ids = inputs["prompt_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        j0, K = int(inputs["j0"]), int(inputs["K"])
        L = ids.size(1)
        assert L == j0 + K, (L, j0, K)
        pos = torch.arange(L, device=device).unsqueeze(0)
        out = model(ids, attention_mask=attn, position_ids=pos, use_cache=False, return_dict=True, logits_to_keep=K + 1)
        logits = out.logits[0, :K, :].float()            # positions j0-1 .. j0+K-2 predict ids[j0 .. j0+K-1]
        y = ids[0, j0:j0 + K]
        y_logit = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        other = logits.scatter(-1, y.unsqueeze(-1), float("-inf")).max(dim=-1).values
        hinge = F.relu(y_logit - other + margin).mean()
        with torch.no_grad():
            with self._ftpo_null_ref_context(model):
                ref = model(ids, attention_mask=attn, position_ids=pos, use_cache=False, return_dict=True, logits_to_keep=K + 1).logits[0, :K, :].float()
        tether = torch.ones_like(logits, dtype=torch.bool)
        tether.scatter_(-1, y.unsqueeze(-1), False)
        mse = (tether * (logits - ref).pow(2)).sum() / tether.sum()
        loss = hinge + lambda_mse * mse
        with torch.no_grad():
            top1 = (logits.argmax(-1) == y).float().mean()
            gap = (y_logit - other).mean()
            ref_top1 = (ref.argmax(-1) == y).float().mean()
        metrics = {"pref_loss": hinge.detach(), "loop_hinge": hinge.detach(), "loop_mse": mse.detach(), "loop_top1": top1, "loop_ref_top1": ref_top1, "loop_gap": gap.detach(), "mse_elem": mse.detach()}
        self._store_ftpo_metrics(metrics, train_eval="train")
        return loss, metrics

    def _loop_span_klmask_loss(self, model, inputs):
        """v6: at each of the K copy positions the target distribution q_t = p_base with the copied token y_t masked out and renormalised
        (probability mass moves to what the BASE already considers plausible, never to an arbitrary end token); loss = KL(q_t || p_adapter).
        Tether: KL(p_base || p_adapter) on the T positions just before the copy window (the tail of the FIRST occurrence: natural text) keeps generation healthy."""
        lam_t = getattr(self, "loop_span_lambda_tether", 1.0)
        device = next(model.parameters()).device
        ids = inputs["prompt_ids"].to(device); attn = inputs["attention_mask"].to(device)
        j0, K = int(inputs["j0"]), int(inputs["K"]); L = ids.size(1); assert L == j0 + K
        T = max(0, min(int(getattr(self, "loop_span_tether_tokens", 24)), j0 - 2))
        pos = torch.arange(L, device=device).unsqueeze(0)
        keep = T + K + 1
        lg = model(ids, attention_mask=attn, position_ids=pos, use_cache=False, return_dict=True, logits_to_keep=keep).logits[0].float()
        with torch.no_grad():
            with self._ftpo_null_ref_context(model):
                rf = model(ids, attention_mask=attn, position_ids=pos, use_cache=False, return_dict=True, logits_to_keep=keep).logits[0].float()
        y = ids[0, j0:j0 + K]
        lp = F.log_softmax(lg[T:T + K], dim=-1); lp_ref = F.log_softmax(rf[T:T + K], dim=-1)
        mix = getattr(self, "loop_span_q_mix", 0.5); gamma = getattr(self, "loop_span_gamma", 3.0); tau = getattr(self, "loop_span_tau", 1.3)
        rfc = rf[T:T + K]
        if mix <= 0.0:                                                     # pure mask + renormalise
            q_logits = rfc.scatter(-1, y.unsqueeze(-1), float("-inf"))
            q = F.softmax(q_logits, dim=-1)
        else:                                                              # DS 64: smooth shift, no renormalising of tiny residuals
            lpr = F.log_softmax(rfc, dim=-1)
            shifted = (lpr - gamma * F.one_hot(y, lpr.size(-1)).to(lpr.dtype)) / tau
            q = mix * lpr.exp() + (1.0 - mix) * F.softmax(shifted, dim=-1)
        logq = torch.log(q.clamp_min(1e-30))
        kl_copy = (q * (logq - lp)).sum(-1).mean()
        if T > 0:
            lpt = F.log_softmax(lg[:T], dim=-1); lpt_ref = F.log_softmax(rf[:T], dim=-1)
            kl_tether = (lpt_ref.exp() * (lpt_ref - lpt)).sum(-1).mean()
        else:
            kl_tether = lg.new_tensor(0.0)
        eos = [i for i in getattr(self, "loop_span_eos_ids", []) if 0 <= i < lg.size(-1)]
        if eos:
            allp = F.softmax(lg, dim=-1)[:, eos].sum(-1); allp_ref = F.softmax(rf, dim=-1)[:, eos].sum(-1)
            eos_pen = F.relu(allp - 1.5 * allp_ref).pow(2).mean() * 100.0     # probability-scale penalty, x100 so it is comparable to KL terms
        else:
            eos_pen = lg.new_tensor(0.0)
        loss = kl_copy + lam_t * kl_tether + getattr(self, "loop_span_mu_eos", 2.0) * eos_pen
        with torch.no_grad():
            p_y = lp.gather(-1, y.unsqueeze(-1)).squeeze(-1).exp().mean(); p_y_ref = lp_ref.gather(-1, y.unsqueeze(-1)).squeeze(-1).exp().mean()
            top1 = (lg[T:T + K].argmax(-1) == y).float().mean(); ref_top1 = (rf[T:T + K].argmax(-1) == y).float().mean()
        metrics = {"pref_loss": kl_copy.detach(), "loop_eos_pen": eos_pen.detach(), "loop_kl_copy": kl_copy.detach(), "loop_kl_tether": kl_tether.detach(), "loop_py": p_y, "loop_py_ref": p_y_ref, "loop_top1": top1, "loop_ref_top1": ref_top1, "mse_elem": kl_tether.detach()}
        self._store_ftpo_metrics(metrics, train_eval="train")
        return loss, metrics

    def _letter_group_loss(self, model, inputs):
        lambda_mse = getattr(self, "lambda_mse", 0.4)
        clip_epsilon_logits = getattr(self, "clip_epsilon_logits", 2)
        lambda_pbm = getattr(self, "lambda_pbm", 0.5)
        lambda_H = getattr(self, "lambda_H", 0.1)

        device = next(model.parameters()).device
        ids = inputs["prompt_ids"].to(device)
        attn = inputs["attention_mask"].to(device).bool()
        K, L = ids.shape

        chosen_ids = inputs["chosen_ids"].to(device).long()
        rejected_ids = inputs["rejected_ids"].to(device).long()
        letter_ids = inputs["letter_ids"].to(device).long()
        orders = inputs["orders"].to(device).long()
        gold_pos = inputs["gold_pos"].to(device).long()
        n = letter_ids.numel()

        seq_len = attn.sum(1)
        pad_off = (L - seq_len).unsqueeze(1)
        arange_L = torch.arange(L, device=device).unsqueeze(0)
        pos_full = (arange_L - pad_off).clamp(min=0)
        pos_full = pos_full.masked_fill(attn == 0, 0)

        outputs = model(
            ids,
            attention_mask=attn,
            position_ids=pos_full,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        logits_last = outputs.logits[:, -1, :].float()  # [K, V]

        rows = torch.arange(K, device=device)

        # (1) clipped-hinge FTPO preference, one chosen/rejected token per render
        delta_tok = logits_last[rows, chosen_ids] - logits_last[rows, rejected_ids]
        weights = torch.clamp(
            (clip_epsilon_logits - delta_tok) / clip_epsilon_logits, 0.0, 1.0
        )
        hinge = (F.softplus(clip_epsilon_logits - delta_tok) * weights).mean()

        # (2) letter-position probability mass balance
        letter_logits = logits_last[:, letter_ids]  # [K, n]
        p = F.softmax(letter_logits, dim=-1)
        p_c = p.clamp(min=1e-8)
        logp = p_c.log()

        pc = torch.zeros(K, n, device=device, dtype=p.dtype)
        pc.scatter_(1, orders, p)  # pc[r, c] = prob of ORIGINAL content c in render r
        log_pc = pc.clamp(min=1e-8).log()
        log_pc_dev = log_pc - log_pc.mean(dim=0, keepdim=True)
        b_log = log_pc_dev.pow(2).mean(dim=0).mean()

        # (3) letter entropy
        H = -(p_c * logp).sum(dim=-1).mean()

        # (4) tether to frozen base (null-ref context), non-target vocab only
        with torch.no_grad():
            if self.ref_model is None:
                with self._ftpo_null_ref_context(model):
                    ref_logits_last = model(
                        ids,
                        attention_mask=attn,
                        position_ids=pos_full,
                        use_cache=False,
                        return_dict=True,
                        logits_to_keep=1,
                    ).logits[:, -1, :].float()
            else:
                ref_logits_last = self.ref_model(
                    ids,
                    attention_mask=attn,
                    position_ids=pos_full,
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=1,
                ).logits[:, -1, :].float()

        tether_mask = torch.ones_like(logits_last, dtype=torch.bool)
        tether_mask[rows, chosen_ids] = False
        tether_mask[rows, rejected_ids] = False
        tether_mask[:, letter_ids] = False  # all option letters are intentionally moved (hinge/PBM); tether protects the rest of the vocab
        diff = logits_last - ref_logits_last
        mse_elem_raw = (tether_mask * diff.pow(2)).sum() / tether_mask.sum().clamp(min=1)
        mse_loss = lambda_mse * mse_elem_raw

        loss = getattr(self, "letter_hinge_weight", 1.0) * hinge + lambda_pbm * b_log + lambda_H * H + mse_loss

        # metrics (mirroring existing conventions: detached tensors + _store_ftpo_metrics)
        pred_pos = letter_logits.argmax(dim=-1)  # [K]
        letter_acc = (pred_pos == gold_pos).float().mean()
        pred_content = orders[rows, pred_pos]
        consistency = torch.tensor(
            float((pred_content == pred_content[0]).all().item()), device=device
        )

        metrics = {
            "pref_loss": hinge.detach(),
            "letter_hinge": hinge.detach(),
            "letter_pbm": b_log.detach(),
            "letter_H": H.detach(),
            "letter_mse": mse_elem_raw.detach(),
            "letter_acc": letter_acc.detach(),
            "letter_consistency": consistency.detach(),
        }
        self._store_ftpo_metrics(metrics, train_eval="train")

        return loss, metrics

    def _prepare_dataset(self, dataset, *args, **_):
        return dataset
