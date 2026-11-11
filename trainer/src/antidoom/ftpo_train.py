# Derived from https://github.com/Liquid4All/antidoom (Copyright Liquid AI, Apache-2.0).
# Modified by Vixero Labs: additional losses, dataset kinds and training options. See NOTICE.
from __future__ import annotations

import functools
import inspect
import logging
import math
import getpass
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import cached_file
from trl import DPOConfig, DPOTrainer

from antidoom.config import TrainConfig
from antidoom.ftpo_data import (
    ConcatFTPODataset,
    DPOPairDataset,
    FTPODataset,
    KLReplayDataset,
    LetterGroupDataset,
    LoopSpanDataset,
    SFTDataset,
    SoftFTPODataset,
    has_trimmed_source_prompt,
    read_jsonl,
    row_source_label,
    sample_rows,
)
from antidoom.checkpoint_guard import DiskGuardCallback, get_last_complete_checkpoint
from antidoom.letter_eval import LetterEvalCallback
from antidoom.reference_ftpo_trainer import FTPOTrainer, ThresholdStop

logger = logging.getLogger(__name__)


from transformers import TrainerCallback


class DisableInputGradsCallback(TrainerCallback):
    """HF's gradient_checkpointing_enable() force-enables embedding-output requires_grad, which makes autograd traverse and RECOMPUTE all
    64 layers although only the last 10 (+lm_head) carry adapters. Removing the hook (after HF re-adds it) skips backward/recompute for
    the frozen layers. Validated by antidoom_data/b3_test.py (identical loss/grad-norm, measured speedup)."""

    def _off(self, model):
        # Three places register a require-grads forward hook on the input embedding (our explicit call, HF gradient_checkpointing_enable,
        # TRL DPOTrainer.__init__); disable_input_require_grads() removes only the LAST handle, so clear them all.
        try:
            emb = model.get_input_embeddings()
            n = len(emb._forward_hooks)
            emb._forward_hooks.clear()
            logger.info("cleared %d forward hook(s) on input embeddings", n)
        except Exception as e:
            logger.info("could not clear embedding hooks: %s", e)

    def on_train_begin(self, args, state, control, model=None, **kw):
        if model is not None:
            self._off(model)
            logger.info("input-require-grads hook removed (frozen layers skip backward/recompute)")

    def on_step_begin(self, args, state, control, model=None, **kw):
        if model is not None and state.global_step == 0:
            self._off(model)


def _format_float(value: float) -> str:
    text = f"{value:.8g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        sign = ""
        if exponent.startswith(("+", "-")):
            sign, exponent = exponent[0], exponent[1:]
        exponent = exponent.lstrip("0") or "0"
        if sign == "+":
            sign = ""
        text = f"{mantissa}e{sign}{exponent}"
    return text


def _format_epochs(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else _format_float(value)


def _format_counter(counter: Counter[str], limit: int = 8) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counter.most_common(limit))


def _sanitize_name_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._+-]+", "-", value.strip())
    value = value.strip("-_")
    return value or "unknown"


def _model_basename(model_name: str) -> str:
    return model_name.rstrip("/").rsplit("/", 1)[-1]


def _parse_parent_model_name(model_name: str) -> dict[str, str]:
    name = _model_basename(model_name)
    size_match = re.search(r"(\d+B)", name, flags=re.IGNORECASE)
    parent_match = re.search(r"_(\d+)_HF(?:_|$)", name)
    if parent_match is None:
        numbers = re.findall(r"\d+", name)
        parent_id = numbers[-1] if numbers else "unknown"
    else:
        parent_id = parent_match.group(1)
    family_match = re.search(r"(liquidthink\d*)", name, flags=re.IGNORECASE)
    return {
        "size": size_match.group(1) if size_match else "model",
        "parent_id": parent_id,
        "family": family_match.group(1) if family_match else "",
    }


def build_merged_model_name(cfg: TrainConfig, *, train_examples: int) -> str:
    parent = _parse_parent_model_name(cfg.model_name)
    user = _sanitize_name_part(getpass.getuser())
    job_id = _sanitize_name_part(os.environ.get("SLURM_JOB_ID") or "local")
    parts = [
        f"{user}_antidoom{parent['size']}",
        f"from{parent['parent_id']}",
    ]
    if parent["family"]:
        parts.append(parent["family"])
    parts.extend(
        [
            f"{train_examples}-samples",
            f"lr{_format_float(cfg.learning_rate)}",
            f"ep{_format_epochs(cfg.num_epochs)}",
            job_id,
            "HF",
        ]
    )
    return _sanitize_name_part("_".join(parts))


def resolve_merged_output_dir(cfg: TrainConfig, *, train_examples: int) -> Path:
    if not cfg.auto_merged_output_name:
        return cfg.merged_output_dir
    return cfg.merged_output_dir.parent / build_merged_model_name(
        cfg,
        train_examples=train_examples,
    )


def filter_trimmed_source_prompt_rows(rows: list[dict]) -> tuple[list[dict], int]:
    kept = [row for row in rows if not has_trimmed_source_prompt(row)]
    return kept, len(rows) - len(kept)


def _make_dpo_config(cfg: TrainConfig) -> DPOConfig:
    kwargs = {
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "warmup_ratio": cfg.warmup_ratio,
        "num_train_epochs": cfg.num_epochs,
        "learning_rate": cfg.learning_rate,
        "logging_steps": cfg.logging_steps,
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "optim": cfg.optim,
        "seed": cfg.seed,
        "output_dir": str(cfg.output_dir),
        "max_length": cfg.max_seq_length,
        "max_prompt_length": cfg.max_seq_length // 2,
        "beta": 0.1,
        "weight_decay": cfg.weight_decay,
        "report_to": "tensorboard",
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "bf16": cfg.bf16,
        "fp16": False,
        "remove_unused_columns": False,
        "disable_tqdm": False,
        "max_grad_norm": cfg.max_grad_norm,
        # Must be set here, not just via model.gradient_checkpointing_enable() before
        # Trainer construction: HF Trainer re-syncs the model's checkpointing state from
        # args.gradient_checkpointing when it wraps the model, silently undoing any prior
        # manual enable() call. This was the actual root cause of the "fix" in the model-
        # setup code below appearing to do nothing (identical ~406s/it timing and ~45.4GB
        # peak memory to a run with no checkpointing at all, step 1 through 7).
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    supported_params = inspect.signature(DPOConfig).parameters
    supported_kwargs = {key: value for key, value in kwargs.items() if key in supported_params}
    dropped = sorted(set(kwargs) - set(supported_kwargs))
    if dropped:
        logger.info("DPOConfig does not support these arguments; dropping: %s", ", ".join(dropped))
    return DPOConfig(**supported_kwargs)


def _scoped_target_regex(cfg: TrainConfig, n_layers: int) -> str:
    """Regex (PEFT fullmatch) selecting target modules only in the last n_layers_unfrozen layers,
    plus lm_head if requested. Structural fix: frozen layers never receive an adapter, so a
    trainability reset (see the TRL "ref" adapter issue below) cannot re-enable them."""
    excl = set(getattr(cfg, "lora_exclude_modules", []) or [])
    names = "|".join(re.escape(t) for t in sorted(cfg.target_modules) if t != "lm_head" and t not in excl)
    lset = getattr(cfg, "lora_layer_set", None)
    layers = "|".join(str(i) for i in (sorted(lset) if lset else range(n_layers - cfg.n_layers_unfrozen, n_layers)))
    pattern = rf".*\.layers\.({layers})\.(.*\.)?({names})"
    if "lm_head" in cfg.target_modules:
        pattern = rf"({pattern})|lm_head"
    return pattern


def _make_lora_config(cfg: TrainConfig, n_layers: int | None = None) -> LoraConfig:
    targets = cfg.target_modules
    if cfg.freeze_early_layers and n_layers:
        targets = _scoped_target_regex(cfg, n_layers)
        logger.info("LoRA target_modules regex (last %d of %d layers): %s", cfg.n_layers_unfrozen, n_layers, targets)
    kwargs = {
        "r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "bias": "none",
        "target_modules": targets,
    }
    rp = getattr(cfg, "lora_rank_pattern", None)
    if rp:
        kwargs["rank_pattern"] = {k: int(v) for k, v in rp.items()}
        kwargs["alpha_pattern"] = {k: int(v) for k, v in rp.items()}   # alpha = rank (scale 1) per module type
    if cfg.lora_ensure_weight_tying:
        kwargs["ensure_weight_tying"] = True
    supported_params = inspect.signature(LoraConfig).parameters
    supported_kwargs = {key: value for key, value in kwargs.items() if key in supported_params}
    dropped = sorted(set(kwargs) - set(supported_kwargs))
    if dropped:
        logger.info("LoraConfig does not support these arguments; dropping: %s", ", ".join(dropped))
    return LoraConfig(**supported_kwargs)


def _discover_transformer_blocks(base_model) -> tuple[str, list | torch.nn.ModuleList]:
    candidate_paths = (
        "model.transformer.layers",
        "model.layers",
        "model.decoder.layers",
        "layers",
        "transformer.layers",
        "transformer.h",
        "backbone.layers",
    )
    for path in candidate_paths:
        obj = base_model
        found = True
        for part in path.split("."):
            if not hasattr(obj, part):
                found = False
                break
            obj = getattr(obj, part)
        if found and isinstance(obj, (list, torch.nn.ModuleList)) and len(obj) > 0:
            return path, obj

    seen: list[tuple[str, int, str]] = []
    best_name = ""
    best_mod = None
    best_score = (-1, -1)
    for name, mod in base_model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) < 2:
            continue
        first = mod[0]
        seen.append((name, len(mod), type(first).__name__))
        looks_like_block = any(
            hasattr(first, attr)
            for attr in ("self_attn", "attention", "self_attention", "mlp", "ffn", "mixer")
        ) or any(word in type(first).__name__.lower() for word in ("layer", "block"))
        score = (len(mod), 1 if looks_like_block else 0)
        if score > best_score:
            best_name = name
            best_mod = mod
            best_score = score

    if best_mod is not None:
        return best_name, best_mod

    details = ", ".join(
        f"{name}(len={length}, first={child_type})"
        for name, length, child_type in sorted(seen, key=lambda item: -item[1])[:15]
    )
    raise RuntimeError(f"Could not locate transformer layers list. Scanned ModuleLists: {details}")


def _compile_layer_regex(layers_path: str) -> re.Pattern[str]:
    return re.compile(rf"(?:^|\.){re.escape(layers_path)}\.(\d+)\.")


def freeze_lora_to_last_k_layers(
    peft_model,
    *,
    k: int,
    target_modules: tuple[str, ...],
) -> None:
    base = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    layers_path, blocks = _discover_transformer_blocks(base)
    num_layers = len(blocks)
    cut = max(0, num_layers - int(k))
    layer_regex = _compile_layer_regex(layers_path)

    def matches_target(name: str) -> bool:
        return any(token in name for token in target_modules)

    frozen = 0
    trainable = 0
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            continue
        match = layer_regex.search(name)
        if match is not None:
            layer_idx = int(match.group(1))
            if layer_idx < cut:
                param.requires_grad_(False)
                frozen += param.numel()
            else:
                param.requires_grad_(True)
                trainable += param.numel()
        elif matches_target(name):
            param.requires_grad_(True)
            trainable += param.numel()
        else:
            param.requires_grad_(False)
            frozen += param.numel()

    offenders = []
    for name, param in peft_model.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        match = layer_regex.search(name)
        if match is not None and int(match.group(1)) < cut:
            offenders.append(name)
    if offenders:
        raise AssertionError(
            "LoRA params outside the last trainable layers remained trainable: "
            + ", ".join(offenders[:20])
        )

    logger.info(
        "LoRA layer freeze: path=%s, total_layers=%d, keeping_last=%d, frozen_params=%d, trainable_params=%d",
        layers_path,
        num_layers,
        int(k),
        frozen,
        trainable,
    )


def log_trainable_lora_by_layer(peft_model) -> None:
    base = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    try:
        layers_path, _ = _discover_transformer_blocks(base)
        layer_regex = _compile_layer_regex(layers_path)
    except Exception as exc:
        logger.warning("could not summarize trainable LoRA by layer: %s", exc)
        return

    counts: Counter[int] = Counter()
    lm_head_trainable = False
    total = 0
    for name, param in peft_model.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        total += param.numel()
        match = layer_regex.search(name)
        if match is not None:
            counts[int(match.group(1))] += param.numel()
        if "lm_head" in name:
            lm_head_trainable = True
    layer_summary = ", ".join(f"{layer}={count}" for layer, count in sorted(counts.items()))
    logger.info("trainable LoRA params by layer: %s", layer_summary or "none")
    logger.info("trainable lm_head LoRA: %s; total trainable LoRA params: %d", lm_head_trainable, total)


def _adapter_key_with_default_name(key: str, name: str = "default") -> str:
    replacements = {
        ".lora_A.weight": f".lora_A.{name}.weight",
        ".lora_B.weight": f".lora_B.{name}.weight",
        ".lora_embedding_A.weight": f".lora_embedding_A.{name}",
        ".lora_embedding_B.weight": f".lora_embedding_B.{name}",
        ".modules_to_save.weight": f".modules_to_save.{name}.weight",
        ".modules_to_save.bias": f".modules_to_save.{name}.bias",
    }
    for old, new in replacements.items():
        key = key.replace(old, new)
    return key


def _load_adapter_state_direct(model, adapter_dir: Path, adapter_name: str = "default") -> None:
    """Beat-ThinkingCap KL-anchor (DS #114a): adapter_name lets this load the SAME saved weights
    into a differently-named PEFT adapter (e.g. 'ref', a frozen second copy alongside the
    trainable 'default'), not just the default one every prior call site used."""
    from safetensors.torch import load_file

    adapter_path = adapter_dir / "adapter_model.safetensors"
    if adapter_path.exists():
        state = load_file(adapter_path)
    else:
        adapter_path = adapter_dir / "adapter_model.bin"
        state = torch.load(adapter_path, map_location="cpu")

    state = {_adapter_key_with_default_name(key, adapter_name): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Beat-ThinkingCap KL-anchor fix: when a SECOND adapter already exists on the model (e.g.
    # loading "ref" after "default" is already populated), `missing` naturally includes every
    # OTHER adapter's params too (this narrower call only supplies `state` for `adapter_name`) --
    # that's expected, not an error. Only flag a missing key as real if it belongs to THIS
    # adapter_name specifically (parseable from its own naming: '.default.'/'.ref.'/etc.).
    bad_missing = [key for key in missing if (f".{adapter_name}." in key or key.endswith(f".{adapter_name}"))
                   and ("lora_" in key or "modules_to_save" in key)]
    if bad_missing or unexpected:
        raise RuntimeError(
            "Adapter state did not load cleanly. "
            f"bad_missing={bad_missing[:40]}, unexpected={unexpected[:40]}"
        )


def _torch_dtype_from_name(name: str):
    dtypes = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in dtypes:
        raise ValueError(f"unsupported dtype {name!r}; expected one of {sorted(dtypes)}")
    return dtypes[name]


def merge_lora_adapter(
    *,
    model_name: str,
    adapter_dir: Path,
    merged_output_dir: Path,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    max_shard_size: str = "5GB",
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=_torch_dtype_from_name(dtype),
        device_map=device_map,
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig.from_pretrained(adapter_dir)
    model = get_peft_model(base, lora_cfg)
    _load_adapter_state_direct(model, adapter_dir)

    logger.info("merging LoRA adapter from %s into %s", adapter_dir, model_name)
    merged = model.merge_and_unload()
    merged.eval()
    merged_output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(
        merged_output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(merged_output_dir)
    _restore_parent_config_json(model_name, merged_output_dir)
    logger.info("saved merged model to %s", merged_output_dir)


def _restore_parent_config_json(model_name: str, output_dir: Path) -> None:
    """Keep raw parent config metadata compatible with downstream loaders.

    Some model families round-trip through `save_pretrained` with a newer
    Transformers config schema than vLLM/eval infra expects. LoRA merging does
    not change architecture, so the raw parent config is the safest metadata to
    ship alongside the merged weights.
    """
    source: Path | None = None
    model_path = Path(model_name)
    if model_path.is_dir() and (model_path / "config.json").exists():
        source = model_path / "config.json"
    else:
        try:
            cached = cached_file(
                model_name,
                "config.json",
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
        except Exception as exc:
            logger.warning("could not locate parent config.json for %s: %s", model_name, exc)
            cached = None
        if cached:
            source = Path(cached)

    if source is None or not source.exists():
        logger.warning("parent config.json not found; leaving merged config as saved")
        return

    destination = output_dir / "config.json"
    shutil.copyfile(source, destination)
    logger.info("restored parent config.json from %s", source)


def _save_merged_model(trainer: FTPOTrainer, tokenizer, cfg: TrainConfig) -> None:
    if not cfg.merge_lora:
        logger.info("skipping LoRA merge because merge_lora=false")
        return
    if cfg.load_in_4bit:
        raise ValueError("merge_lora=true is not supported with load_in_4bit=true")
    accelerator = getattr(trainer, "accelerator", None)
    model_to_merge = (
        accelerator.unwrap_model(trainer.model) if accelerator is not None else trainer.model
    )
    if not hasattr(model_to_merge, "merge_and_unload"):
        raise AttributeError("trained model does not expose merge_and_unload(); cannot save merged model")

    logger.info("merging LoRA adapter into base model")
    merged_model = model_to_merge.merge_and_unload()
    merged_model.eval()
    cfg.merged_output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(cfg.merged_output_dir, safe_serialization=True)
    tokenizer.save_pretrained(cfg.merged_output_dir)
    _restore_parent_config_json(cfg.model_name, cfg.merged_output_dir)
    logger.info("saved merged model to %s", cfg.merged_output_dir)


def run_train(cfg: TrainConfig) -> None:
    if cfg.merge_lora and cfg.load_in_4bit:
        raise ValueError("merge_lora=true is not supported with load_in_4bit=true")
    if cfg.dataset_jsonl is None:
        raise ValueError("train.dataset_jsonl is required")
    if not cfg.model_name:
        raise ValueError("model_name is required for training")

    logger.info(
        "training paths: dataset_jsonl=%s, output_dir=%s, merged_output_dir=%s",
        cfg.dataset_jsonl,
        cfg.output_dir,
        cfg.merged_output_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(cfg.dataset_jsonl)
    logger.info("loaded %d FTPO rows from %s", len(rows), cfg.dataset_jsonl)
    logger.info(
        "raw rows by source: %s",
        _format_counter(Counter(row_source_label(row) for row in rows)),
    )
    logger.info(
        "raw rows by rejected token: %s",
        _format_counter(Counter(str(row.get("rejected_decoded")) for row in rows)),
    )
    logger.info(
        "raw rows by chosen candidate count: %s",
        _format_counter(Counter(str(len(row.get("multi_chosen_decoded") or [])) for row in rows)),
    )

    rows_without_trimmed_prompts, trimmed_prompt_rows = filter_trimmed_source_prompt_rows(rows)
    if trimmed_prompt_rows:
        logger.info(
            "%d/%d rows pass trimmed source_prompt filter; discarded %d rows",
            len(rows_without_trimmed_prompts),
            len(rows),
            trimmed_prompt_rows,
        )

    # Letter (MCQ position-bias) rows are single-token preferences at the 'Answer:' position. Their tokens (' A',' B',...)
    # repeat thousands of times, so the loop-data rebalancing (_rejected_normalisation / chosen regularisation) would
    # delete most of them: keep them out of sample_rows and build them with regularisation and stop-word filtering OFF.
    letter_rows = [r for r in rows_without_trimmed_prompts if r.get("kind") == "letter"]
    group_rows = [r for r in rows_without_trimmed_prompts if r.get("kind") == "letter_group"]
    span_rows = [r for r in rows_without_trimmed_prompts if r.get("kind") == "loop_span"]
    soft_rows = [r for r in rows_without_trimmed_prompts if r.get("kind") == "ftpo_soft"]
    sft_rows = [r for r in rows_without_trimmed_prompts if r.get("kind") == "sft_ce"]
    # Ruminconfab nudge-guard DPO retrain (DS #168-180): 'hard'/'soft' = DPO preference
    # pairs (DPOPairDataset), 'sequence' = KL-replay anchor rows (KLReplayDataset). Letter-
    # bias KL-replay reuses kind='letter_group' above, no new extraction needed for it.
    dpo_pair_rows = [r for r in rows_without_trimmed_prompts if r.get("kind") in ("hard", "soft")]
    kl_replay_rows = [r for r in rows_without_trimmed_prompts if r.get("kind") == "sequence"]
    rows_without_trimmed_prompts = [r for r in rows_without_trimmed_prompts if r.get("kind") not in ("letter", "letter_group", "loop_span", "ftpo_soft", "sft_ce")]
    logger.info("split off %d letter rows; %d loop rows go through sampling", len(letter_rows), len(rows_without_trimmed_prompts))

    min_chosen_rows = [
        row
        for row in rows_without_trimmed_prompts
        if len(row.get("multi_chosen_decoded") or []) >= cfg.min_chosen_tokens
    ]
    logger.info(
        "%d/%d rows pass min_chosen_tokens=%d before tokenization",
        len(min_chosen_rows),
        len(rows_without_trimmed_prompts),
        cfg.min_chosen_tokens,
    )

    sampled_rows = sample_rows(
        rows_without_trimmed_prompts,
        max_train_examples=cfg.max_train_examples,
        rejected_strength=cfg.rejected_regularisation_strength,
        min_chosen_tokens=cfg.min_chosen_tokens,
        seed=cfg.seed,
        source_balance_mode=cfg.source_balance_mode,
    )
    logger.info(
        "sampled %d/%d rows after pre-tokenization filters "
        "(max_train_examples=%s, min_chosen_tokens=%d, source_balance_mode=%s)",
        len(sampled_rows),
        len(rows_without_trimmed_prompts),
        cfg.max_train_examples,
        cfg.min_chosen_tokens,
        cfg.source_balance_mode,
    )
    logger.info(
        "sampled rows by source: %s",
        _format_counter(Counter(row_source_label(row) for row in sampled_rows)),
    )

    train_dataset = None
    if sampled_rows:
        train_dataset = FTPODataset(
            sampled_rows,
            tokenizer,
            max_seq_length=cfg.max_seq_length,
            filter_rejected_stop_words=cfg.filter_rejected_stop_words,
            chosen_regularisation_strength=cfg.chosen_regularisation_strength,
        )
        logger.info(
            "tokenized FTPO dataset kept %d/%d sampled rows; filter counts: %s",
            len(train_dataset),
            len(sampled_rows),
            _format_counter(train_dataset.filter_counts, limit=12),
        )
    else:
        logger.info("no loop rows in the dataset file: training on letter rows only")
    if soft_rows:
        soft_dataset = SoftFTPODataset(soft_rows, tokenizer, max_seq_length=cfg.max_seq_length)
        logger.info("soft-FTPO dataset kept %d/%d rows; filter counts: %s", len(soft_dataset), len(soft_rows), _format_counter(soft_dataset.filter_counts, limit=8))
        train_dataset = soft_dataset if train_dataset is None else ConcatFTPODataset(train_dataset, soft_dataset)
    if span_rows:
        span_dataset = LoopSpanDataset(span_rows, tokenizer, max_seq_length=cfg.max_seq_length, span_tokens=cfg.loop_span_tokens)
        logger.info("loop-span dataset kept %d/%d rows; filter counts: %s", len(span_dataset), len(span_rows), _format_counter(span_dataset.filter_counts, limit=8))
        train_dataset = span_dataset if train_dataset is None else ConcatFTPODataset(train_dataset, span_dataset)
    if sft_rows:
        sft_dataset = SFTDataset(sft_rows, tokenizer, max_seq_length=cfg.max_seq_length)
        logger.info("SFT-CE dataset kept %d/%d rows; filter counts: %s", len(sft_dataset), len(sft_rows), _format_counter(sft_dataset.filter_counts, limit=8))
        train_dataset = sft_dataset if train_dataset is None else ConcatFTPODataset(train_dataset, sft_dataset)
    if group_rows:
        group_dataset = LetterGroupDataset(group_rows, tokenizer, max_seq_length=cfg.max_seq_length)
        logger.info("letter-group dataset kept %d/%d groups; filter counts: %s", len(group_dataset), len(group_rows),
                    _format_counter(group_dataset.filter_counts, limit=8))
        train_dataset = group_dataset if train_dataset is None else ConcatFTPODataset(train_dataset, group_dataset)
    if letter_rows:
        letter_dataset = FTPODataset(
            letter_rows,
            tokenizer,
            max_seq_length=cfg.max_seq_length,
            filter_rejected_stop_words=False,
            chosen_regularisation_strength=0.0,
        )
        logger.info("letter dataset kept %d/%d rows; filter counts: %s", len(letter_dataset), len(letter_rows),
                    _format_counter(letter_dataset.filter_counts, limit=8))
        train_dataset = letter_dataset if train_dataset is None else ConcatFTPODataset(train_dataset, letter_dataset)
    if dpo_pair_rows:
        dpo_dataset = DPOPairDataset(dpo_pair_rows, tokenizer, max_seq_length=cfg.max_seq_length)
        logger.info("DPO-pair dataset kept %d/%d rows; filter counts: %s", len(dpo_dataset), len(dpo_pair_rows),
                    _format_counter(dpo_dataset.filter_counts, limit=8))
        train_dataset = dpo_dataset if train_dataset is None else ConcatFTPODataset(train_dataset, dpo_dataset)
    if kl_replay_rows:
        kl_dataset = KLReplayDataset(kl_replay_rows, tokenizer, max_seq_length=cfg.max_seq_length)
        logger.info("KL-replay dataset kept %d/%d rows; filter counts: %s", len(kl_dataset), len(kl_replay_rows),
                    _format_counter(kl_dataset.filter_counts, limit=8))
        train_dataset = kl_dataset if train_dataset is None else ConcatFTPODataset(train_dataset, kl_dataset)
    effective_batch = cfg.batch_size * cfg.gradient_accumulation_steps
    estimated_steps = math.ceil(len(train_dataset) * cfg.num_epochs / effective_batch)
    logger.info(
        "training schedule: examples=%d, batch_size=%d, gradient_accumulation_steps=%d, "
        "effective_batch=%d, num_epochs=%s, estimated_optimizer_steps=%d",
        len(train_dataset),
        cfg.batch_size,
        cfg.gradient_accumulation_steps,
        effective_batch,
        cfg.num_epochs,
        estimated_steps,
    )
    cfg.merged_output_dir = resolve_merged_output_dir(
        cfg,
        train_examples=len(train_dataset),
    )
    if cfg.merge_lora:
        logger.info("merged model output directory: %s", cfg.merged_output_dir)

    if cfg.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=quant,
            device_map={"": 0},
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            device_map={"": 0},
            trust_remote_code=True,
        )

    # Gradient checkpointing: NOT enabled anywhere in this trainer by default,
    # which is almost certainly the real cause of the repeated OOMs tonight.
    # v5's own production trainer (v5q_train_prod.py, same model/hardware)
    # enables this by default and measures peak memory at 33.7G/46G even on a
    # worst-case 6144-token row with a second (KL) forward pass -- far below
    # where we've been sitting (39-45G) without it.
    #
    # Attempt 1: manual gradient_checkpointing_enable()+enable_input_require_grads()
    # called BEFORE get_peft_model() -- no measurable memory/timing change, flag
    # likely didn't survive PEFT wrapping.
    # Attempt 2: prepare_model_for_kbit_training() (canonical helper) -- OOM'd
    # almost immediately, 6s in, on lm_head's forward, 44GB already in use before
    # backward ever ran. Root cause: this helper upcasts all non-4bit params to
    # fp32 for numerical stability -- on a 28.8B model with a ~250k vocab hitting
    # lm_head (one of our target_modules), that upcast itself is the memory cost,
    # unrelated to checkpointing. Dropping that helper; keeping only the specific
    # ordering fix (enable AFTER get_peft_model, not before) without the upcast.
    model.config.use_cache = False

    n_layers = getattr(model.config, "num_hidden_layers", None) or getattr(
        getattr(model.config, "text_config", None), "num_hidden_layers", None
    )
    lora_cfg = _make_lora_config(cfg, n_layers)
    model = get_peft_model(model, lora_cfg)
    if getattr(cfg, "init_adapter_path", None):
        _load_adapter_state_direct(model, Path(cfg.init_adapter_path))
        logger.info("initialised LoRA weights from %s (fresh optimizer/scheduler)", cfg.init_adapter_path)

    # Beat-ThinkingCap KL-anchor (DS #114a): a SECOND, frozen PEFT adapter ("ref") loaded with the
    # SAME weights as an anchor checkpoint (typically the same one 'default' was just initialised
    # from), so the trainer can compute lambda*KL(p_ref||p_new) and stop plain SFT from drifting
    # away from a known-good adapter with nothing holding it in place. Opt-in: no effect unless
    # sft_kl_anchor_path is set (backward compatible with every prior training round).
    if getattr(cfg, "sft_kl_anchor_path", None) and getattr(cfg, "sft_kl_lambda", 0.0) > 0:
        model.add_adapter("ref", lora_cfg)
        _load_adapter_state_direct(model, Path(cfg.sft_kl_anchor_path), adapter_name="ref")
        n_frozen = 0
        for name, param in model.named_parameters():
            if "ref" in name.split(".") and param.requires_grad:
                param.requires_grad_(False)
                n_frozen += 1
        model.set_adapter("default")
        logger.info("KL-anchor: loaded frozen 'ref' adapter from %s (lambda=%s), froze %d ref params",
                    cfg.sft_kl_anchor_path, cfg.sft_kl_lambda, n_frozen)

    # Ruminconfab DPO retrain (DS #184, correcting #168): a SEPARATE frozen adapter for the
    # DPO reference specifically -- standard DPO uses the INITIAL POLICY as its reference,
    # not the same anchor as the KL-replay term. "ref" above stays c10 (the KL-replay safety
    # anchor); "dpo_ref" here is loaded from dpo_ref_adapter_path (= init_adapter_path = s1337
    # in this run's config). Opt-in via dpo_ref_adapter_path, no effect otherwise.
    if getattr(cfg, "dpo_ref_adapter_path", None):
        model.add_adapter("dpo_ref", lora_cfg)
        _load_adapter_state_direct(model, Path(cfg.dpo_ref_adapter_path), adapter_name="dpo_ref")
        n_frozen_dpo_ref = 0
        for name, param in model.named_parameters():
            if "dpo_ref" in name.split(".") and param.requires_grad:
                param.requires_grad_(False)
                n_frozen_dpo_ref += 1
        model.set_adapter("default")
        logger.info("DPO-ref: loaded frozen 'dpo_ref' adapter from %s, froze %d params",
                    cfg.dpo_ref_adapter_path, n_frozen_dpo_ref)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    logger.info("post-get_peft_model: is_gradient_checkpointing=%s",
                getattr(model, "is_gradient_checkpointing", None))
    if cfg.freeze_early_layers and not getattr(cfg, "lora_layer_set", None):
        freeze_lora_to_last_k_layers(
            model,
            k=cfg.n_layers_unfrozen,
            target_modules=tuple(cfg.target_modules),
        )
        log_trainable_lora_by_layer(model)
    model.train()
    model.print_trainable_parameters()

    init_params = inspect.signature(DPOTrainer.__init__).parameters
    processing_kw = {"processing_class" if "processing_class" in init_params else "tokenizer": tokenizer}

    trainer = FTPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=train_dataset,
        **processing_kw,
        args=_make_dpo_config(cfg),
    )

    # TRL DPOTrainer.__init__ calls model.add_adapter("ref", ...) for a PeftModel with ref_model=None.
    # PEFT's inject_adapter() ends with set_adapter(active_adapters), which re-enables requires_grad
    # on EVERY lora_A/lora_B of "default" -- this silently un-froze all 64 layers in round 1. FTPOTrainer
    # never uses "ref" (its null-reference path is disable_adapter()), and it costs ~7.7GB of fp32 VRAM.
    #
    # Beat-ThinkingCap KL-anchor (DS #114a): if WE already created a "ref" adapter (loaded with a
    # real anchor checkpoint) before trainer construction, do NOT blindly delete it here -- but
    # ALSO do not trust that TRL's own add_adapter("ref", ...) call inside FTPOTrainer.__init__
    # left our weights alone (it may have re-created "ref" fresh/zero-init'd, silently clobbering
    # our anchor). Re-load + re-freeze unconditionally in that case, cheap insurance against a
    # subtle correctness bug that would otherwise be invisible (no crash, just a wrong KL target).
    peft_model = trainer.accelerator.unwrap_model(trainer.model)
    using_kl_anchor = bool(getattr(cfg, "sft_kl_anchor_path", None) and getattr(cfg, "sft_kl_lambda", 0.0) > 0)
    if "ref" in getattr(peft_model, "peft_config", {}):
        if using_kl_anchor:
            _load_adapter_state_direct(peft_model, Path(cfg.sft_kl_anchor_path), adapter_name="ref")
            n_frozen = 0
            for name, param in peft_model.named_parameters():
                if "ref" in name.split(".") and param.requires_grad:
                    param.requires_grad_(False)
                    n_frozen += 1
            peft_model.set_adapter("default")
            logger.info("KL-anchor: re-loaded+re-froze 'ref' adapter after trainer construction "
                        "(guards against TRL's own add_adapter('ref',...) clobbering it), froze %d params",
                        n_frozen)
            # Ruminconfab onset-token FTPO (DS #141): point the FTPO trainer's OWN internal
            # reference computation (_ftpo_null_ref_context, used by the hard-FTPO and
            # ftpo_soft loss paths for lambda_mse / p_ref) at this frozen "ref" adapter
            # instead of its default behaviour (disable_adapter() -> base model). Every prior
            # round (v8/v9b/v9c) left this unset and so tethered to BASE; this run explicitly
            # tethers to whatever checkpoint sft_kl_anchor_path points at (letterfix2-c10 for
            # this run) instead, per DS's explicit design ("preserve the adapter" not "regress
            # toward base"). No 'kind=sft_ce' rows exist in this run's data, so sft_kl_lambda>0
            # here has NO OTHER effect (that separate SFT-CE+KL loss path never fires) -- its
            # only real job in this run is satisfying `using_kl_anchor` above so 'ref' gets
            # loaded with real weights instead of deleted.
            trainer.ref_adapter_name = "ref"
            trainer.model_adapter_name = "default"
            logger.info("FTPO reference tether: ref_adapter_name='ref' (loaded from %s), "
                        "model_adapter_name='default' -- FTPO null-ref now uses this adapter, "
                        "not base", cfg.sft_kl_anchor_path)
        else:
            peft_model.delete_adapter("ref")
            peft_model.set_adapter("default")
            torch.cuda.empty_cache()
            logger.info("deleted unused TRL 'ref' adapter")

    # DPO-ref (DS #184): TRL only ever calls add_adapter("ref", ...) by that hardcoded name,
    # so "dpo_ref" is not at risk of the same clobbering -- still verify+re-freeze as cheap
    # insurance rather than assume.
    if getattr(cfg, "dpo_ref_adapter_path", None) and "dpo_ref" in getattr(peft_model, "peft_config", {}):
        _load_adapter_state_direct(peft_model, Path(cfg.dpo_ref_adapter_path), adapter_name="dpo_ref")
        n_frozen_dpo_ref = 0
        for name, param in peft_model.named_parameters():
            if "dpo_ref" in name.split(".") and param.requires_grad:
                param.requires_grad_(False)
                n_frozen_dpo_ref += 1
        peft_model.set_adapter("default")
        trainer.dpo_ref_adapter_name = "dpo_ref"
        logger.info("DPO-ref: verified+re-froze 'dpo_ref' adapter after trainer construction "
                    "(loaded from %s), froze %d params; trainer.dpo_ref_adapter_name='dpo_ref'",
                    cfg.dpo_ref_adapter_path, n_frozen_dpo_ref)
    if cfg.freeze_early_layers and not getattr(cfg, "lora_layer_set", None):
        freeze_lora_to_last_k_layers(
            peft_model, k=cfg.n_layers_unfrozen, target_modules=tuple(cfg.target_modules)
        )
    first_trainable = n_layers - cfg.n_layers_unfrozen if (cfg.freeze_early_layers and n_layers) else 0
    layer_set = set(getattr(cfg, "lora_layer_set", None) or [])
    if layer_set:
        first_trainable = 0
    bad, n_trainable = [], 0
    for name, prm in peft_model.named_parameters():
        if not prm.requires_grad:
            continue
        n_trainable += prm.numel()
        m = re.search(r"\.layers\.(\d+)\.", name)
        if m and int(m.group(1)) < first_trainable:
            bad.append(name)
        if layer_set and m and int(m.group(1)) not in layer_set:
            bad.append(name)
    if bad:
        raise RuntimeError(f"FREEZE FAILED: {len(bad)} trainable params in frozen layers, e.g. {bad[:3]}")
    logger.info("freeze verified after trainer init: %d trainable params, none in layers < %d", n_trainable, first_trainable)
    # avoid peft forcing a duplicate copy of the (unmodified) base lm_head into every checkpoint
    peft_model.save_pretrained = functools.partial(peft_model.save_pretrained, save_embedding_layers=False)

    trainer.lambda_mse_target = cfg.lambda_mse_target
    trainer.tau_mse_target = cfg.tau_mse_target
    trainer.lambda_mse = cfg.lambda_mse
    trainer.lambda_pbm = cfg.lambda_pbm
    trainer.loop_span_margin = cfg.loop_span_margin
    trainer.loop_span_lambda_mse = cfg.loop_span_lambda_mse
    trainer.loop_span_mode = cfg.loop_span_mode
    trainer.loop_span_tether_tokens = cfg.loop_span_tether_tokens
    trainer.loop_span_lambda_tether = cfg.loop_span_lambda_tether
    trainer.loop_span_q_mix = cfg.loop_span_q_mix
    trainer.loop_span_gamma = cfg.loop_span_gamma
    trainer.loop_span_tau = cfg.loop_span_tau
    trainer.loop_span_eos_ids = [int(x) for x in cfg.loop_span_eos_ids]
    trainer.loop_span_mu_eos = cfg.loop_span_mu_eos
    trainer.letter_hinge_weight = cfg.letter_hinge_weight
    trainer.lambda_H = cfg.lambda_H
    trainer.sft_kl_lambda = getattr(cfg, "sft_kl_lambda", 0.0)
    trainer.dpo_beta = getattr(cfg, "dpo_beta", 0.1)
    trainer.dpo_sft_weight = getattr(cfg, "dpo_sft_weight", 0.05)
    trainer.kl_replay_weight = getattr(cfg, "kl_replay_weight", 0.2)
    trainer.clip_epsilon_logits = cfg.clip_epsilon_logits
    if cfg.early_stopping_chosen_win is not None:
        trainer.add_callback(
            ThresholdStop("chosen_win", threshold=cfg.early_stopping_chosen_win, higher_is_better=True)
        )

    if cfg.disable_input_grads:
        trainer.add_callback(DisableInputGradsCallback())
    trainer.add_callback(DiskGuardCallback(cfg.output_dir, run_tag=str(cfg.output_dir)))
    if cfg.letter_eval_jsonl and Path(cfg.letter_eval_jsonl).exists():
        _cb = LetterEvalCallback(cfg.letter_eval_jsonl, tokenizer, every=cfg.letter_eval_every, max_questions=cfg.letter_eval_max_questions)
        _cb.attach(trainer)
        trainer.add_callback(_cb)
        logger.info("letter validation callback: %d questions, every %d steps", len(_cb.groups), cfg.letter_eval_every)
    resume_ckpt = None
    if cfg.resume:
        if Path(cfg.output_dir).is_dir():
            resume_ckpt = get_last_complete_checkpoint(str(cfg.output_dir))
        logger.info("resume_from_checkpoint=%s", resume_ckpt)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # Beat-ThinkingCap KL-anchor: restrict the final save to just "default" -- without this,
    # save_pretrained() writes EVERY adapter in peft_config (including our frozen "ref" anchor,
    # a full ~1.2GB duplicate of an already-saved external checkpoint) into its own subdirectory.
    _save_kwargs = {}
    if "ref" in getattr(trainer.model, "peft_config", {}):
        _save_kwargs["selected_adapters"] = ["default"]
    trainer.model.save_pretrained(cfg.output_dir, save_embedding_layers=False, **_save_kwargs)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("saved LoRA adapter to %s", cfg.output_dir)
    _save_merged_model(trainer, tokenizer, cfg)
