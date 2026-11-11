# Derived from https://github.com/Liquid4All/antidoom (Copyright Liquid AI, Apache-2.0).
# Modified by Vixero Labs: additional losses, dataset kinds and training options. See NOTICE.
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _coerce_paths(obj: Any) -> None:
    for f in fields(obj):
        if f.name.endswith(("_jsonl", "_dir")):
            value = getattr(obj, f.name)
            if isinstance(value, str) and value:
                setattr(obj, f.name, Path(value))


@dataclass
class GenerationConfig:
    api_base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model_name: str = ""
    tokenizer_name: str = ""
    output_jsonl: Path = Path("runs/ftpo_pairs.jsonl")
    generations_jsonl: Path = Path("runs/generations.jsonl")
    hf_dataset: str | None = "LiquidAI/antidoom-mix-v1.0"
    hf_split: str = "train"
    hf_source_field: str | None = "source"
    hf_source_values: list[str] = field(default_factory=list)
    input_jsonl: Path | None = None
    prompt_field: str = "conversations"
    max_prompts: int | None = None
    target_pairs: int | None = 10_000
    target_pair_batch_size: int = 256
    threads: int = 32
    generation_backend: str = "vllm"
    request_batch_size: int = 99999
    save_token_trace: bool = False
    token_trace_statuses: list[str] = field(default_factory=lambda: ["loop_detected"])
    vllm_dtype: str = "bfloat16"
    vllm_tensor_parallel_size: int | None = None
    vllm_max_model_len: int | None = 12000
    vllm_gpu_memory_utilization: float = 0.85
    vllm_enable_prefix_caching: bool = False
    vllm_trust_remote_code: bool = True
    vllm_kwargs: dict[str, Any] = field(default_factory=dict)
    prompt_template: str = (
        "{prompt}\n\n"
        'Think through the problem step by step, then respond with your final answer as "Answer: <your answer>".'
    )
    system_prompt: str = ""
    max_prompt_tokens: int | None = None
    max_new_tokens: int = 1000
    temperature: float = 0.1
    temperatures: list[float] | None = None
    top_p: float | None = 1.0
    top_k: int | None = 50
    min_p: float | None = 0.01
    top_logprobs: int = 20
    timeout: int = 480
    stop: list[str] = field(default_factory=list)
    repetition: dict[str, Any] = field(default_factory=dict)
    chosen_max_tokens: int = 20
    chosen_top_k: int = 50
    chosen_min_p: float | None = None
    chosen_min_decoded_chars: int = 1
    chosen_require_alnum: bool = False
    chosen_skip_raw_token_substrings: list[str] = field(default_factory=list)
    skip_decoded_shorter_than: int | None = None
    skip_eval_key_substrings: list[str] = field(
        default_factory=lambda: ["bfcl", "tau", "browse", "search"]
    )

    def __post_init__(self) -> None:
        _coerce_paths(self)


@dataclass
class TrainConfig:
    dataset_jsonl: Path | None = None
    model_name: str = ""
    output_dir: Path = Path("runs/ftpo_lora")
    merged_output_dir: Path = Path("runs/ftpo_merged")
    auto_merged_output_name: bool = True
    merge_lora: bool = True
    max_seq_length: int = 4096
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_epochs: float = 1.0
    learning_rate: float = 1e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 3407
    max_train_examples: int | None = None
    rejected_regularisation_strength: float = 0.8
    chosen_regularisation_strength: float = 0.0
    source_balance_mode: str = "off"
    min_chosen_tokens: int = 1
    filter_rejected_stop_words: bool = True
    lora_r: int = 128
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_ensure_weight_tying: bool = False
    freeze_early_layers: bool = True
    n_layers_unfrozen: int = 10
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    )
    load_in_4bit: bool = False
    bf16: bool = True
    lambda_mse_target: float = 0.05
    tau_mse_target: float = 0.5
    lambda_mse: float = 0.4
    clip_epsilon_logits: float = 2.0
    early_stopping_chosen_win: float | None = 0.85
    # HF Trainer optimizer; ROCm/AMD should set "adamw_torch" to avoid the bitsandbytes dependency.
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 2.5
    logging_steps: int = 5
    save_steps: int = 50
    save_total_limit: int = 2
    resume: bool = True
    disable_input_grads: bool = False
    init_adapter_path: str | None = None   # initialise LoRA weights from a saved adapter dir (fresh optimizer/scheduler; phase-2 training)
    sft_kl_anchor_path: str | None = None  # beat-ThinkingCap KL-anchor (DS consult #114a): load a SECOND, frozen copy of this adapter
                                            # as a "ref" PEFT adapter alongside the trainable "default" one; sft_ce loss adds
                                            # lambda*KL(p_ref||p_new) at the same response positions, keeping SFT from drifting
                                            # away from a known-good (loop-safe) adapter with no anchor holding it in place.
    sft_kl_lambda: float = 0.0             # 0.0 = no KL term (plain sft_ce, unchanged default behavior)
    lambda_pbm: float = 0.5
    lora_layer_set: list[int] | None = None      # explicit layer indices for LoRA (overrides last-k); e.g. all full-attention layers + last GDN layers
    lora_rank_pattern: dict[str, int] | None = None   # per-module-name rank (PEFT rank_pattern), alpha = rank
    lora_exclude_modules: list[str] = field(default_factory=list)  # target module names to drop (e.g. in_proj_a, in_proj_b)
    loop_span_tokens: int = 32       # K supervised tokens of the 2nd occurrence of a repeated unit
    loop_span_margin: float = 1.0
    loop_span_lambda_mse: float = 1.0
    loop_span_mode: str = "hinge"        # 'hinge' (v5) | 'klmask' (v6: KL to base with the copied token masked+renormalised)
    loop_span_tether_tokens: int = 24     # klmask: KL(base||adapter) tether on the last T tokens of the FIRST occurrence
    loop_span_lambda_tether: float = 1.0
    loop_span_q_mix: float = 0.5          # klmask: q = mix*p_base + (1-mix)*softmax((log p_base - gamma*1[y])/tau); mix=0 -> pure mask+renorm
    loop_span_gamma: float = 3.0
    loop_span_tau: float = 1.3
    loop_span_eos_ids: list[int] = field(default_factory=lambda: [248046])   # end-of-turn ids guarded against mass growth (checked below)
    loop_span_mu_eos: float = 2.0
    letter_hinge_weight: float = 1.0
    lambda_H: float = 0.1
    letter_eval_jsonl: str = ""
    letter_eval_every: int = 25
    letter_eval_max_questions: int = 200
    # Ruminconfab nudge-guard DPO retrain (DS #168-180): kind='hard'/'soft' (DPO pairs, via
    # DPOPairDataset) and kind='sequence' (KL-replay, via KLReplayDataset) rows, mixed into
    # the SAME cfg.dataset_jsonl file per this codebase's established convention (one file,
    # split by row 'kind' -- see letter/letter_group/loop_span/ftpo_soft/sft_ce for
    # precedent). letter-bias KL-replay rows reuse kind='letter_group' unchanged, already
    # routed by the existing LetterGroupDataset -- no new field needed for that.
    dpo_beta: float = 0.1
    dpo_sft_weight: float = 0.05    # DS #177 sec.3: CE on the chosen response, folded into the dpo_pair loss
    kl_replay_weight: float = 0.2   # DS #177 sec.2 / #180: D_KL(p_ref||p_theta), ref-weighted, matches sft_ce's established anchor direction
    # DS #184 (correcting #168's original spec): DPO's reference must be the INITIAL POLICY
    # (standard DPO), a SEPARATE frozen adapter from the KL-replay anchor ("ref" = c10 above).
    # When set, loads a second frozen "dpo_ref" adapter -- _dpo_pair_loss uses this one,
    # _kl_replay_loss keeps using "ref" (c10). Opt-in, no effect on any config that never sets it.
    dpo_ref_adapter_path: str | None = None

    def __post_init__(self) -> None:
        _coerce_paths(self)


@dataclass
class AppConfig:
    model_name: str | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: Path | None) -> AppConfig:
    if path is None:
        return AppConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    unknown = set(data) - {"model_name", "generation", "train"}
    if unknown:
        raise ValueError(f"Unknown top-level config key(s): {sorted(unknown)}")
    for name in ("generation", "train"):
        section = data.get(name)
        if section is not None and not isinstance(section, dict):
            raise ValueError(f"config section {name!r} must be a mapping")
    model_name = data.get("model_name")
    return AppConfig(
        model_name=str(model_name) if model_name is not None else None,
        generation=GenerationConfig(**(data.get("generation") or {})),
        train=TrainConfig(**(data.get("train") or {})),
    )
