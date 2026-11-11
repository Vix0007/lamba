"""Disk-space guard + checkpoint completion marker (DS consult #107, item #2: "atomic
checkpointing + disk-space guard + manifest + auto-clean" -- built after a real disk-full
mid-checkpoint-save crash corrupted a checkpoint dir and required manual detection+deletion
before resume would work cleanly (2026-09-26 effreason round-2 incident)).

Scope, deliberately conservative: true atomic writes (write-to-tmp + fsync + rename) would mean
monkey-patching transformers/peft's internal save_pretrained, which is higher-risk than the
benefit justifies here. Instead: (1) refuse to even ATTEMPT a save when free disk is below a
safety margin (prevents the failure mode from triggering at all), and (2) write a lightweight
COMPLETE marker + manifest entry right after a save finishes, so resume logic can detect and skip
a corrupt/partial checkpoint automatically instead of needing a human to find and delete it.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from transformers import TrainerCallback

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path.home() / "antidoom_data" / "checkpoint_manifest.jsonl"
DEFAULT_MIN_FREE_GB = 25  # matches this project's existing watchdog2.py DISK_LOW threshold


class DiskGuardCallback(TrainerCallback):
    """Refuses an impending checkpoint save if free disk is below min_free_gb (cleanly stops
    training instead of letting a save crash mid-write and corrupt the checkpoint dir). Writes a
    COMPLETE marker + manifest entry after every checkpoint that does save successfully."""

    def __init__(self, output_dir, min_free_gb: float = DEFAULT_MIN_FREE_GB, run_tag: str = ""):
        self.output_dir = Path(output_dir)
        self.min_free_gb = min_free_gb
        self.run_tag = run_tag or self.output_dir.name

    def on_step_end(self, args, state, control, **kw):
        if not control.should_save:
            return control
        free_gb = shutil.disk_usage("/").free / 1e9
        if free_gb < self.min_free_gb:
            logger.error(
                "DiskGuardCallback: only %.1fGB free (< %.0fGB safety margin) -- REFUSING this "
                "checkpoint save and stopping training cleanly instead of risking a mid-write "
                "crash. Free disk space and resume (config resume:true will pick up the last "
                "COMPLETE checkpoint).", free_gb, self.min_free_gb,
            )
            control.should_save = False
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kw):
        ckpt_dir = self.output_dir / f"checkpoint-{state.global_step}"
        if not ckpt_dir.is_dir():
            return control
        # Beat-ThinkingCap KL-anchor: HF Trainer's PEFT-integrated auto-save writes EVERY adapter
        # in peft_config into its own subdirectory, including our frozen "ref" anchor -- a full,
        # unchanging duplicate of an already-saved external checkpoint (~1.2GB per save, wasted).
        # Delete it here: nothing is lost, the real "ref" weights live safely at their original
        # path (sft_kl_anchor_path), this training run's own checkpoints never need to resume it.
        ref_dir = ckpt_dir / "ref"
        if ref_dir.is_dir():
            import shutil
            freed = sum(f.stat().st_size for f in ref_dir.rglob("*") if f.is_file())
            shutil.rmtree(ref_dir)
            logger.info("checkpoint-%d: deleted duplicated frozen 'ref' adapter subdir (%.2fGB freed)",
                        state.global_step, freed / 1e9)
        try:
            files = {f.name: f.stat().st_size for f in ckpt_dir.iterdir() if f.is_file()}
            total_bytes = sum(files.values())
            (ckpt_dir / "COMPLETE").write_text(json.dumps({"files": files, "total_bytes": total_bytes}))
            MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(MANIFEST_PATH, "a") as f:
                f.write(json.dumps({
                    "run_tag": self.run_tag, "output_dir": str(self.output_dir),
                    "checkpoint": ckpt_dir.name, "step": state.global_step,
                    "total_bytes": total_bytes, "ts": time.time(), "status": "active",
                }) + "\n")
            logger.info("checkpoint-%d saved + marked COMPLETE (%.2fGB)",
                        state.global_step, total_bytes / 1e9)
        except Exception as e:
            logger.warning("DiskGuardCallback.on_save: could not write COMPLETE marker: %s", e)
        return control


def get_last_complete_checkpoint(output_dir):
    """Like transformers.trainer_utils.get_last_checkpoint, but skips any checkpoint-N dir that
    doesn't have a COMPLETE marker (corrupt/partial from a crashed save), falling back to the
    next-highest step instead of requiring a human to find and delete it. Checkpoints saved
    before this feature existed (no marker) are still accepted if they look structurally valid
    (has adapter_model.safetensors with non-trivial size) -- only refuses ones that are missing
    that entirely or clearly truncated."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    cands = []
    for d in output_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("checkpoint-"):
            continue
        try:
            step = int(d.name.split("-")[-1])
        except ValueError:
            continue
        cands.append((step, d))
    for step, d in sorted(cands, key=lambda x: -x[0]):
        if (d / "COMPLETE").exists():
            return str(d)
        # backward-compat: no marker (pre-existing checkpoint) -- accept if it looks intact
        am = d / "adapter_model.safetensors"
        if am.exists() and am.stat().st_size > 10_000_000:  # >10MB: a real adapter, not a partial write
            logger.info("get_last_complete_checkpoint: %s has no COMPLETE marker but looks "
                        "structurally valid (pre-dates this feature) -- accepting", d.name)
            return str(d)
        logger.warning("get_last_complete_checkpoint: skipping %s (no COMPLETE marker, and "
                       "adapter_model.safetensors missing or too small -- likely a crashed/"
                       "partial save)", d.name)
    return None
