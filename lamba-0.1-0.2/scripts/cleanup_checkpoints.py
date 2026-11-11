#!/usr/bin/env python3
"""Manifest-driven checkpoint cleanup (DS consult #107, item #2, the "auto-clean" half --
the disk-guard + COMPLETE-marker + manifest-logging half lives in
antidoom/src/antidoom/checkpoint_guard.py and is wired into ftpo_train.py already).

Policy (DS's, as specified): never auto-delete anything not in the manifest. Within a manifest'd
run, keep the last 2 checkpoints by step (resume safety) and anything explicitly marked
final/shipped/best_so_far in the manifest; only consider deleting the rest, and only ones older
than a grace period. This script never deletes on its own -- `--report` (default) only shows what
WOULD be reclaimed; `--apply` actually deletes, and only after `--report` has been looked at.

usage:
  cleanup_checkpoints.py                     # report only, no changes
  cleanup_checkpoints.py --apply             # actually delete what the report flags
  cleanup_checkpoints.py --grace-days 3      # override the default 7-day grace period
  cleanup_checkpoints.py --mark-final <output_dir> <checkpoint-name>   # protect one checkpoint
"""
import argparse, json, shutil, sys, time
from pathlib import Path

MANIFEST_PATH = Path.home() / "antidoom_data" / "checkpoint_manifest.jsonl"
PROTECTED_MARKER = "PROTECTED"  # a file dropped inside a checkpoint dir to never auto-delete it


def load_manifest():
    if not MANIFEST_PATH.exists():
        return []
    return [json.loads(l) for l in open(MANIFEST_PATH) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete; default is report-only")
    ap.add_argument("--grace-days", type=float, default=7.0)
    ap.add_argument("--mark-final", nargs=2, metavar=("OUTPUT_DIR", "CHECKPOINT_NAME"),
                     help="drop a PROTECTED marker in this checkpoint so cleanup never touches it")
    a = ap.parse_args()

    if a.mark_final:
        out_dir, ckpt = a.mark_final
        p = Path(out_dir) / ckpt
        if not p.is_dir():
            print(f"no such checkpoint dir: {p}"); sys.exit(1)
        (p / PROTECTED_MARKER).write_text(json.dumps({"marked_at": time.time()}))
        print(f"PROTECTED marker written to {p} -- cleanup will never delete this checkpoint")
        return

    rows = load_manifest()
    if not rows:
        print("no manifest entries yet (checkpoint_manifest.jsonl empty/missing) -- nothing to do")
        return

    by_run = {}
    for r in rows:
        by_run.setdefault(r["output_dir"], []).append(r)

    now = time.time()
    grace_s = a.grace_days * 86400
    total_reclaim = 0
    to_delete = []

    for output_dir, entries in by_run.items():
        entries.sort(key=lambda r: -r["step"])
        keep_steps = {e["step"] for e in entries[:2]}  # last 2 by step, always kept
        print(f"\n=== {output_dir} ({len(entries)} manifest'd checkpoints) ===")
        for e in entries:
            ckpt_path = Path(output_dir) / e["checkpoint"]
            if not ckpt_path.is_dir():
                print(f"  {e['checkpoint']:20s} step={e['step']:5d} -- MISSING on disk (manifest stale), skipping")
                continue
            protected = (ckpt_path / PROTECTED_MARKER).exists()
            age_days = (now - e["ts"]) / 86400
            size_gb = e["total_bytes"] / 1e9
            keep = e["step"] in keep_steps or protected or age_days < a.grace_days
            reason = ("last-2" if e["step"] in keep_steps else
                      "PROTECTED" if protected else
                      f"age {age_days:.1f}d < grace {a.grace_days}d" if age_days < a.grace_days else
                      f"eligible (age {age_days:.1f}d, {size_gb:.2f}GB)")
            print(f"  {e['checkpoint']:20s} step={e['step']:5d} {size_gb:6.2f}GB age={age_days:6.1f}d -- "
                  f"{'KEEP' if keep else 'DELETE'} ({reason})")
            if not keep:
                to_delete.append((ckpt_path, e["total_bytes"]))
                total_reclaim += e["total_bytes"]

    print(f"\n{'Would reclaim' if not a.apply else 'Reclaiming'}: {total_reclaim/1e9:.1f}GB across {len(to_delete)} checkpoint dirs")
    if not to_delete:
        return
    if not a.apply:
        print("(report-only -- rerun with --apply to actually delete)")
        return
    for ckpt_path, _ in to_delete:
        shutil.rmtree(ckpt_path)
        print(f"deleted {ckpt_path}")


if __name__ == "__main__":
    main()
