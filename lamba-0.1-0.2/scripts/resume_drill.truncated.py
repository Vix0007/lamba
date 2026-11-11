#!/usr/bin/env python3
"""resume_drill.py - kill a LoRA training run mid-flight, relaunch it, verify HF Trainer resumed correctly.

Launches:  uv run antidoom -c <yaml> -r <run_root> --model-name <model> train --dataset-jsonl <file>
Waits for  <run_root>/lora/checkpoint-2 to be fully written, SIGKILLs the process tree
(parent + children, explicit PIDs via psutil - never pkill -f), relaunches the same
command, then verifies resume marker, final global_step, LR-schedule continuity and
loss continuity across the restart. Exits non-zero on any FAIL.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover
    print("FATAL: psutil is required (uv add psutil / pip install psutil)", file=sys.stderr)
    raise SystemExit(2)

RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    ok = bool(ok)
    RESULTS.append((label, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok


# --------------------------------------------------------------------------- process

def build_cmd(a: argparse.Namespace) -> list[str]:
    return [
        a.uv, "run", "antidoom",
        "-c", str(a.yaml),
        "-r", str(a.run_root),
        "--model-name", str(a.model_name),
        "train",
        "--dataset-jsonl", str(a.dataset_jsonl),
    ]


def launch(a: argparse.Namespace, tag: str, log_dir: Path):
    cmd = build_cmd(a)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_p = log_dir / f"{tag}.stdout.log"
    err_p = log_dir / f"{tag}.stderr.log"
    print(f"[launch:{tag}] {' '.join(cmd)}\n             stdout={out_p}\n             stderr={err_p}", flush=True)
    fo = open(out_p, "wb")
    fe = open(err_p, "wb")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=fo,
            stderr=fe,
            stdin=subprocess.DEVNULL,
            cwd=str(a.cwd) if a.cwd else None,
            start_new_session=True,
        )
    except Exception:
        fo.close()
        fe.close()
        raise
    return proc, fo, fe, out_p, err_p


def kill_tree(pid: int, grace: float = 20.0):
    """SIGKILL every child (deepest first) then the root, by explicit PID."""
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True, "process already gone", []
    procs = list(reversed(root.children(recursive=True))) + [root]
    pids = [p.pid for p in procs]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            continue
        except Exception as e:  # AccessDenied etc.
            print(f"    WARN: SIGKILL pid {p.pid} failed: {e}", flush=True)
    _gone, alive = psutil.wait_procs(procs, timeout=grace)
    if alive:
        return False, f"{len(alive)} process(es) survived SIGKILL: {[p.pid for p in alive]}", pids
    return True, f"SIGKILLed {len(pids)} process(es): {pids}", pids


def gpu_pids() -> set[int]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=25,
        )
    except Exception:
        return set()
    return {int(tok) for tok in r.stdout.split() if tok.strip().isdigit()}


# --------------------------------------------------------------------------- artifacts

def wait_for_checkpoint(ckpt: Path, timeout: float, poll: float, proc=None):
    """Block until <ckpt>/trainer_state.json parses twice identically and the adapter exists."""
    state = ckpt / "trainer_state.json"
    adapter = ckpt / "adapter_model.safetensors"
    deadline = time.monotonic() + timeout
    note = "checkpoint dir / files not present yet"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return None, f"trainer exited early (rc={proc.returncode}) before {ckpt.name} was complete"
        if state.exists() and adapter.exists():
            try:
                first = json.loads(state.read_text())
                time.sleep(0.75)
                second = json.loads(state.read_text())
                if first and first == second:
                    return second, "ok"
                note = "trainer_state.json still changing between two reads"
            except Exception as e:
                note = f"trainer_state.json not parseable yet ({type(e).__name__})"
        time.sleep(poll)
    return None, f"timeout after {timeout:.0f}s ({note})"


def collect_states(outdir: Path):
    cands = []
    root_state = outdir / "trainer_state.json"
    if root_state.exists():
        cands.append(root_state)
    for d in sorted(outdir.glob("checkpoint-*")):
        q = d / "trainer_state.json"
        if q.exists():
            cands.append(q)
    states = []
    for q in cands:
        try:
            s = json.loads(q.read_text())
        except Exception:
            continue
        try:
            gs = int(s.get("global_step") or 0)
        except (TypeError, ValueError):
            gs = 0
        states.append((gs, q, s))
    return states


def series(hist, key: str):
    """Sorted, step-deduplicated [(step, finite float)] for `key` in log_history."""
    pts: dict[int, float] = {}
    for e in hist or []:
        if not isinstance(e, dict):
            continue
        v, s = e.get(key), e.get("step")
        if v is None or s is None:
            continue
        try:
            v = float(v)
            s = int(s)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        pts[s] = v
    return sorted(pts.items())


# --------------------------------------------------------------------------- logs

def discover_logs(run_root: Path, log_dir: Path, extra):
    out, seen = [], set()
    cands = []
    for base in (Path(run_root), Path(log_dir)):
        try:
            if base.exists():
                cands += list(base.rglob("*.log")) + list(base.rglob("*.txt"))
        except OSError:
            pass
    cands += [Path(e) for e in extra]
    for p in cands:
        try:
            if not p.is_file():
                continue
            rp = str(p.resolve())
        except OSError:
            continue
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
        if len(out) >= 200:
            break
    return out


def read_text(p: Path, max_bytes: int = 4_000_000) -> str:
    try:
        with open(p, "rb") as f:
            return f.read(max_bytes).decode("utf-8", "replace")
    except Exception:
        return ""


def read_new(p: Path, offset: int, max_bytes: int = 4_000_000) -> str:
    try:
        with open(p, "rb") as f:
            f.seek(offset)
            return f.read(max_bytes).decode("utf-8", "replace")
    except Exception:
        return ""


def find_resume_marker(text: str, ckpt_name: str):
    if "resume_from_checkpoint" not in text:
        return False, "no 'resume_from_checkpoint' string in the second run's new log output"
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if "resume_from_checkpoint" in ln:
            window = lines[max(0, i - 2): i + 3]
            wtext = "\n".join(window)
            if ckpt_name in wtext:
                return True, " | ".join(x.strip() for x in window if x.strip())[:400]
    marker = text.find("resume_from_checkpoint")
    near = [m.start() for m in re.finditer(re.escape(ckpt_name), text) if abs(m.start() - marker) < 1500]
    if near:
        start = max(0, min(marker, min(near)) - 200)
        return True, text[start:start + 400].replace("\n", " | ")
    return False, f"found 'resume_from_checkpoint' but no '{ckpt_name}' within 1500 chars of it"


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Kill a LoRA training run at checkpoint-N, relaunch, verify HF resume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--yaml", required=True, help="config passed to -c")
    ap.add_argument("--run-root", required=True, help="passed to -r (contains lora/)")
    ap.add_argument("--model-name", required=True, help="passed to --model-name")
    ap.add_argument("--dataset-jsonl", required=True, help="passed to --dataset-jsonl")
    ap.add_argument("--total-steps", type=int, required=True, help="expected final global_step")
    ap.add_argument("--checkpoint-step", type=int, default=2, help="kill once checkpoint-<N> is complete")
    ap.add_argument("--wait-timeout", type=float, default=1800.0, help="seconds to wait for checkpoint-N")
    ap.add_argument("--run-timeout", type=float, default=7200.0, help="seconds to wait for run2 to exit")
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--uv", default="uv")
    ap.add_argument("--log-dir", default=None, help="where to write drill stdio logs (default <run_root>/resume_drill_logs)")
    ap.add_argument("--extra-log", action="append", default=[], help="extra log file(s) to scan (repeatable)")
    ap.add_argument("--cwd", default=None)
    a = ap.parse_args()

    run_root = Path(a.run_root).expanduser()
    outdir = run_root / "lora"
    log_dir = Path(a.log_dir).expanduser() if a.log_dir else run_root / "resume_drill_logs"
    ckpt_dir = outdir / f"checkpoint-{a.checkpoint_step}"
    ckpt_name = ckpt_dir.name

    print("=" * 78)
    print(f"resume drill | kill at {ckpt_dir} | expect final global_step={a.total_steps}")
    print("=" * 78, flush=True)

    p1 = p2 = None
    fo1 = fe1 = fo2 = fe2 = None
    try:
        # ---------------- run 1 ----------------
        p1, fo1, fe1, _o1, _e1 = launch(a, "run1", log_dir)
        state1, note = wait_for_checkpoint(ckpt_dir, a.wait_timeout, a.poll, p1)
        if state1 is None:
            ok, det, pids = kill_tree(p1.pid)
            print(f"[kill] {det}", flush=True)
            check(f"setup: run1 reached a complete {ckpt_name}", False, note)
            return summarize()
        hist1 = state1.get("log_history") or []
        try:
            step1 = int(state1.get("global_step") or a.checkpoint_step)
        except (TypeError, ValueError):
            step1 = a.checkpoint_step
        print(f"[run1] {ckpt_name} complete: global_step={step1}, log_history={len(hist1)} entries", flush=True)
        check(f"setup: run1 reached a complete {ckpt_name}", True,
              f"global_step={step1}, log_history={len(hist1)}")

        # ---------------- kill ----------------
        ok_kill, det_kill, killed_pids = kill_tree(p1.pid)
        for f in (fo1, fe1):
            try:
                f.close()
            except Exception:
                pass
        fo1 = fe1 = None
        print(f"[kill] {det_kill}", flush=True)
        check("setup: run1 process tree SIGKILLed (explicit PIDs, no pkill)", ok_kill, det_kill)
        try:
            leftover = gpu_pids() & set(killed_pids)
            if leftover:
                print(f"[warn] pids {sorted(leftover)} still appear to hold GPU memory after SIGKILL", flush=True)
        except Exception:
            pass

        # ---------------- snapshot logs, run 2 ----------------
        pre_logs = discover_logs(run_root, log_dir, a.extra_log)
        offsets = {}
        for p in pre_logs:
            try:
                offsets[str(p.resolve())] = p.stat().st_size
            except OSError:
                offsets[str(p.resolve())] = 0
        offsets[str(Path(a.extra_log[0]).resolve())] = offsets.get(str(Path(a.extra_log[0]).resolve()), 0) if a.extra_log else 0

        p2, fo2, fe2, out2, err2 = launch(a, "run2",