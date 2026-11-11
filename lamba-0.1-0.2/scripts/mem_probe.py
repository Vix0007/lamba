#!/usr/bin/env python3
"""mem_probe.py — sample GPU mem, PID-tree RSS and system MemAvailable until a PID exits.

Usage: mem_probe.py <pid> <out.csv>
"""

import csv
import os
import subprocess
import sys
import time

import psutil

INTERVAL_S = 5.0
GPU_LIMIT_MIB = 43500.0
AVAIL_FLOOR_MIB = 2500.0
NVSMI_CMD = [
    "nvidia-smi",
    "--query-gpu=memory.used",
    "--format=csv,noheader,nounits",
]


def gpu_used_mib():
    """Sum of memory.used over all GPUs (MiB). Returns None if unavailable."""
    try:
        res = subprocess.run(
            NVSMI_CMD, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    total = 0.0
    seen = False
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        tok = line.split()[0].replace("MiB", "").strip()
        try:
            total += float(tok)
        except ValueError:
            return None
        seen = True
    return total if seen else None


def tree_rss_mib(pid):
    """Summed RSS (MiB) of pid + all descendants. None if pid is gone."""
    try:
        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return None
    except psutil.Error:
        procs = []
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return total / (1024.0 * 1024.0)


def is_alive(pid):
    """True only if pid exists and is not a zombie."""
    if not psutil.pid_exists(pid):
        return False
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        return True  # transient (e.g. AccessDenied) -> assume still running


def main(argv):
    if len(argv) != 3:
        print("usage: mem_probe.py <pid> <out.csv>", file=sys.stderr)
        return 2
    try:
        pid = int(argv[1])
    except ValueError:
        print("error: <pid> must be an integer", file=sys.stderr)
        return 2
    out_path = argv[2]

    if not is_alive(pid):
        print("error: pid %d is not running" % pid, file=sys.stderr)
        return 1

    fresh = (not os.path.exists(out_path)) or os.path.getsize(out_path) == 0

    peak_gpu = None
    peak_rss = None
    min_avail = None
    n_valid_gpu = 0
    n_valid_rss = 0

    try:
        with open(out_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            if fresh:
                writer.writerow(["time", "gpu_mib", "rss_mib", "mem_avail_mib"])
                fh.flush()

            while True:
                ts = time.strftime("%Y-%m-%dT%H:%M:%S")

                gpu = gpu_used_mib()
                rss = tree_rss_mib(pid)
                avail_mib = psutil.virtual_memory().available / (1024.0 * 1024.0)

                writer.writerow([
                    ts,
                    "" if gpu is None else "%.1f" % gpu,
                    "" if rss is None else "%.1f" % rss,
                    "%.1f" % avail_mib,
                ])
                fh.flush()

                if gpu is not None:
                    peak_gpu = gpu if peak_gpu is None else max(peak_gpu, gpu)
                    n_valid_gpu += 1
                if rss is not None:
                    peak_rss = rss if peak_rss is None else max(peak_rss, rss)
                    n_valid_rss += 1
                min_avail = avail_mib if min_avail is None else min(min_avail, avail_mib)

                if not is_alive(pid):
                    break

                # Sleep in slices so we stop within ~0.5 s of the PID exiting.
                deadline = time.monotonic() + INTERVAL_S
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(0.5, remaining))
    except KeyboardInterrupt:
        print("\ninterrupted; reporting partial results", file=sys.stderr)

    print("peak_gpu_mib      = %s" % ("n/a" if peak_gpu is None else "%.1f" % peak_gpu))
    print("peak_rss_mib      = %s" % ("n/a" if peak_rss is None else "%.1f" % peak_rss))
    print("min_mem_avail_mib = %s" % ("n/a" if min_avail is None else "%.1f" % min_avail))

    failed = []
    if peak_gpu is not None and peak_gpu > GPU_LIMIT_MIB:
        failed.append("peak GPU %.1f MiB > %.0f MiB" % (peak_gpu, GPU_LIMIT_MIB))
    if min_avail is not None and min_avail < AVAIL_FLOOR_MIB:
        failed.append("min MemAvailable %.1f MiB < %.0f MiB" % (min_avail, AVAIL_FLOOR_MIB))

    if failed:
        print("VERDICT: FAIL (%s)" % "; ".join(failed))
    else:
        note = ""
        if n_valid_gpu == 0:
            note = " [WARN: GPU memory never readable -> not verified]"
        elif n_valid_rss == 0:
            note = " [WARN: RSS never readable]"
        print("VERDICT: PASS%s" % note)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
