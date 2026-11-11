#!/usr/bin/env python3
"""Malay->Indonesian drift rate on Malay answers (fastText ms-id, max-sentence P(indonesian)) + regex markers.
usage: ms_drift_eval.py FILE.jsonl [FILE2 ...]   (identity eval files: rows with lang=='ms', field 'content')"""
import json, re, subprocess, sys
IDP = re.compile(r"\b(nggak|gak|enggak|banget|udah|gimana|aja|kok|lho|sih|asisten|namaaku|bisa|dong|nih|kayak|gitu|silakan|dikembangkan|tim|bahasa indonesia)\b", re.I)
for fn in sys.argv[1:]:
    T = [(r.get("content") or "").strip()[:3000] for r in map(json.loads, open(fn)) if r.get("lang") == "ms"]
    T = [t for t in T if t]
    S = json.loads(subprocess.run(["/home/ubuntu/venv-ft312/bin/python", "/home/ubuntu/antidoom_data/ms_id_score.py"], input=json.dumps(T), capture_output=True, text=True, check=True).stdout)
    n = len(T); hi = sum(s >= 0.9 for s in S); mid = sum(0.15 <= s < 0.9 for s in S); rx = sum(bool(IDP.search(t)) for t in T)
    print(f"{fn}: ms answers {n} | indonesian(>=0.9) {hi} ({100*hi/max(n,1):.1f}%) | mixed(0.15-0.9) {mid} ({100*mid/max(n,1):.1f}%) | regex markers {rx} ({100*rx/max(n,1):.1f}%)")
