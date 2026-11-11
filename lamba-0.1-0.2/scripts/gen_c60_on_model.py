#!/usr/bin/env python3
"""Run the 60 round-2 C prompts (rumination/persona-confab) on another served model with the
IDENTICAL protocol used for the c10 round-2 run (gen_ruminconfab_rollouts.py: temp 0.7, top_p 0.9,
per-id fixed seed, 8 rounds, 4096 tok/round, same system prompt). Usage: gen_c60_on_model.py <served_model> <out.jsonl>"""
import sys
from pathlib import Path
sys.path.insert(0, "/home/ubuntu/antidoom_data")
import gen_ruminconfab_rollouts as g
g.MODEL_NAME = sys.argv[1]
g.OUT = Path(sys.argv[2])
g.PROMPTS = Path("/home/ubuntu/antidoom_data/ruminconfab_c60_prompts.jsonl")
g.CONCURRENCY = 8
g.main()
