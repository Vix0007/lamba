"""
Generalized, resumable/crash-safe semantic loop detector.
Reads <source>/not_detected.jsonl, embeds sentences via BGE-M3, flags within-rollout
near-duplicate sentence pairs (paraphrased/interspersed loops that exact-match misses),
writes to <source>/semantic_loop_results.jsonl.

Resumable: on restart, skips rollout_ids already present in the output file and
appends new results. Flushes after every item so a hard kill loses at most one item.
"""
import json
import os
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

SIM_THRESHOLD = 0.88
MIN_SENTENCE_GAP = 3
BATCH_EMBED_SIZE = 256

SOURCES = {
    "v3": {
        "in_path": "/home/ubuntu/antidoom_data/v3/not_detected.jsonl",
        "out_path": "/home/ubuntu/antidoom_data/v3/semantic_loop_results.jsonl",
    },
    "v4": {
        "in_path": "/home/ubuntu/antidoom_data/v4/not_detected.jsonl",
        "out_path": "/home/ubuntu/antidoom_data/v4/semantic_loop_results.jsonl",
    },
}


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 15]


def load_done_ids(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["rollout_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main(source_name):
    cfg = SOURCES[source_name]
    print("Loading BGE-M3...", flush=True)
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = AutoModel.from_pretrained("BAAI/bge-m3", dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def embed_batch(texts, batch_size=BATCH_EMBED_SIZE):
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
            out = model(**inputs)
            emb = out.last_hidden_state[:, 0]
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_vecs.append(emb.float().cpu())
        return torch.cat(all_vecs, dim=0)

    data = [json.loads(l) for l in open(cfg["in_path"])]
    done_ids = load_done_ids(cfg["out_path"])
    remaining = [d for d in data if d["rollout_id"] not in done_ids]
    print(f"[{source_name}] {len(data)} total, {len(done_ids)} already done, {len(remaining)} remaining", flush=True)

    out_f = open(cfg["out_path"], "a")
    t0 = time.time()
    found_ct = 0
    for idx, d in enumerate(remaining):
        sents = split_sentences(d["rollout_text"])
        result = {"rollout_id": d["rollout_id"], "pilot_id": d["pilot_id"], "domain": d["domain"],
                   "num_sentences": len(sents)}
        if len(sents) < MIN_SENTENCE_GAP + 2:
            result["semantic_loop_found"] = False
        else:
            vecs = embed_batch(sents)
            sim = (vecs @ vecs.T).numpy()
            n = len(sents)
            mask = np.triu(np.ones((n, n), dtype=bool), k=MIN_SENTENCE_GAP)
            masked_sim = np.where(mask, sim, -1.0)
            best_sim = masked_sim.max()
            if best_sim > SIM_THRESHOLD:
                i, j = np.unravel_index(masked_sim.argmax(), masked_sim.shape)
                result["semantic_loop_found"] = True
                result["semantic_sim"] = float(best_sim)
                result["semantic_sent_i"] = sents[int(i)]
                result["semantic_sent_j"] = sents[int(j)]
                result["semantic_gap"] = int(j - i)
                found_ct += 1
            else:
                result["semantic_loop_found"] = False
        out_f.write(json.dumps(result, default=str) + "\n")
        out_f.flush()
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(remaining) - idx - 1) / rate
            print(f"  [{source_name}] {idx+1}/{len(remaining)} done, {rate:.1f}/s, ETA {eta/60:.1f} min, "
                  f"{found_ct} semantic hits so far", flush=True)

    out_f.close()
    print(f"[{source_name}] Done in {(time.time()-t0)/60:.1f} min. "
          f"Semantic loop found in {found_ct}/{len(remaining)} this run.", flush=True)


if __name__ == "__main__":
    for src in sys.argv[1:] or ["v3", "v4"]:
        main(src)
