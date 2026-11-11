import json, re, time
import torch
from transformers import AutoTokenizer, AutoModel

DATA_PATH = "/home/ubuntu/antidoom_data/v5/loop_extracted.jsonl"
OUT_PATH = "/home/ubuntu/antidoom_data/v5/semantic_loop_results.jsonl"
SIM_THRESHOLD = 0.88
MIN_SENTENCE_GAP = 3  # skip near-adjacent sentences (normal flow)

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if len(p.strip()) > 15]

print("Loading BGE-M3...")
tok = AutoTokenizer.from_pretrained('BAAI/bge-m3')
model = AutoModel.from_pretrained('BAAI/bge-m3', dtype=torch.float16).cuda().eval()

@torch.no_grad()
def embed_batch(texts, batch_size=256):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tok(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to('cuda')
        out = model(**inputs)
        emb = out.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, dim=-1)
        all_vecs.append(emb.float().cpu())
    return torch.cat(all_vecs, dim=0)

data = [json.loads(l) for l in open(DATA_PATH)]
not_detected = [d for d in data if not d["detected_span"]]
print(f"Processing {len(not_detected)} not-detected rollouts")

t0 = time.time()
results = []
for idx, d in enumerate(not_detected):
    sents = split_sentences(d["rollout_text"])
    if len(sents) < MIN_SENTENCE_GAP + 2:
        results.append({**{k: v for k, v in d.items() if k != "rollout_text"}, "semantic_loop_found": False, "num_sentences": len(sents)})
        continue
    vecs = embed_batch(sents)
    sim = (vecs @ vecs.T).numpy()
    import numpy as np
    n = len(sents)
    mask = np.triu(np.ones((n, n), dtype=bool), k=MIN_SENTENCE_GAP)
    masked_sim = np.where(mask, sim, -1.0)
    best_sim = masked_sim.max()
    best_pair = None
    if best_sim > SIM_THRESHOLD:
        i, j = np.unravel_index(masked_sim.argmax(), masked_sim.shape)
        best_pair = (int(i), int(j))
    result = {k: v for k, v in d.items() if k != "rollout_text"}
    result["num_sentences"] = len(sents)
    if best_pair:
        i, j = best_pair
        result["semantic_loop_found"] = True
        result["semantic_sim"] = best_sim
        result["semantic_sent_i"] = sents[i]
        result["semantic_sent_j"] = sents[j]
        result["semantic_gap"] = j - i
    else:
        result["semantic_loop_found"] = False
    results.append(result)
    if (idx + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        eta = (len(not_detected) - idx - 1) / rate
        print(f"  {idx+1}/{len(not_detected)} done, {rate:.1f}/s, ETA {eta/60:.1f} min")

with open(OUT_PATH, "w") as f:
    for r in results:
        f.write(json.dumps(r, default=str) + "\n")

found_ct = sum(1 for r in results if r.get("semantic_loop_found"))
print(f"\nDone in {(time.time()-t0)/60:.1f} min")
print(f"Semantic loop found in {found_ct}/{len(results)} ({found_ct/len(results):.1%})")
