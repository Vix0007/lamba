import os, json, re, requests
import numpy as np

API_KEY = os.environ["COHERE_API_KEY"]
URL = "https://api.cohere.com/v2/embed"

def embed(texts, input_type="clustering", batch_size=96):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = requests.post(URL, headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }, json={"model": "embed-v4.0", "input_type": input_type, "texts": batch, "output_dimension": 512})
        resp.raise_for_status()
        all_vecs.extend(resp.json()["embeddings"]["float"])
    return np.array(all_vecs)

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if len(p.strip()) > 15]

data = [json.loads(l) for l in open('/home/ubuntu/antidoom_data/v5/loop_extracted.jsonl')]
sample = [d for d in data if d["rollout_id"] == "sci_phy_0134-03"][0]

sents = split_sentences(sample["rollout_text"])
print(f"Split into {len(sents)} sentences")

vecs = embed(sents)
print(f"Embedded shape: {vecs.shape}")

norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
sim = norm @ norm.T

threshold = 0.85
pairs = []
for i in range(len(sents)):
    for j in range(i+3, len(sents)):
        if sim[i][j] > threshold:
            pairs.append((i, j, sim[i][j]))

pairs.sort(key=lambda x: -x[2])
print(f"\nFound {len(pairs)} near-duplicate sentence pairs (sim>{threshold}, >=3 sentences apart)")
for i, j, s in pairs[:6]:
    print(f"\n  sim={s:.3f}  sent[{i}]: {sents[i][:150]}")
    print(f"            sent[{j}]: {sents[j][:150]}")
