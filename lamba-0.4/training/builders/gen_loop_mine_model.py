import json,sys,os,urllib.request
from concurrent.futures import ThreadPoolExecutor
items=[json.loads(l) for l in open(sys.argv[1])]; out=sys.argv[2]; K=int(sys.argv[3])
done=set()
if os.path.exists(out): done={json.loads(l)['pilot_id'] for l in open(out)}
todo=[r for r in items if r.get('pilot_id') or r.get('id')]
todo=[r for r in todo if (r.get('pilot_id') or r.get('id')) not in done]
def one(r):
    pid=r.get('pilot_id') or r.get('id')
    body=json.dumps({"model":sys.argv[4],"messages":[{"role":"user","content":r['prompt']}],"n":K,"temperature":0.7,"top_p":0.95,"max_tokens":4000,"repetition_penalty":1.0}).encode()
    q=urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    try: d=json.loads(urllib.request.urlopen(q,timeout=3600).read())
    except Exception as e: return None
    cs=[{"reasoning":c['message'].get('reasoning') or c['message'].get('reasoning_content') or '',"content":c['message'].get('content') or '',"finish":c['finish_reason']} for c in d['choices']]
    return {"pilot_id":pid,"domain":r.get('domain'),"prompt":r['prompt'],"comps":cs}
with ThreadPoolExecutor(12) as ex, open(out,'a') as f:
    for res in ex.map(one,todo):
        if res: f.write(json.dumps(res,ensure_ascii=False)+"\n"); f.flush()
print("LOOPMINE_DONE")
