"""Build tool-decision DPO pairs from mined files (l03_mine_native output). usage: td_build_pairs.py <mined1.jsonl> [<mined2.jsonl> ...]
Policy label (Article I, web constitution): for SET A (obscure/recent/local/costly-if-wrong) chosen = first-turn completion that CALLS web_search, rejected = completion that answers directly / only announces.
SET B (easy): chosen = direct answer, rejected = needless search. Lock-in: chosen = real tool call, rejected = announce/direct.
Quality gates: both samples complete (finish=stop, </think> present, not truncated); chosen = web_search call whose query overlaps the prompt, no leak words; rejected not an honest 'I'm not sure' (ambiguous label); 1 pair per state."""
import json, sys, re, hashlib, collections, random, os
sys.path.insert(0,'/home/ubuntu/antidoom_data')
import importlib.util
spec=importlib.util.spec_from_file_location("TE","/home/ubuntu/antidoom_data/l03_tool_eval.py"); TE=importlib.util.module_from_spec(spec); spec.loader.exec_module(TE)
LEAK=re.compile(r"system prompt|my instructions|constitution|my guidelines|CLAUDE\.md",re.I)
UNSURE=re.compile(r"tidak (?:pasti|tahu|yakin)|tak (?:pasti|tahu|yakin)|kurang yakin|not sure|don'?t know|i'?m unsure|belum pernah dengar|never heard of|nggak (?:tahu|yakin)|gak (?:tahu|yakin)|tidak familiar|not familiar",re.I)
STOP=set("what who that this with from have does about tahu apakah siapa yang untuk dengan atau juga adalah itu ini the and tau gak nggak kamu aku saya dari ada".split())
def good(s): return s["finish"]=="stop" and s["has_think_close"] and not s["truncated"]
def words(t): return {w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]{3,}",t.lower()) if w not in STOP}
def query_text(s):
    m=re.search(r"<tool_call>(.*?)(?:</tool_call>|$)",s["text"],re.S); return m.group(1) if m else ""
def overlap_ok(prompt,s):
    q=query_text(s).lower(); pw=words(prompt); return any(w in q for w in pw) if pw else True
def user_prompt(st):
    u=[m for m in st["messages"] if m["role"]=="user"]; return u[0]["content"] if u else ""
pairs=[]; stats=collections.Counter(); audit=[]; replay=[]
for path in sys.argv[1:]:
    for l in open(path):
        r=json.loads(l)
        if r.get("error"): stats["state_error"]+=1; continue
        meta=r["meta"]; st={"messages":None}; samples=r["samples"]
        # recover messages from rendered context? mined rows carry meta only -> look up by state_id in the states files
        st_id=r["state_id"]; STATE=STATES.get(st_id) if 'STATES' in globals() else None
        pairs.append((r,meta))
# load states for messages
STATES={}
for f in ["/home/ubuntu/antidoom_data/td/td_states_1810.jsonl", "/home/ubuntu/eval_results/id1810q/td_states_s3.jsonl", "/home/ubuntu/eval_results/id1810q/td_states_q4.jsonl"]:
    for l in open(f): s=json.loads(l); STATES[s["state_id"]]=s
out=[]
for r,meta in pairs:
    st=STATES.get(r["state_id"])
    if not st: stats["no_state"]+=1; continue
    S=[s for s in r["samples"] if good(s)]; stats["states"]+=1; stats["samples_good"]+=len(S); stats["samples"]+=len(r["samples"])
    called=[s for s in S if s["has_tool_call"] and "web_search" in (s.get("tool_call_names") or [])]
    other=[s for s in S if not s["has_tool_call"]]
    ann=[s for s in other if TE.matches_announce(s["answer"])]; direct=[s for s in other if s not in ann]
    prompt=user_prompt(st); e=meta.get("expect","search"); kind=meta.get("kind") or meta.get("src") or "first"
    key=r["state_id"]; rnd=random.Random(int(hashlib.md5(key.encode()).hexdigest()[:8],16))
    def row(cho,rej,tag):
        return {"kind":"hard","weight":1.0,"task_id":f"td|{key}","rep":0,"context_messages":st["messages"],"rendered_context":r["rendered_context"],"chosen":cho["text"],"rejected":rej["text"],"src":"tooldecision","td_tag":tag,"lang":meta["lang"],"cond":meta["cond"],"kind_td":kind}
    if e=="search":
        cc=[s for s in called if overlap_ok(prompt,s) and not LEAK.search(s["reasoning"]+" "+s["answer"])]
        rr=[s for s in (direct+ann) if not UNSURE.search(s["answer"]) and not LEAK.search(s["answer"])]
        stats[f"A_{kind}_states"]+=1
        if not S: stats["no_good"]+=1; continue
        if not called: stats[f"A_{kind}_never_called"]+=1; continue
        if not (direct+ann): stats[f"A_{kind}_always_called"]+=1; continue
        if not cc: stats["A_no_clean_chosen"]+=1; continue
        if not rr: stats["A_only_unsure_rejected"]+=1; continue
        p=row(rnd.choice(cc),rnd.choice(rr),f"A_{kind}"); out.append(p); stats[f"pair_A_{kind}"]+=1
    else:
        stats["B_states"]+=1
        cc=[s for s in direct if len(s["answer"].strip())>=20 and not LEAK.search(s["answer"])]
        rr=[s for s in called]
        if cc and rr: out.append(row(rnd.choice(cc),rnd.choice(rr),"B")); stats["pair_B"]+=1
        elif cc and not rr:
            stats["B_always_direct(replay candidate)"]+=1
            replay.append({"kind":"sequence","domain":"td_direct_replay","source_id":key,"rendered_context":r["rendered_context"],"response_text":rnd.choice(cc)["text"]})
        else: stats["B_other"]+=1
os.makedirs("/home/ubuntu/antidoom_data/td",exist_ok=True)
with open("/home/ubuntu/antidoom_data/td/td_pairs_1810.jsonl","w") as f:
    for p in out: f.write(json.dumps(p,ensure_ascii=False)+"\n")
random.Random(9).shuffle(replay)
with open("/home/ubuntu/antidoom_data/td/td_replay_1810.jsonl","w") as f:
    for x in replay[:120]: f.write(json.dumps(x,ensure_ascii=False)+"\n")
rnd=random.Random(3); aud=rnd.sample(out,min(60,len(out)))
with open("/home/ubuntu/antidoom_data/td/td_pairs_1810_AUDIT.md","w") as f:
    f.write("# tool-decision pairs: random audit sample (read ALL; precision gate >=90%)\n")
    for p in aud:
        f.write(f"\n## {p['task_id']} [{p['td_tag']} {p['lang']} {p['cond']}]\nPROMPT: {p['context_messages'][-1]['content'][:200]}\n--- CHOSEN (tail): {p['chosen'][-420:]!r}\n--- REJECTED (tail): {p['rejected'][-420:]!r}\n")
print(json.dumps(dict(stats),indent=1)); print("pairs",len(out))
