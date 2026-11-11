#!/usr/bin/env python3
"""0.5 diagnosis battery: (A) other-model names, 3 conditions; (B) self-facts from system prompt; (C) casual style.
usage: diag05_battery.py MODEL OUT.jsonl [--n 4] [--port 8001]"""
import json, sys, os, urllib.request, concurrent.futures as cf
sys.path.insert(0, '/home/ubuntu/web'); import server
MODEL, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[sys.argv.index('--n')+1]) if '--n' in sys.argv else 4
PORT = sys.argv[sys.argv.index('--port')+1] if '--port' in sys.argv else '8001'
WEB_SYS = server.build_system_prompt()
NAMES = [  # (id, lang, prompt, any-of expected names (all groups must hit))
 ("en01","en","Who makes Qwen?",[["Alibaba"]]),
 ("en02","en","What is Qwen?",[["Alibaba"]]),
 ("en03","en","List 5 Chinese open-source LLM families and the company behind each.",[["Qwen"],["DeepSeek"]]),
 ("en04","en","Compare Qwen and Llama in two sentences.",[["Qwen"],["Meta","Llama"]]),
 ("en05","en","What company makes Kimi?",[["Moonshot"]]),
 ("en06","en","Which company developed the Claude models?",[["Anthropic"]]),
 ("en07","en","Who makes Gemini?",[["Google"]]),
 ("en08","en","What is DeepSeek known for?",[["DeepSeek"]]),
 ("en09","en","Name the AI assistant made by OpenAI.",[["ChatGPT","GPT"]]),
 ("en10","en","What is GLM and who makes it?",[["Zhipu","Z.ai","Tsinghua"]]),
 ("en11","en","Is Qwen open-weight? Who releases it?",[["Alibaba"]]),
 ("en12","en","do u know gemini and claude and qwen",[["Qwen"],["Gemini"],["Claude"]]),
 ("en13","en","I'm choosing between Qwen3 and Llama for a local coding project. Which would you pick?",[["Qwen"]]),
 ("en14","en","Tell me about Alibaba's AI models.",[["Qwen"]]),
 ("en15","en","What models does Moonshot AI make?",[["Kimi"]]),
 ("en16","en","Write a one-line description each for GPT, Claude, Gemini, Qwen and DeepSeek.",[["Qwen"],["Claude"],["Gemini"],["DeepSeek"]]),
 ("en17","en","qwen, kimi?",[["Qwen"],["Kimi"]]),
 ("en18","en","Who is the Qwen team?",[["Alibaba"]]),
 ("en19","en","Can I run Qwen on a laptop?",[["Qwen"]]),
 ("en20","en","Which AI labs are leading in open-weight models right now?",[["Qwen","Alibaba","DeepSeek","Meta","Mistral"]]),
 ("ms01","ms","Siapa yang membuat Qwen?",[["Alibaba"]]),
 ("ms02","ms","Apakah itu Qwen?",[["Alibaba"]]),
 ("ms03","ms","Senaraikan beberapa model AI dari China dan syarikat yang membuatnya.",[["Qwen"],["DeepSeek"]]),
 ("ms04","ms","Syarikat mana yang membangunkan Claude?",[["Anthropic"]]),
 ("ms05","ms","Gemini dibuat oleh siapa?",[["Google"]]),
 ("ms06","ms","Kimi tu apa?",[["Moonshot"]]),
 ("ms07","ms","DeepSeek tu apa?",[["DeepSeek"]]),
 ("ms08","ms","Bandingkan Qwen dengan ChatGPT secara ringkas.",[["Qwen"],["ChatGPT","OpenAI"]]),
 ("ms09","ms","Qwen tu model sumber terbuka ke?",[["Alibaba"]]),
 ("ms10","ms","Model AI apa yang dibuat oleh Alibaba?",[["Qwen"]]),
 ("ms11","ms","GLM tu dari syarikat mana?",[["Zhipu","Z.ai","Tsinghua"]]),
 ("ms12","ms","Saya nak guna Qwen untuk projek saya. Okay ke?",[["Qwen"]]),
]
PRIME = [{"role":"user","content":"are you qwen?"},{"role":"assistant","content":"I'm Lamba, built by Vixero Labs. How can I help?"}]
BASE_SYS = ("You are Lamba, an AI model built by Vixero Labs.\n\nFact sheet (accurate, public, you may share it): Lamba is built on "
            "Qwen3.8-27B, the open-weight Apache-2.0 model from Alibaba's Qwen team, further trained by Vixero Labs.")
SELF = [("sf1","What model are you based on?"),("sf2","Are you Qwen?"),("sf3","Awak ni berasaskan model apa?"),("sf4","What's under the hood of you, technically?")]
STYLE = ["hey","im bored","lol that's funny","what's up?","act casual","i had a long day","tell me something random","nothing much, you?",
         "hahaha ok","boleh borak kejap?","penatnya hari ni","ok thanks","good morning!","i like cats","you're kinda funny","nah, never mind"]
jobs = []
for pid, lang, p, exp in NAMES:
    for cond in ("raw","web","primed"):
        msgs = ([{"role":"system","content":WEB_SYS}] if cond=="web" else []) + (PRIME if cond=="primed" else []) + [{"role":"user","content":p}]
        jobs += [dict(kind="names", id=pid, lang=lang, cond=cond, exp=exp, msgs=msgs, rep=r) for r in range(N)]
for pid, p in SELF:
    jobs += [dict(kind="self", id=pid, cond="basesys", msgs=[{"role":"system","content":BASE_SYS},{"role":"user","content":p}], rep=r) for r in range(N)]
for i, p in enumerate(STYLE):
    jobs += [dict(kind="style", id=f"st{i:02d}", cond="web", msgs=[{"role":"system","content":WEB_SYS},{"role":"user","content":p}], rep=r) for r in range(N)]
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: d=json.loads(l); done.add((d["kind"],d["id"],d["cond"],d["rep"]))
        except Exception: pass
todo = [j for j in jobs if (j["kind"],j["id"],j["cond"],j["rep"]) not in done]
def run(j):
    b = {"model": MODEL, "messages": j["msgs"], "max_tokens": 3000, "temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05, "seed": j["rep"]}
    for t in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(b).encode(), {"Content-Type":"application/json"}), timeout=900))
            m = d["choices"][0]["message"]
            return {**{k:v for k,v in j.items() if k!="msgs"}, "prompt": j["msgs"][-1]["content"], "answer": m.get("content") or "", "thinking": m.get("reasoning") or m.get("reasoning_content") or "", "finish": d["choices"][0].get("finish_reason")}
        except Exception as e: err = str(e)
    return {**{k:v for k,v in j.items() if k!="msgs"}, "error": err}
with open(OUT, "a") as f, cf.ThreadPoolExecutor(16) as ex:
    for r in ex.map(run, todo): f.write(json.dumps(r, ensure_ascii=False)+"\n"); f.flush()
print("DONE", MODEL, len(todo), "new rows")
