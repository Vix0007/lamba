import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
def main():
    t=time.time()
    llm=LLM(model="/home/ubuntu/models/soup-linear-l1.0-lmh",quantization="fp8",dtype="bfloat16",gpu_memory_utilization=0.85,
            max_model_len=8192,trust_remote_code=True,enforce_eager=True,enable_lora=True,max_lora_rank=256,max_loras=1)
    print("LOADED",time.time()-t,flush=True)
    tok=llm.get_tokenizer()
    p=tok.apply_chat_template([{"role":"user","content":"Explain why the sky is blue in three sentences."}],tokenize=False,add_generation_prompt=True)
    sp=SamplingParams(temperature=0.0,max_tokens=120)
    a=llm.generate([p],sp)[0].outputs[0].text
    b=llm.generate([p],sp,lora_request=LoRARequest("ftpo1",1,"/home/ubuntu/adapters_ftpo1"))[0].outputs[0].text
    print("BASE   :",repr(a[:300])); print("ADAPTER:",repr(b[:300])); print("DIFFER:",a!=b,flush=True)
    
if __name__=='__main__':
    main()
