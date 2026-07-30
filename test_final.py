# test_final.py
from nanovllm import LLM, SamplingParams

llm = LLM(
    model="./models/Qwen2-0.5B",
    max_num_seqs=4,
    max_num_batched_tokens=512,
    num_kvcache_blocks=6,
    kvcache_block_size=256,
    enable_remote_swap=True,
    remote_host="127.0.0.1",
    remote_port=12345,
)

prompts = [f"请详细介绍第{i}种大模型推理显存优化技术" for i in range(20)]
outputs = llm.generate(prompts, SamplingParams(max_tokens=400))

for i, out in enumerate(outputs[:3]):
    print(f"序列{i}: {out['text'][:100]}...")