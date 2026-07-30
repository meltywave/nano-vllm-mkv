from nanovllm import LLM, SamplingParams
import time
import random
import string

def generate_random_prompt(offset):
    random_prefix = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
    return f"[{random_prefix}] 请详细介绍第{offset}种大模型显存优化技术，讲清原理、实现、优缺点，至少200字"

def main():
    sampling_params = SamplingParams(max_tokens=256, temperature=0.7)

    llm = LLM(
        model="./models/Qwen2-0.5B",
        max_num_seqs=4,              # 4条并发，制造压力
        max_num_batched_tokens=512,
        num_kvcache_blocks=6,        # 6个块，4条×2块=8块需求，必然触发Swap
        kvcache_block_size=256,
        enable_remote_swap=True,
        remote_host="127.0.0.1",
        remote_port=12345,
        remote_evict_batch_size=2
    )

    all_outputs = []
    batch_num = 6
    per_batch = 5

    print("="*60)
    print("分层分批压测Swap：分6批、每批5条随机前缀序列，6个物理块，4条并发")
    print(f"每序列最大生成256tokens，强制触发跨层Swap")
    print("="*60)
    start_time = time.time()

    for b in range(batch_num):
        prompts = [generate_random_prompt(b * per_batch + i) for i in range(per_batch)]
        print(f"\n--- 执行第{b+1}/{batch_num}批 ---")
        batch_out = llm.generate(prompts, sampling_params)
        all_outputs.extend(batch_out)

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"压测全部完成，总序列 {len(all_outputs)}，总耗时：{total_time:.2f}s")
    print("="*60)

    print("\n前3条生成结果片段：")
    for idx, out in enumerate(all_outputs[:3]):
        print(f"\n序列{idx}: {str(out)[:100]}...")

if __name__ == "__main__":
    main()