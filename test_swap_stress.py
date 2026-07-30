from nanovllm import LLM, SamplingParams
import time

def main():
    llm = LLM(
        model="./models/Qwen2-0.5B",
        max_num_seqs=4,
        max_num_batched_tokens=512,
        num_kvcache_blocks=6,
        kvcache_block_size=256,
        enable_remote_swap=True,
        remote_host="127.0.0.1",
        remote_port=12345,
        remote_evict_batch_size=2
    )

    prompts = [
        f"请详细介绍第{i}种大模型推理显存优化技术，完整说明实现原理、优缺点、适用场景"
        for i in range(20)
    ]
    sampling_params = SamplingParams(max_tokens=400, temperature=0.7)

    print("="*60)
    print("开始压测：20条不同前缀长序列，仅6个物理KV块，强制触发多级Swap")
    print("="*60)

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time

    print("\n" + "="*60)
    print(f"压测完成，总耗时：{total_time:.2f}s")
    # 统计已在 generate 中自动打印，这里不重复调用
    print("="*60)

    for i, output in enumerate(outputs[:5]):
        print(f"\n序列 {i} 输出片段：{output['text'][:80]}...")

if __name__ == "__main__":
    main()