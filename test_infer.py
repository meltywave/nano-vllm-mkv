from nanovllm import LLM, SamplingParams
import time

def main():
    model_path = "./models/Qwen2-0.5B"
    engine = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_num_seqs=20,
        max_num_batched_tokens=256,  # 单次批处理token极小，序列长时间并发占用块
        num_kvcache_blocks=6,         # 本地仅6个KV块，极限压缩
        kvcache_block_size=256,
        enable_remote_swap=True,
        remote_host="127.0.0.1",
        remote_port=12345,
        remote_evict_batch_size=4
    )

    long_prompt = "写一篇1000字大模型KV Cache远端Swap技术完整论文，包含原理、paged attention协同、LRU淘汰、网络传输优化、性能对比实验"
    sampling_cfg = SamplingParams(max_tokens=1024, temperature=0.7)

    # 提交10条超长推理任务，并发压力拉满
    total_req = 10
    for i in range(total_req):
        engine.add_request(long_prompt, sampling_cfg)
        print(f"已提交第{i+1}条超长推理任务")
        time.sleep(0.05)

    outputs = {}
    while not engine.is_finished():
        output, _ = engine.step()
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids

    print("===== 推理生成结果 =====")
    for seq_id in sorted(outputs.keys()):
        text = engine.tokenizer.decode(outputs[seq_id])
        print(f"\n序列 {seq_id} 输出片段：{text[:200]}...")

if __name__ == "__main__":
    main()