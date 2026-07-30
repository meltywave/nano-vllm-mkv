import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 绕过新版本transformers本地文件夹repo_id校验BUG
import transformers.utils.hub as hub_module
def dummy_validate(*args, **kwargs):
    return
hub_module._validate_repo_id = dummy_validate

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams

def main():
    print("===== 初始化 Nano-vLLM 引擎 =====")
    engine = LLMEngine(
        model="./models/Qwen3-0.6B",
        tensor_parallel_size=1,
        kvcache_block_size=256,
        max_gpu_blocks=300,
        # enforce_eager=True
    )
    print("引擎初始化完成，准备提交推理请求")

    prompts = [
        "请详细介绍人工智能发展历史",
        "讲解Transformer架构的核心原理",
        "简述大模型KV Cache优化方案有哪些",
        "什么是Prefill阶段和Decode阶段",
        "远端内存Swap对于LLM推理的价值",
        "如何使用LRU策略淘汰冷KV块",
    ]

    sampling_params = SamplingParams(
        max_tokens=128,
        temperature=0.7
    )

    print("开始批量推理……")
    results = engine.generate(prompts, sampling_params)

    for idx, res in enumerate(results):
        print(f"\n【Prompt {idx+1} 输出】\n{res['text']}")

if __name__ == "__main__":
    main()