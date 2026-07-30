import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = "/home/aurora/work/kvcache/nano-vllm/models/Qwen3-0.6B"
    print("Model path:", path)
    print("Is dir?", os.path.isdir(path))

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "Write a long article about artificial intelligence and large language model memory bottleneck, KV Cache swap mechanism.",
        "Explain multi-level cache strategy for LLM inference, GPU memory, CPU RAM, remote network storage",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    llm.scheduler.block_manager.print_swap_statistics()


if __name__ == "__main__":
    main()