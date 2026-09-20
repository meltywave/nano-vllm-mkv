import os
from time import perf_counter

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from swap_test import (
    OUTPUT_TOKENS,
    PROMPT_SOURCES,
    TARGET_PROMPT_TOKENS,
    build_exact_prompt,
)


def main():
    torch.manual_seed(0)

    model_path = os.path.expanduser(
        "~/tools/huggingface/Qwen3-0.6B/"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )

    prompts = [
        build_exact_prompt(tokenizer, title, source)
        for title, source in PROMPT_SOURCES
    ]

    prompt_lengths = [len(prompt) for prompt in prompts]
    print("Prompt token lengths:", prompt_lengths)
    assert prompt_lengths == [TARGET_PROMPT_TOKENS] * 4

    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        kvcache_block_size=256,

        # 与 swap_test.py 使用相同的 GPU 容量和 workload。
        # CPU 只容纳一条 512-token 序列，下一次抢占会先把最冷的
        # CPU 序列降级到 SSD，再复用 CPU blocks。
        num_kvcache_blocks=8,
        num_cpu_kvcache_blocks=2,
        num_ssd_kvcache_blocks=16,
    )

    block_manager = llm.scheduler.block_manager
    gpu_blocks = len(block_manager.blocks)
    scheduler_cpu_blocks = len(block_manager.cpu_blocks)
    scheduler_ssd_blocks = len(block_manager.ssd_blocks)

    runner_cpu_blocks = (
        0
        if llm.model_runner.cpu_kv_cache is None
        else llm.model_runner.cpu_kv_cache.shape[2]
    )
    runner_ssd_blocks = (
        0
        if llm.model_runner.ssd_kv_cache is None
        else llm.model_runner.ssd_kv_cache.num_blocks
    )

    print("GPU KV cache blocks:", gpu_blocks)
    print("Runner CPU KV cache blocks:", runner_cpu_blocks)
    print("Scheduler CPU KV cache blocks:", scheduler_cpu_blocks)
    print("Runner SSD KV cache blocks:", runner_ssd_blocks)
    print("Scheduler SSD KV cache blocks:", scheduler_ssd_blocks)

    assert gpu_blocks == 8, (
        f"Expected 8 GPU KV cache blocks, got {gpu_blocks}"
    )
    assert runner_cpu_blocks == 2, (
        f"Expected 2 CPU KV cache blocks, got {runner_cpu_blocks}"
    )
    assert scheduler_cpu_blocks == 2, (
        f"Expected 2 scheduler CPU KV cache blocks, got {scheduler_cpu_blocks}"
    )
    assert runner_ssd_blocks == 16, (
        f"Expected 16 SSD KV cache blocks, got {runner_ssd_blocks}"
    )
    assert scheduler_ssd_blocks == 16, (
        f"Expected 16 scheduler SSD KV cache blocks, got {scheduler_ssd_blocks}"
    )

    sampling_params = SamplingParams(
        temperature=1e-5,
        max_tokens=OUTPUT_TOKENS,
        ignore_eos=False,
    )

    started = perf_counter()
    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
    )
    elapsed = perf_counter() - started

    output_lengths = [
        len(output["token_ids"])
        for output in outputs
    ]

    scheduler = llm.scheduler
    swap_out_blocks = scheduler.num_swap_out_blocks
    swap_in_blocks = scheduler.num_swap_in_blocks
    gpu_to_cpu_blocks = scheduler.num_gpu_to_cpu_blocks
    cpu_to_ssd_blocks = scheduler.num_cpu_to_ssd_blocks
    gpu_to_ssd_blocks = scheduler.num_gpu_to_ssd_blocks
    ssd_to_gpu_blocks = scheduler.num_ssd_to_gpu_blocks

    print("\nCompletion token lengths:", output_lengths)
    print("Swap-out blocks:", swap_out_blocks)
    print("Swap-in blocks:", swap_in_blocks)
    print("GPU -> CPU blocks:", gpu_to_cpu_blocks)
    print("CPU -> SSD blocks:", cpu_to_ssd_blocks)
    print("GPU -> SSD blocks:", gpu_to_ssd_blocks)
    print("SSD -> GPU blocks:", ssd_to_gpu_blocks)
    print("Elapsed seconds:", round(elapsed, 3))
    print(
        "Generation throughput:",
        round(sum(output_lengths) / elapsed, 2),
        "tokens/s",
    )

    assert output_lengths == [OUTPUT_TOKENS] * 4
    assert swap_out_blocks > 0, "No GPU swap-out occurred"
    assert swap_in_blocks > 0, "No swap-in to GPU occurred"
    assert gpu_to_cpu_blocks > 0, "No GPU-to-CPU swap occurred"
    assert cpu_to_ssd_blocks > 0, "No CPU-to-SSD demotion occurred"
    assert ssd_to_gpu_blocks > 0, "No SSD-to-GPU swap occurred"

    for index, output in enumerate(outputs):
        preview = output["text"][:200].replace("\n", " ")
        print(f"Output {index}: {preview!r}")


if __name__ == "__main__":
    main()
