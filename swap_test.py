import os
from time import perf_counter

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


TARGET_PROMPT_TOKENS = 512
OUTPUT_TOKENS = 320


PROMPT_SOURCES = [
    (
        "Distributed systems",
        "Explain how a distributed storage system maintains consistency "
        "during leader failure, network partition, replica recovery, and "
        "concurrent client writes. Discuss logs, epochs, quorum decisions, "
        "failure detection, and recovery ordering using concrete examples. ",
    ),
    (
        "Operating systems",
        "Analyze virtual memory behavior under mixed sequential and random "
        "access workloads. Discuss page tables, TLB misses, page faults, "
        "replacement policies, NUMA placement, memory mapping, and swap. ",
    ),
    (
        "LLM inference",
        "Design an efficient large language model inference engine. Discuss "
        "continuous batching, paged KV cache, prefix reuse, prefill, decode, "
        "GPU scheduling, CPU offloading, synchronization, and throughput. ",
    ),
    (
        "Database systems",
        "Explain transaction processing in a distributed database. Discuss "
        "MVCC, write-ahead logging, isolation levels, deadlock handling, "
        "replication, checkpoints, crash recovery, and query scheduling. ",
    ),
]


def build_exact_prompt(tokenizer, title, source):
    prefix = (
        f"Topic: {title}\n"
        "Write a detailed technical report based on the following material. "
        "Preserve important assumptions and reason step by step.\n\n"
    )

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    source_ids = tokenizer.encode(source, add_special_tokens=False)

    if not source_ids:
        raise RuntimeError(f"Tokenizer produced no tokens for {title}")

    remaining = TARGET_PROMPT_TOKENS - len(prefix_ids)
    if remaining <= 0:
        return prefix_ids[:TARGET_PROMPT_TOKENS]

    repeats = (remaining + len(source_ids) - 1) // len(source_ids)
    token_ids = prefix_ids + (source_ids * repeats)

    return token_ids[:TARGET_PROMPT_TOKENS]


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
        cache_engine=2,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        kvcache_block_size=256,

        # 4 * 512-token prompts ???? 8 ? GPU blocks?
        # ??? decode ??????????? 3 ? block?
        # ?????? swap-out?
        num_kvcache_blocks=8,
        num_cpu_kvcache_blocks=16,
    )

    block_manager = llm.scheduler.block_manager
    gpu_blocks = len(block_manager.blocks)
    scheduler_cpu_blocks = len(block_manager.cpu_blocks)

    runner_cpu_blocks = (
        0
        if llm.model_runner.cpu_kv_cache is None
        else llm.model_runner.cpu_kv_cache.shape[2]
    )

    print("GPU KV cache blocks:", gpu_blocks)
    print("Runner CPU KV cache blocks:", runner_cpu_blocks)
    print("Scheduler CPU KV cache blocks:", scheduler_cpu_blocks)

    assert gpu_blocks == 8, (f"Expected 8 GPU KV cache blocks, got {gpu_blocks}")
    assert runner_cpu_blocks == 16, (f"Expected 16 CPU KV cache blocks, got {runner_cpu_blocks}")
    assert scheduler_cpu_blocks == 16, (f"Expected 16 scheduler CPU KV cache blocks, got {scheduler_cpu_blocks}")

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

    print("\nCompletion token lengths:", output_lengths)
    print("Swap-out blocks:", swap_out_blocks)
    print("Swap-in blocks:", swap_in_blocks)
    print("Elapsed seconds:", round(elapsed, 3))
    print(
        "Generation throughput:",
        round(sum(output_lengths) / elapsed, 2),
        "tokens/s",
    )

    assert output_lengths == [OUTPUT_TOKENS] * 4
    assert swap_out_blocks > 0, "No GPU-to-CPU swap occurred"
    assert swap_in_blocks > 0, "No CPU-to-GPU swap occurred"

    for index, output in enumerate(outputs):
        preview = output["text"][:200].replace("\n", " ")
        print(f"Output {index}: {preview!r}")


if __name__ == "__main__":
    main()
