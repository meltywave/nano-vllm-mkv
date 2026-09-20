import os
import threading
from time import perf_counter

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.engine.remote_cache_server import RemoteCacheServer
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

    prompt_sources = PROMPT_SOURCES + PROMPT_SOURCES[:2]
    prompts = [
        build_exact_prompt(tokenizer, title, source)
        for title, source in prompt_sources
    ]

    prompt_lengths = [len(prompt) for prompt in prompts]
    print("Prompt token lengths:", prompt_lengths)
    assert prompt_lengths == [TARGET_PROMPT_TOKENS] * 6

    remote_server = RemoteCacheServer(("127.0.0.1", 0))
    server_thread = threading.Thread(
        target=remote_server.serve_forever,
        daemon=True,
    )
    server_thread.start()
    llm = None

    try:
        llm = LLM(
            model_path,
            enforce_eager=True,
            tensor_parallel_size=1,
            max_model_len=1024,
            max_num_seqs=6,
            max_num_batched_tokens=3072,
            kvcache_block_size=256,

            # CPU and SSD can each hold one 512-token sequence. Six
            # concurrent prompts force the coldest SSD sequence to remote.
            num_kvcache_blocks=8,
            num_cpu_kvcache_blocks=2,
            num_ssd_kvcache_blocks=2,
            num_remote_kvcache_blocks=16,
            remote_kvcache_host="127.0.0.1",
            remote_kvcache_port=remote_server.server_port,
        )

        block_manager = llm.scheduler.block_manager
        gpu_blocks = len(block_manager.blocks)
        scheduler_cpu_blocks = len(block_manager.cpu_blocks)
        scheduler_ssd_blocks = len(block_manager.ssd_blocks)
        scheduler_remote_blocks = len(block_manager.remote_blocks)

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
        runner_remote_blocks = (
            0
            if llm.model_runner.remote_kv_cache is None
            else llm.model_runner.remote_kv_cache.num_blocks
        )

        print("GPU KV cache blocks:", gpu_blocks)
        print("Runner CPU KV cache blocks:", runner_cpu_blocks)
        print("Scheduler CPU KV cache blocks:", scheduler_cpu_blocks)
        print("Runner SSD KV cache blocks:", runner_ssd_blocks)
        print("Scheduler SSD KV cache blocks:", scheduler_ssd_blocks)
        print("Runner remote KV cache blocks:", runner_remote_blocks)
        print("Scheduler remote KV cache blocks:", scheduler_remote_blocks)

        assert gpu_blocks == 8
        assert runner_cpu_blocks == scheduler_cpu_blocks == 2
        assert runner_ssd_blocks == scheduler_ssd_blocks == 2
        assert runner_remote_blocks == scheduler_remote_blocks == 16

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
        print("\nCompletion token lengths:", output_lengths)
        print("Swap-out blocks:", scheduler.num_swap_out_blocks)
        print("Swap-in blocks:", scheduler.num_swap_in_blocks)
        print("GPU -> CPU blocks:", scheduler.num_gpu_to_cpu_blocks)
        print("CPU -> SSD blocks:", scheduler.num_cpu_to_ssd_blocks)
        print("SSD -> remote blocks:", scheduler.num_ssd_to_remote_blocks)
        print("Remote -> GPU blocks:", scheduler.num_remote_to_gpu_blocks)
        print("Elapsed seconds:", round(elapsed, 3))
        print(
            "Generation throughput:",
            round(sum(output_lengths) / elapsed, 2),
            "tokens/s",
        )

        assert output_lengths == [OUTPUT_TOKENS] * 6
        assert scheduler.num_swap_out_blocks > 0
        assert scheduler.num_swap_in_blocks > 0
        assert scheduler.num_gpu_to_cpu_blocks > 0
        assert scheduler.num_cpu_to_ssd_blocks > 0
        assert scheduler.num_ssd_to_remote_blocks > 0
        assert scheduler.num_remote_to_gpu_blocks > 0

        for index, output in enumerate(outputs):
            preview = output["text"][:200].replace("\n", " ")
            print(f"Output {index}: {preview!r}")
    finally:
        if llm is not None:
            llm.exit()
        remote_server.shutdown()
        remote_server.server_close()
        server_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
