import os
os.environ["HF_HUB_OFFLINE"] = "1"

from nanovllm.engine.block_manager import BlockManager

def main():
    print("===== Start Remote Swap Framework Init Test =====")
    # 仅初始化BlockManager，建立TCP连接，不加载LLM
    bm = BlockManager(num_blocks=16, block_size=128)
    print("✅ SUCCESS: BlockManager created, connected to remote KV Server!")
    print("Swap core module initialization passed.")

if __name__ == "__main__":
    main()