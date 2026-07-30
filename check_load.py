import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, AutoModelForCausalLM

# 换成兼容4.43.4版本的Qwen2
model_id = "Qwen/Qwen2-0.5B"
print(f"尝试在线加载 {model_id}")

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    local_files_only=False
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    local_files_only=False
)
print("✅ 模型加载预检通过！")