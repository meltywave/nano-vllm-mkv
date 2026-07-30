from transformers import AutoTokenizer, AutoModelForCausalLM

# 这里改成你真实模型绝对路径
model_path = "/home/xxx/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True
)
print("✅ 模型加载成功！依赖降级方案生效")