import os
import transformers.utils.hub as hub_module

original_validate = hub_module._validate_repo_id

def patched_validate_repo_id(repo_id: str):
    # 如果是本地存在文件夹，直接放行，跳过正则校验
    if os.path.isdir(repo_id):
        return
    # 其余情况执行原版校验
    original_validate(repo_id)

# 动态替换函数
hub_module._validate_repo_id = patched_validate_repo_id
print("✅ Monkey Patch applied: allow local directory as model path")