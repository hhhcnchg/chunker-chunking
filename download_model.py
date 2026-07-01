"""
下载 bge-small-zh-v1.5 到本地（使用国内镜像）
"""
import os
import sys

# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
LOCAL_PATH = os.path.expanduser("~/models/bge-small-zh-v1.5")

print(f"[下载] 模型: {MODEL_NAME}")
print(f"[下载] 本地路径: {LOCAL_PATH}")
print(f"[下载] 使用镜像: https://hf-mirror.com")
print()

try:
    from huggingface_hub import snapshot_download
    print("[下载] 开始下载...")
    path = snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=LOCAL_PATH,
        local_dir_use_symlinks=False,
    )
    print(f"\n[完成] 下载到: {path}")

    # 验证文件
    files = os.listdir(path)
    print(f"[验证] 文件列表: {files}")

except Exception as e:
    print(f"[错误] {e}")
    print("\n尝试 pip install huggingface_hub 后重试")
