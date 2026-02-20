"""
从 ModelScope 下载 Qwen3-ASR 模型
解决 HuggingFace 访问慢的问题
"""

import os
from pathlib import Path
from modelscope import snapshot_download

def download_qwen3_asr(
    model_id: str = "qwen/Qwen3-ASR-1.7B",
    cache_dir: str = "./checkpoints",
    local_dir: str = None
):
    """
    从 ModelScope 下载模型
    
    Args:
        model_id: ModelScope 模型 ID
        cache_dir: 缓存目录
        local_dir: 本地保存目录（可选，默认使用 cache_dir）
    """
    print(f"Downloading model: {model_id}")
    print(f"Cache directory: {Path(cache_dir).absolute()}")
    
    try:
        model_path = snapshot_download(
            model_id,
            cache_dir=cache_dir,
            local_dir=local_dir
        )
        print(f"\nModel downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nTrying alternative method...")
        return None


if __name__ == "__main__":
    # 创建 checkpoints 目录
    os.makedirs("checkpoints", exist_ok=True)
    
    # 下载模型
    model_path = download_qwen3_asr(
        model_id="qwen/Qwen3-ASR-1.7B",
        cache_dir="./checkpoints"
    )
    
    if model_path:
        print("\nDownload completed successfully!")
        print(f"You can now use this path in your scripts: {model_path}")
