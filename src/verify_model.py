"""
验证下载的模型是否完整，并测试显存占用
使用 qwen-asr 官方包的方式加载
"""

import os
import sys
import torch
from pathlib import Path

def get_model_path():
    """获取模型本地路径"""
    possible_paths = [
        "checkpoints/qwen/Qwen3-ASR-1.7B",
        "checkpoints/qwen/Qwen3-ASR-1___7B",
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"[OK] 找到模型路径: {path}")
            return path
    
    checkpoints_dir = Path("checkpoints")
    for pattern in ["**/Qwen3-ASR*", "**/qwen/**"]:
        matches = list(checkpoints_dir.glob(pattern))
        if matches:
            print(f"[OK] 找到模型路径: {matches[0]}")
            return str(matches[0])
    
    return None

def verify_model_files(model_path):
    """验证模型文件完整性"""
    print("\n" + "="*60)
    print("验证模型文件完整性")
    print("="*60)
    
    model_dir = Path(model_path)
    required_files = [
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ]
    
    all_exist = True
    for file in required_files:
        file_path = model_dir / file
        exists = file_path.exists()
        status = "[OK]" if exists else "[FAIL]"
        print(f"{status} {file}")
        all_exist = all_exist and exists
    
    safetensors_files = list(model_dir.glob("*.safetensors"))
    print(f"\n[OK] 找到 {len(safetensors_files)} 个权重文件:")
    total_size = 0
    for f in safetensors_files:
        size_mb = f.stat().st_size / 1024**2
        total_size += size_mb
        print(f"  - {f.name}: {size_mb:.1f} MB")
    
    print(f"\n总权重大小: {total_size/1024:.2f} GB")
    
    return all_exist

def test_model_load_qwen_asr(model_path):
    """使用 qwen-asr 官方包测试模型加载"""
    print("\n" + "="*60)
    print("测试模型加载 (使用 qwen-asr 包)")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("[WARN] 未检测到 CUDA，跳过 GPU 测试")
        return None
    
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"\nGPU: {gpu_name}")
    print(f"总显存: {total_vram:.2f} GB")
    
    # 清理显存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1024**2
    print(f"\n加载前显存占用: {mem_before:.2f} MB")
    
    try:
        from qwen_asr import Qwen3ASRModel
        
        print("\n正在加载模型 (INT8 量化)...")
        print("注意: qwen-asr 会自动处理量化，无需手动配置")
        
        # 使用 qwen-asr 官方方式加载
        # 尝试使用较小的 dtype 来节省显存
        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.float16,  # 使用 FP16 节省显存
            device_map="cuda:0",
            max_inference_batch_size=1,  # 最小 batch size
            max_new_tokens=256,
        )
        
        print("[OK] 模型加载成功!")
        
        # 记录显存
        mem_after = torch.cuda.memory_allocated() / 1024**2
        mem_peak = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"\n加载后显存占用: {mem_after:.2f} MB ({mem_after/1024:.2f} GB)")
        print(f"显存峰值: {mem_peak:.2f} MB ({mem_peak/1024:.2f} GB)")
        print(f"模型占用显存: {(mem_after - mem_before):.2f} MB ({(mem_after - mem_before)/1024:.2f} GB)")
        
        # 测试推理
        print("\n--- 测试简单推理 ---")
        
        # 创建一个简单的测试音频（静音）
        import numpy as np
        sample_rate = 16000
        duration = 1  # 1秒
        dummy_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        print("输入: 1秒静音音频")
        
        with torch.no_grad():
            results = model.transcribe(audio=(dummy_audio, sample_rate))
        
        print(f"输出: '{results[0].text}'")
        print("[OK] 推理测试通过!")
        
        # 清理
        del model
        torch.cuda.empty_cache()
        
        return {
            "status": "success",
            "memory_mb": mem_after,
            "peak_memory_mb": mem_peak,
            "model_memory_mb": mem_after - mem_before,
        }
        
    except ImportError as e:
        print(f"[FAIL] 无法导入 qwen_asr: {e}")
        print("请确保已安装: pip install qwen-asr")
        return {"status": "import_error", "error": str(e)}
        
    except Exception as e:
        print(f"[FAIL] 模型加载或推理失败: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

def test_baseline_inference(model_path):
    """测试基线推理 - 使用官方推荐方式"""
    print("\n" + "="*60)
    print("基线推理测试")
    print("="*60)
    
    try:
        from qwen_asr import Qwen3ASRModel
        
        # 如果有测试音频就用，否则跳过
        test_audio = "data/test_sample.wav"
        
        if not Path(test_audio).exists():
            print(f"[WARN] 测试音频不存在: {test_audio}")
            print("跳过基线推理测试")
            return None
        
        print(f"加载模型...")
        model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="cuda:0",
        )
        
        print(f"推理: {test_audio}")
        import time
        start = time.time()
        
        results = model.transcribe(audio=test_audio)
        
        elapsed = time.time() - start
        print(f"\n结果:")
        print(f"  语言: {results[0].language}")
        print(f"  文本: {results[0].text}")
        print(f"  耗时: {elapsed:.2f}s")
        
        del model
        torch.cuda.empty_cache()
        
        return {
            "status": "success",
            "text": results[0].text,
            "time": elapsed
        }
        
    except Exception as e:
        print(f"[FAIL] {e}")
        return {"status": "failed", "error": str(e)}

def main():
    print("\n" + "="*60)
    print("Qwen3-ASR 模型验证工具")
    print("="*60)
    
    # 1. 获取模型路径
    model_path = get_model_path()
    if not model_path:
        print("[FAIL] 未找到模型，请检查 checkpoints 目录")
        return
    
    # 2. 验证文件完整性
    if not verify_model_files(model_path):
        print("\n[FAIL] 模型文件不完整!")
        return
    
    # 3. 测试模型加载
    result = test_model_load_qwen_asr(model_path)
    
    # 4. 测试基线推理（如果有测试音频）
    if result and result.get("status") == "success":
        baseline_result = test_baseline_inference(model_path)
    
    # 5. 总结
    print("\n" + "="*60)
    print("验证完成总结")
    print("="*60)
    
    if result and result.get("status") == "success":
        print(f"\n[OK] 模型加载成功")
        print(f"  显存占用: {result['model_memory_mb']:.1f} MB ({result['model_memory_mb']/1024:.2f} GB)")
        
        if result['peak_memory_mb'] < 3500:
            print("  [OK] 符合 4GB 显存预算要求!")
        else:
            print("  [WARN] 显存占用较高，Phase 2-3 可能需要进一步优化")
        
        print("\n下一步:")
        print("  1. 准备测试音频 (data/test_sample.wav)")
        print("  2. 运行完整 baseline: python src/baseline_inference.py")
    else:
        print(f"\n[FAIL] 验证失败")
        if result:
            print(f"  错误: {result.get('error', 'Unknown')}")

if __name__ == "__main__":
    main()
