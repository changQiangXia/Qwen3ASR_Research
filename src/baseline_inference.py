"""
Phase 1: Qwen3-ASR 基线推理脚本
支持 wav 和 m4a 格式
使用 qwen-asr 官方包
目标：在 4GB 显存下跑通量化后的模型，建立性能基线
"""

import os
import time
import torch
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Optional
import json

# 显存监控工具
class GPUMonitor:
    """监控 GPU 显存使用情况"""
    
    @staticmethod
    def get_gpu_memory_info():
        """获取当前 GPU 显存使用情况（MB）"""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "max_allocated": 0}
        
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**2,
            "reserved": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
            "free": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2
        }
    
    @staticmethod
    def print_memory_info(label=""):
        """打印显存信息"""
        info = GPUMonitor.get_gpu_memory_info()
        print(f"\n[{label}] GPU Memory:")
        print(f"  Allocated: {info['allocated']:.2f} MB ({info['allocated']/1024:.2f} GB)")
        print(f"  Reserved:  {info['reserved']:.2f} MB")
        print(f"  Max Allocated: {info['max_allocated']:.2f} MB ({info['max_allocated']/1024:.2f} GB)")
        return info


class AudioPreprocessor:
    """音频预处理，支持多种格式"""
    
    @staticmethod
    def preprocess_audio(audio_path: str) -> str:
        """
        预处理音频文件，确保是 16kHz 单声道 WAV
        支持 wav, mp3, m4a 等格式
        
        Args:
            audio_path: 原始音频路径
            
        Returns:
            处理后的 wav 文件路径（如果是 m4a 会生成临时 wav）
        """
        audio_path = Path(audio_path)
        suffix = audio_path.suffix.lower()
        
        # 已经是 wav 格式，直接返回
        if suffix == '.wav':
            return str(audio_path)
        
        # 其他格式，尝试转换
        if suffix in ['.m4a', '.mp3', '.flac', '.ogg', '.aac']:
            return AudioPreprocessor._convert_to_wav(audio_path)
        
        # 未知格式，尝试直接处理
        print(f"[WARN] 未知音频格式: {suffix}，尝试直接加载")
        return str(audio_path)
    
    @staticmethod
    def _convert_to_wav(audio_path: Path) -> str:
        """将音频转换为 16kHz 单声道 WAV"""
        output_path = audio_path.with_suffix('.wav')
        
        # 如果已存在同名 wav，直接返回
        if output_path.exists():
            print(f"[OK] 找到已转换的 WAV: {output_path.name}")
            return str(output_path)
        
        # 尝试使用 pydub 转换
        try:
            from pydub import AudioSegment
            
            print(f"[INFO] 转换 {audio_path.suffix} -> wav...")
            
            # 根据后缀确定格式
            format_map = {
                '.m4a': 'm4a',
                '.mp3': 'mp3',
                '.flac': 'flac',
                '.ogg': 'ogg',
                '.aac': 'aac'
            }
            audio_format = format_map.get(audio_path.suffix.lower(), None)
            
            audio = AudioSegment.from_file(str(audio_path), format=audio_format)
            
            # 转换为单声道、16kHz
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # 导出
            audio.export(str(output_path), format="wav")
            
            print(f"[OK] 转换完成: {output_path.name}")
            return str(output_path)
            
        except ImportError:
            print(f"[ERROR] 需要 pydub 来转换 {audio_path.suffix} 格式")
            print("请安装: pip install pydub")
            raise
        except Exception as e:
            print(f"[ERROR] 转换失败: {e}")
            raise


class Qwen3ASRBaseline:
    """
    Qwen3-ASR 基线推理类
    使用 qwen-asr 官方包
    支持 wav, m4a, mp3 等多种格式
    """
    
    def __init__(
        self,
        model_path: str = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda:0",
    ):
        # 自动检测模型路径
        if model_path is None:
            model_path = self._find_model_path()
        
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        
        self.model = None
        
        print(f"\n{'='*60}")
        print(f"Qwen3-ASR Baseline Inference")
        print(f"Model: {model_path}")
        print(f"Dtype: {dtype}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")
    
    def _find_model_path(self) -> str:
        """自动查找模型路径"""
        possible_paths = [
            "checkpoints/qwen/Qwen3-ASR-1.7B",
            "checkpoints/qwen/Qwen3-ASR-1___7B",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                print(f"[OK] 使用本地模型: {path}")
                return path
        
        # 如果本地没有，使用 ModelScope ID
        return "Qwen/Qwen3-ASR-1.7B"
    
    def load_model(self):
        """加载模型"""
        from qwen_asr import Qwen3ASRModel
        
        print("Loading model...")
        GPUMonitor.print_memory_info("Before Model Load")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            device_map=self.device,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )
        
        print("[OK] Model loaded successfully!")
        GPUMonitor.print_memory_info("After Model Load")
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Dict:
        """
        语音识别
        
        Args:
            audio_path: 音频文件路径（支持 wav, m4a, mp3 等）
            language: 指定语言（如 "Chinese", "English"），None 表示自动检测
            
        Returns:
            包含转录文本和元信息的字典
        """
        if self.model is None:
            self.load_model()
        
        # 预处理音频（格式转换）
        processed_audio = AudioPreprocessor.preprocess_audio(audio_path)
        
        # 清理显存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        mem_before = GPUMonitor.get_gpu_memory_info()
        
        # 推理
        start_time = time.time()
        
        results = self.model.transcribe(
            audio=processed_audio,
            language=language,
        )
        
        inference_time = time.time() - start_time
        
        # 记录显存峰值
        mem_after = GPUMonitor.get_gpu_memory_info()
        
        result = results[0]
        
        return {
            "text": result.text,
            "language": result.language,
            "inference_time": inference_time,
            "memory_before_mb": mem_before["allocated"],
            "memory_after_mb": mem_after["allocated"],
            "memory_peak_mb": mem_after["max_allocated"],
            "audio_file": audio_path,
            "processed_file": processed_audio,
        }
    
    def evaluate_dataset(
        self,
        audio_dir: str,
        reference_file: str = None
    ) -> Dict:
        """
        评估整个数据集
        
        Args:
            audio_dir: 音频文件目录
            reference_file: 参考文本文件（可选）
        
        Returns:
            评估结果汇总
        """
        # 支持多种音频格式
        audio_files = []
        for ext in ['*.wav', '*.m4a', '*.mp3', '*.flac', '*.ogg']:
            audio_files.extend(list(Path(audio_dir).glob(ext)))
        
        print(f"\nFound {len(audio_files)} audio files")
        
        results = []
        for audio_file in audio_files:
            print(f"\nProcessing: {audio_file.name}")
            try:
                result = self.transcribe(str(audio_file))
                result["file"] = audio_file.name
                results.append(result)
                print(f"  Text: {result['text'][:50] if result['text'] else '(empty)'}...")
                print(f"  Lang: {result['language']}")
                print(f"  Time: {result['inference_time']:.2f}s")
                print(f"  Peak VRAM: {result['memory_peak_mb']:.2f} MB")
            except Exception as e:
                print(f"  Error: {e}")
        
        # 汇总统计
        if results:
            summary = {
                "total_files": len(results),
                "avg_inference_time": np.mean([r["inference_time"] for r in results]),
                "avg_memory_peak": np.mean([r["memory_peak_mb"] for r in results]),
                "max_memory_peak": max([r["memory_peak_mb"] for r in results]),
                "results": results
            }
        else:
            summary = {"total_files": 0, "results": []}
        
        return summary


def main():
    """主函数：运行基线测试"""
    
    # 初始化
    asr = Qwen3ASRBaseline()
    
    # 单条测试
    print("\n" + "="*60)
    print("Single Audio Test")
    print("="*60)
    
    # 测试音频列表（支持 wav 和 m4a）
    test_audios = [
        r"D:\pythonProjects\Qwen3ASR_Research\data\test_m4a.wav",  # 你的录音
        "data/test_m4a.m4a",
        "data/test_recording.m4a",
        "data/test_recording.wav",
        "data/test_silence.wav",
        "data/test_sine.wav",
        "data/test_noisy.wav",
    ]
    
    # 找到存在的音频文件
    test_audio = None
    for audio in test_audios:
        if Path(audio).exists():
            test_audio = audio
            break
    
    if test_audio:
        print(f"\n测试音频: {test_audio}")
        result = asr.transcribe(test_audio)
        print(f"\n结果:")
        print(f"  语言: {result['language']}")
        print(f"  转录: '{result['text']}'")
        print(f"  推理时间: {result['inference_time']:.2f}s")
        print(f"  峰值显存: {result['memory_peak_mb']:.2f} MB ({result['memory_peak_mb']/1024:.2f} GB)")
    else:
        print(f"未找到测试音频")
        print("支持的格式: wav, m4a, mp3, flac, ogg")
        print("请将音频文件放入 data/ 目录")
    
    # 保存基线记录
    baseline_record = {
        "model_id": "Qwen3-ASR-1.7B",
        "model_path": asr.model_path,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "test_completed": test_audio is not None,
    }
    
    if test_audio:
        baseline_record.update({
            "sample_result": {
                "audio": test_audio,
                "language": result['language'],
                "text": result['text'],
                "inference_time": result['inference_time'],
                "peak_memory_mb": result['memory_peak_mb'],
            }
        })
    
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/baseline_record.json", "w", encoding="utf-8") as f:
        json.dump(baseline_record, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("Phase 1 Baseline test completed!")
    print("Record saved to: outputs/baseline_record.json")
    print("="*60)
    
    # 显存总结
    if test_audio:
        print("\n显存占用总结:")
        print(f"  峰值显存: {result['memory_peak_mb']:.1f} MB ({result['memory_peak_mb']/1024:.2f} GB)")
        if result['memory_peak_mb'] < 4000:
            print("  [OK] 符合 4GB 显存预算!")
        else:
            print("  [WARN] 显存占用接近/超过 4GB，Phase 2 需要谨慎优化")


if __name__ == "__main__":
    main()
