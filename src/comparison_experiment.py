"""
Phase 4.1: 多条件对比实验
对比 Baseline (原版 Qwen3-ASR) vs Adapter (带 Fourier Adapter)
在不同噪声类型、SNR 条件下的性能
"""

import os
import json
import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import soundfile as sf
import librosa

from qwen3_with_adapter import Qwen3ASRWithFourierAdapter
from data_preparation import NoiseInjector


@dataclass
class ExperimentResult:
    """实验结果数据结构"""
    audio_file: str
    noise_type: str
    snr_db: float
    baseline_text: str
    adapter_text: str
    baseline_time: float
    adapter_time: float
    baseline_memory: float
    adapter_memory: float


class ComparisonExperiment:
    """
    对比实验主类
    
    实验设计（导师级严谨性）：
    1. 控制变量：同一音频、同一噪声、同一SNR
    2. 重复实验：每种条件多次测试取平均
    3. 多维度对比：准确率（WER）、速度、显存
    """
    
    def __init__(
        self,
        model_path: str = None,
        output_dir: str = "outputs/comparison",
        device: str = "cuda:0",
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*60)
        print("Comparison Experiment: Baseline vs Fourier Adapter")
        print("="*60)
        
        # 加载两个模型版本
        print("\n[1/2] Loading Baseline Model (Frozen)...")
        self.baseline_model = Qwen3ASRWithFourierAdapter(
            model_path=model_path,
            bottleneck_dim=128,
            device=device,
        )
        
        # 临时移除 adapter hook，得到纯 baseline
        if self.baseline_model.hook_handle:
            self.baseline_model.hook_handle.remove()
            self.baseline_model.hook_handle = None
        print("[OK] Baseline ready (Adapter disabled)")
        
        print("\n[2/2] Loading Adapter Model...")
        self.adapter_model = Qwen3ASRWithFourierAdapter(
            model_path=model_path,
            bottleneck_dim=128,
            device=device,
        )
        print("[OK] Adapter ready")
        
        # 噪声注入器
        self.noise_injector = NoiseInjector(sample_rate=16000)
        
        # 清理显存
        torch.cuda.empty_cache()
    
    def measure_inference(
        self,
        model: Qwen3ASRWithFourierAdapter,
        audio_path: str,
        language: str = None,
    ) -> Tuple[str, float, float]:
        """
        测量单次推理的性能
        
        Returns:
            (transcription, time_seconds, peak_memory_mb)
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        
        with torch.no_grad():
            result = model.transcribe(audio_path, language=language)
        
        elapsed = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        return result[0].text, elapsed, peak_memory
    
    def run_single_condition(
        self,
        clean_audio: np.ndarray,
        noise_type: str,
        snr_db: float,
        ground_truth: str = "",
    ) -> ExperimentResult:
        """
        运行单条件对比实验
        
        Args:
            clean_audio: 干净音频 (numpy array)
            noise_type: 噪声类型
            snr_db: 信噪比 (dB)
            ground_truth: 真实文本（用于计算 WER）
        
        Returns:
            ExperimentResult
        """
        # 生成带噪音频
        if noise_type == "clean":
            noisy_audio = clean_audio
        else:
            noisy_audio = self.noise_injector.add_noise(clean_audio, noise_type, snr_db)
        
        # 归一化
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val * 0.95
        
        # 保存临时文件
        temp_file = self.output_dir / f"temp_{noise_type}_{snr_db}.wav"
        sf.write(str(temp_file), noisy_audio, 16000)
        
        # Baseline 推理
        baseline_text, baseline_time, baseline_mem = self.measure_inference(
            self.baseline_model, str(temp_file)
        )
        
        # Adapter 推理
        adapter_text, adapter_time, adapter_mem = self.measure_inference(
            self.adapter_model, str(temp_file)
        )
        
        # 清理临时文件
        temp_file.unlink()
        
        return ExperimentResult(
            audio_file=f"{noise_type}_snr{snr_db}",
            noise_type=noise_type,
            snr_db=snr_db,
            baseline_text=baseline_text,
            adapter_text=adapter_text,
            baseline_time=baseline_time,
            adapter_time=adapter_time,
            baseline_memory=baseline_mem,
            adapter_memory=adapter_mem,
        )
    
    def run_full_comparison(
        self,
        test_audio_path: str,
        noise_conditions: List[Tuple[str, float]] = None,
        ground_truth: str = "",
    ) -> List[ExperimentResult]:
        """
        运行完整对比实验
        
        Args:
            test_audio_path: 测试音频路径
            noise_conditions: 噪声条件列表 [(type, snr), ...]
            ground_truth: 真实文本
        
        Returns:
            实验结果列表
        """
        # 加载干净音频
        clean_audio, sr = librosa.load(test_audio_path, sr=16000, mono=True)
        
        # 默认噪声条件（覆盖各种难度）
        if noise_conditions is None:
            noise_conditions = [
                ("clean", float('inf')),  # 干净
                ("white", 20),           # 高 SNR
                ("white", 10),           # 中 SNR
                ("white", 5),            # 低 SNR（困难）
                ("pink", 10),            # 粉红噪声
            ]
        
        print(f"\nRunning comparison on: {Path(test_audio_path).name}")
        print(f"Audio duration: {len(clean_audio)/16000:.1f}s")
        print(f"Conditions: {len(noise_conditions)}")
        
        results = []
        
        for noise_type, snr_db in tqdm(noise_conditions, desc="Testing conditions"):
            result = self.run_single_condition(
                clean_audio, noise_type, snr_db, ground_truth
            )
            results.append(result)
            
            # 实时显示结果
            print(f"\n  [{noise_type}, SNR={snr_db}dB]")
            print(f"    Baseline: '{result.baseline_text[:40]}...' ({result.baseline_time:.2f}s)")
            print(f"    Adapter:  '{result.adapter_text[:40]}...' ({result.adapter_time:.2f}s)")
        
        return results
    
    def save_results(self, results: List[ExperimentResult], filename: str = "comparison_results.json"):
        """保存实验结果"""
        output = []
        for r in results:
            output.append({
                "audio_file": r.audio_file,
                "noise_type": r.noise_type,
                "snr_db": r.snr_db,
                "baseline_text": r.baseline_text,
                "adapter_text": r.adapter_text,
                "baseline_time": r.baseline_time,
                "adapter_time": r.adapter_time,
                "baseline_memory_mb": r.baseline_memory,
                "adapter_memory_mb": r.adapter_memory,
            })
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n[OK] Results saved to {output_path}")
    
    def generate_report(self, results: List[ExperimentResult]):
        """生成对比报告"""
        print("\n" + "="*60)
        print("Comparison Report")
        print("="*60)
        
        # 计算平均指标
        avg_time_baseline = np.mean([r.baseline_time for r in results])
        avg_time_adapter = np.mean([r.adapter_time for r in results])
        avg_mem_baseline = np.mean([r.baseline_memory for r in results])
        avg_mem_adapter = np.mean([r.adapter_memory for r in results])
        
        print(f"\nAverage Inference Time:")
        print(f"  Baseline: {avg_time_baseline:.3f}s")
        print(f"  Adapter:  {avg_time_adapter:.3f}s")
        print(f"  Overhead: {(avg_time_adapter/avg_time_baseline-1)*100:.1f}%")
        
        print(f"\nAverage Peak Memory:")
        print(f"  Baseline: {avg_mem_baseline:.1f} MB")
        print(f"  Adapter:  {avg_mem_adapter:.1f} MB")
        print(f"  Overhead: {(avg_mem_adapter/avg_mem_baseline-1)*100:.1f}%")
        
        # 按噪声类型分组
        print(f"\nResults by Condition:")
        for r in results:
            snr_str = f"SNR={r.snr_db}dB" if r.snr_db != float('inf') else "Clean"
            print(f"\n  [{r.noise_type}, {snr_str}]")
            print(f"    Baseline: '{r.baseline_text[:50]}...'")
            print(f"    Adapter:  '{r.adapter_text[:50]}...'")


def run_demo_comparison():
    """运行演示对比实验"""
    
    # 准备测试音频
    test_audio = Path("data/test_m4a.wav")
    
    if not test_audio.exists():
        # 创建测试音频
        print("Creating test audio...")
        duration = 5  # 5秒
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # 创建简单的测试信号（模拟语音频谱）
        audio = np.sin(2 * np.pi * 200 * t) * 0.2  # 基频
        audio += np.sin(2 * np.pi * 400 * t) * 0.15  # 二次谐波
        audio += np.sin(2 * np.pi * 600 * t) * 0.1  # 三次谐波
        
        # 添加一些调制（模拟语音变化）
        envelope = np.exp(-((t - duration/2) / (duration/3)) ** 2)
        audio = audio * envelope
        
        test_audio.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(test_audio), audio.astype(np.float32), sample_rate)
        print(f"[OK] Test audio saved to {test_audio}")
    
    # 创建实验
    experiment = ComparisonExperiment()
    
    # 运行对比
    results = experiment.run_full_comparison(
        test_audio_path=str(test_audio),
        noise_conditions=[
            ("clean", float('inf')),
            ("white", 20),
            ("white", 10),
            ("white", 5),
            ("pink", 10),
        ],
    )
    
    # 保存结果
    experiment.save_results(results)
    
    # 生成报告
    experiment.generate_report(results)


if __name__ == "__main__":
    run_demo_comparison()
