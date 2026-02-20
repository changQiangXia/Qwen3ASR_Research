"""
Phase 3.1: 数据构造 - 人工加噪音频数据集
为抗噪微调准备带噪声的训练数据
"""

import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import List, Tuple, Optional
import json


class NoiseInjector:
    """
    噪声注入器 - 为干净音频添加各种类型的噪声
    """
    
    NOISE_TYPES = ['white', 'pink', 'gaussian', 'babble', 'cafe']
    
    def __init__(self, noise_dir: str = None, sample_rate: int = 16000):
        """
        Args:
            noise_dir: 噪声文件目录（用于真实环境噪声）
            sample_rate: 目标采样率
        """
        self.noise_dir = Path(noise_dir) if noise_dir else None
        self.sample_rate = sample_rate
        
        # 加载真实噪声文件（如果提供）
        self.noise_files = {}
        if self.noise_dir and self.noise_dir.exists():
            for noise_type in ['babble', 'cafe', 'street', 'rain']:
                files = list(self.noise_dir.glob(f"*{noise_type}*.wav")) + \
                       list(self.noise_dir.glob(f"*{noise_type}*.mp3"))
                if files:
                    self.noise_files[noise_type] = files
                    print(f"[OK] Loaded {len(files)} {noise_type} noise files")
    
    def add_white_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        """添加白噪声"""
        # 计算信号功率
        signal_power = np.mean(audio ** 2)
        
        # 根据 SNR 计算噪声功率
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # 生成白噪声
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        
        return audio + noise
    
    def add_pink_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        """添加粉红噪声（1/f 噪声，更接近自然噪声）"""
        # 生成白噪声
        white = np.random.randn(len(audio))
        
        # 通过滤波器转换为粉红噪声
        # 使用简化的 Voss-McCartney 算法
        n = len(audio)
        pink = np.zeros(n)
        
        # 多个白噪声源叠加
        num_sources = 8
        sources = [np.random.randn(n) for _ in range(num_sources)]
        
        for i in range(n):
            # 根据频率选择不同的源
            source_idx = int(np.log2(i + 1)) % num_sources if i > 0 else 0
            pink[i] = sources[source_idx][i]
        
        # 归一化
        pink = pink / np.std(pink)
        
        # 调整功率
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        pink = pink * np.sqrt(noise_power)
        
        return audio + pink
    
    def add_real_noise(self, audio: np.ndarray, noise_type: str, snr_db: float) -> np.ndarray:
        """添加真实环境噪声"""
        if noise_type not in self.noise_files or not self.noise_files[noise_type]:
            print(f"[WARN] No {noise_type} noise files available, using white noise instead")
            return self.add_white_noise(audio, snr_db)
        
        # 随机选择一个噪声文件
        noise_file = np.random.choice(self.noise_files[noise_type])
        
        # 加载噪声
        noise, sr = librosa.load(str(noise_file), sr=self.sample_rate, mono=True)
        
        # 如果噪声太短，循环重复
        if len(noise) < len(audio):
            repeats = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, repeats)
        
        # 裁剪到与音频相同长度
        noise = noise[:len(audio)]
        
        # 调整噪声功率
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        current_noise_power = np.mean(noise ** 2)
        
        if current_noise_power > 0:
            noise = noise * np.sqrt(noise_power / current_noise_power)
        
        return audio + noise
    
    def add_noise(self, audio: np.ndarray, noise_type: str, snr_db: float) -> np.ndarray:
        """
        添加指定类型的噪声
        
        Args:
            audio: 干净音频
            noise_type: 噪声类型 ('white', 'pink', 'babble', 'cafe', 'street')
            snr_db: 信噪比 (dB)
        
        Returns:
            带噪音频
        """
        if noise_type in ['white', 'gaussian']:
            return self.add_white_noise(audio, snr_db)
        elif noise_type == 'pink':
            return self.add_pink_noise(audio, snr_db)
        elif noise_type in self.noise_files:
            return self.add_real_noise(audio, noise_type, snr_db)
        else:
            print(f"[WARN] Unknown noise type: {noise_type}, using white noise")
            return self.add_white_noise(audio, snr_db)


def create_noisy_dataset(
    clean_audio_dir: str,
    output_dir: str,
    noise_types: List[str] = None,
    snr_levels: List[float] = None,
    noise_dir: str = None,
):
    """
    创建带噪数据集
    
    Args:
        clean_audio_dir: 干净音频目录
        output_dir: 输出目录
        noise_types: 噪声类型列表
        snr_levels: SNR 级别列表 (dB)
        noise_dir: 真实噪声文件目录
    """
    clean_audio_dir = Path(clean_audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 默认配置
    if noise_types is None:
        noise_types = ['white', 'pink']
    if snr_levels is None:
        snr_levels = [20, 15, 10, 5]  # 从高到低，难度递增
    
    print("="*60)
    print("Creating Noisy Dataset")
    print("="*60)
    print(f"Clean audio dir: {clean_audio_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Noise types: {noise_types}")
    print(f"SNR levels: {snr_levels} dB")
    
    # 初始化噪声注入器
    injector = NoiseInjector(noise_dir=noise_dir, sample_rate=16000)
    
    # 获取所有干净音频
    clean_files = list(clean_audio_dir.glob("*.wav")) + \
                  list(clean_audio_dir.glob("*.mp3"))
    
    print(f"\nFound {len(clean_files)} clean audio files")
    
    # 生成带噪音频
    manifest = []
    
    for i, clean_file in enumerate(clean_files):
        print(f"\n[{i+1}/{len(clean_files)}] Processing: {clean_file.name}")
        
        # 加载干净音频
        audio, sr = librosa.load(str(clean_file), sr=16000, mono=True)
        
        # 为每种噪声类型和 SNR 生成带噪版本
        for noise_type in noise_types:
            for snr in snr_levels:
                # 生成带噪音频
                noisy_audio = injector.add_noise(audio, noise_type, snr)
                
                # 归一化防止削波
                max_val = np.max(np.abs(noisy_audio))
                if max_val > 1.0:
                    noisy_audio = noisy_audio / max_val * 0.95
                
                # 保存
                output_filename = f"{clean_file.stem}_{noise_type}_snr{int(snr)}.wav"
                output_path = output_dir / output_filename
                sf.write(str(output_path), noisy_audio, 16000)
                
                # 记录到 manifest
                manifest.append({
                    "clean_file": str(clean_file.name),
                    "noisy_file": output_filename,
                    "noise_type": noise_type,
                    "snr_db": snr,
                })
                
                print(f"  [OK] {output_filename}")
    
    # 保存 manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("Dataset creation complete!")
    print(f"Total samples: {len(manifest)}")
    print(f"Manifest saved to: {manifest_path}")
    print("="*60)
    
    return manifest


def test_noise_injection():
    """测试噪声注入"""
    print("="*60)
    print("Testing Noise Injection")
    print("="*60)
    
    # 创建测试音频
    duration = 3  # 3秒
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 440Hz 正弦波 + 说话模拟
    clean = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
    
    # 添加一些谐波模拟语音
    clean += np.sin(2 * np.pi * 880 * t).astype(np.float32) * 0.1
    clean += np.sin(2 * np.pi * 1320 * t).astype(np.float32) * 0.05
    
    # 保存干净音频
    test_dir = Path("data/test_noise")
    test_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(test_dir / "clean.wav"), clean, sample_rate)
    
    # 测试各种噪声
    injector = NoiseInjector(sample_rate=sample_rate)
    
    noise_configs = [
        ('white', 20),
        ('white', 10),
        ('white', 5),
        ('pink', 10),
    ]
    
    for noise_type, snr in noise_configs:
        noisy = injector.add_noise(clean, noise_type, snr)
        filename = f"{noise_type}_snr{snr}.wav"
        sf.write(str(test_dir / filename), noisy, sample_rate)
        print(f"[OK] Created: {filename}")
    
    print(f"\nTest files saved to: {test_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare noisy audio dataset")
    parser.add_argument("--clean_dir", type=str, default="data/clean",
                       help="Directory with clean audio files")
    parser.add_argument("--output_dir", type=str, default="data/noisy",
                       help="Output directory for noisy audio")
    parser.add_argument("--noise_dir", type=str, default=None,
                       help="Directory with real noise files (optional)")
    parser.add_argument("--test", action="store_true",
                       help="Run test mode")
    
    args = parser.parse_args()
    
    if args.test:
        test_noise_injection()
    else:
        create_noisy_dataset(
            clean_audio_dir=args.clean_dir,
            output_dir=args.output_dir,
            noise_dir=args.noise_dir,
        )
