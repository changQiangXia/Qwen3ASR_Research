"""
生成测试音频用于 Baseline 测试
"""

import numpy as np
import soundfile as sf
from pathlib import Path

def create_test_audio():
    """生成测试音频文件"""
    
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # 1. 创建静音测试音频（最简单，用于验证流程）
    sample_rate = 16000
    duration = 3  # 3秒
    
    # 静音
    silence = np.zeros(int(sample_rate * duration), dtype=np.float32)
    sf.write(output_dir / "test_silence.wav", silence, sample_rate)
    print(f"[OK] 生成静音音频: {duration}s")
    
    # 2. 创建正弦波测试音频（有实际音频内容）
    freq = 440  # A4 音调
    t = np.linspace(0, duration, int(sample_rate * duration))
    sine_wave = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
    sf.write(output_dir / "test_sine.wav", sine_wave, sample_rate)
    print(f"[OK] 生成正弦波音频: {duration}s, {freq}Hz")
    
    # 3. 添加噪声的正弦波（模拟带噪语音）
    noise = np.random.normal(0, 0.05, len(sine_wave)).astype(np.float32)
    noisy_audio = (sine_wave + noise).astype(np.float32)
    # 归一化
    noisy_audio = noisy_audio / np.max(np.abs(noisy_audio)) * 0.5
    sf.write(output_dir / "test_noisy.wav", noisy_audio, sample_rate)
    print(f"[OK] 生成带噪音频: {duration}s")
    
    print(f"\n测试音频已保存到 data/ 目录")
    print("注意: 这些音频仅用于测试模型能否正常推理")
    print("如需真实语音识别测试，请提供实际语音文件")

if __name__ == "__main__":
    create_test_audio()
