"""
Phase 4.2: 频域特征可视化
展示 Fourier Adapter 如何工作，为论文提供关键图表
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import librosa
import soundfile as sf
from matplotlib.gridspec import GridSpec

from qwen3_with_adapter import Qwen3ASRWithFourierAdapter
from fourier_adapter import FourierAdapter

# 设置 matplotlib 样式（论文级美观）
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


class FeatureVisualizer:
    """
    特征可视化器
    生成论文级图表展示频域处理效果
    """
    
    def __init__(self, output_dir: str = "outputs/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[FeatureVisualizer] Output: {self.output_dir}")
    
    def visualize_spectrum(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        title: str = "Spectrum",
        save_name: str = None,
    ):
        """
        可视化音频的时域波形和频域频谱
        
        Args:
            audio: 音频信号
            sample_rate: 采样率
            title: 图表标题
            save_name: 保存文件名
        """
        fig = plt.figure(figsize=(12, 4))
        gs = GridSpec(1, 3, figure=fig)
        
        # 时域波形
        ax1 = fig.add_subplot(gs[0, 0])
        time = np.arange(len(audio)) / sample_rate
        ax1.plot(time, audio, linewidth=0.5, color='steelblue')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_title(f"{title} - Time Domain")
        ax1.grid(True, alpha=0.3)
        
        # 频域频谱（FFT）
        ax2 = fig.add_subplot(gs[0, 1])
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/sample_rate)
        magnitude = np.abs(fft)
        
        # 只显示正频率
        pos_mask = freqs >= 0
        ax2.plot(freqs[pos_mask], magnitude[pos_mask], linewidth=0.5, color='coral')
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.set_title(f"{title} - Frequency Domain")
        ax2.set_xlim(0, 8000)  # 显示 0-8kHz
        ax2.grid(True, alpha=0.3)
        
        # 语谱图
        ax3 = fig.add_subplot(gs[0, 2])
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(
            D, sr=sample_rate, x_axis='time', y_axis='hz',
            ax=ax3, cmap='viridis'
        )
        ax3.set_title(f"{title} - Spectrogram")
        ax3.set_ylim(0, 8000)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            print(f"[OK] Saved: {save_path}")
        
        plt.show()
        return fig
    
    def visualize_fourier_transform_2d(
        self,
        hidden_states: torch.Tensor,
        save_name: str = "fourier_2d_heatmap.png",
    ):
        """
        可视化 2D 傅里叶变换的效果
        
        Args:
            hidden_states: 隐藏状态 [batch, seq_len, hidden_dim]
            save_name: 保存文件名
        """
        # 取第一个 batch
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[0]  # [seq_len, hidden_dim]
        
        hidden_states = hidden_states.cpu().numpy()
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 原始特征热力图
        ax1 = axes[0, 0]
        im1 = ax1.imshow(
            hidden_states.T,
            aspect='auto',
            cmap='RdBu_r',
            vmin=-np.abs(hidden_states).max(),
            vmax=np.abs(hidden_states).max(),
        )
        ax1.set_xlabel("Sequence Length")
        ax1.set_ylabel("Hidden Dimension")
        ax1.set_title("(a) Original Hidden States")
        plt.colorbar(im1, ax=ax1)
        
        # 2. 2D-FFT 频域表示
        ax2 = axes[0, 1]
        fft_2d = np.fft.fft2(hidden_states)
        fft_magnitude = np.abs(fft_2d)
        fft_magnitude = np.fft.fftshift(fft_magnitude)  # 移频，低频居中
        
        im2 = ax2.imshow(
            np.log(fft_magnitude + 1e-10).T,
            aspect='auto',
            cmap='hot',
        )
        ax2.set_xlabel("Frequency (Sequence)")
        ax2.set_ylabel("Frequency (Hidden)")
        ax2.set_title("(b) 2D-FFT Magnitude (log scale)")
        plt.colorbar(im2, ax=ax2)
        
        # 3. 频率分布（沿序列维度）
        ax3 = axes[1, 0]
        freq_along_seq = np.mean(fft_magnitude, axis=1)
        freqs = np.fft.fftfreq(len(freq_along_seq))
        freqs_shifted = np.fft.fftshift(freqs)
        freq_along_seq_shifted = np.fft.fftshift(freq_along_seq)
        
        ax3.plot(freqs_shifted, freq_along_seq_shifted, color='darkred', linewidth=1.5)
        ax3.set_xlabel("Normalized Frequency")
        ax3.set_ylabel("Average Magnitude")
        ax3.set_title("(c) Frequency Distribution (Sequence)")
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='DC (Low freq)')
        ax3.legend()
        
        # 4. 频率分布（沿隐藏维度）
        ax4 = axes[1, 1]
        freq_along_hidden = np.mean(fft_magnitude, axis=0)
        freqs_h = np.fft.fftfreq(len(freq_along_hidden))
        freqs_h_shifted = np.fft.fftshift(freqs_h)
        freq_along_hidden_shifted = np.fft.fftshift(freq_along_hidden)
        
        ax4.plot(freqs_h_shifted, freq_along_hidden_shifted, color='darkblue', linewidth=1.5)
        ax4.set_xlabel("Normalized Frequency")
        ax4.set_ylabel("Average Magnitude")
        ax4.set_title("(d) Frequency Distribution (Hidden)")
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='DC (Low freq)')
        ax4.legend()
        
        plt.suptitle("2D Fourier Transform Analysis of Hidden States", fontsize=14, y=1.00)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"[OK] Saved: {save_path}")
        plt.show()
        
        return fig
    
    def visualize_comparison_heatmap(
        self,
        results_dict: dict,
        save_name: str = "comparison_heatmap.png",
    ):
        """
        可视化对比实验结果热力图
        
        Args:
            results_dict: 包含 baseline 和 adapter 结果的字典
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 准备数据
        noise_types = list(results_dict.keys())
        metrics = ['WER', 'Inference Time', 'Memory']
        
        # Baseline 数据
        baseline_data = np.array([
            [results_dict[nt]['baseline_wer'] for nt in noise_types],
            [results_dict[nt]['baseline_time'] for nt in noise_types],
            [results_dict[nt]['baseline_mem'] for nt in noise_types],
        ])
        
        # Adapter 数据
        adapter_data = np.array([
            [results_dict[nt]['adapter_wer'] for nt in noise_types],
            [results_dict[nt]['adapter_time'] for nt in noise_types],
            [results_dict[nt]['adapter_mem'] for nt in noise_types],
        ])
        
        # Improvement (Adapter vs Baseline)
        improvement = ((baseline_data - adapter_data) / baseline_data * 100)
        
        # 绘制热力图
        im = axes[1].imshow(improvement, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
        axes[1].set_xticks(range(len(noise_types)))
        axes[1].set_xticklabels(noise_types, rotation=45, ha='right')
        axes[1].set_yticks(range(len(metrics)))
        axes[1].set_yticklabels(metrics)
        axes[1].set_title("(b) Relative Improvement (%)")
        
        # 添加数值标注
        for i in range(len(metrics)):
            for j in range(len(noise_types)):
                text = axes[1].text(j, i, f'{improvement[i, j]:.1f}%',
                                   ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=axes[1])
        
        # Baseline 热力图
        im0 = axes[0].imshow(baseline_data, cmap='YlOrRd', aspect='auto')
        axes[0].set_xticks(range(len(noise_types)))
        axes[0].set_xticklabels(noise_types, rotation=45, ha='right')
        axes[0].set_yticks(range(len(metrics)))
        axes[0].set_yticklabels(metrics)
        axes[0].set_title("(a) Baseline")
        plt.colorbar(im0, ax=axes[0])
        
        # Adapter 热力图
        im2 = axes[2].imshow(adapter_data, cmap='YlOrRd', aspect='auto')
        axes[2].set_xticks(range(len(noise_types)))
        axes[2].set_xticklabels(noise_types, rotation=45, ha='right')
        axes[2].set_yticks(range(len(metrics)))
        axes[2].set_yticklabels(metrics)
        axes[2].set_title("(c) With Fourier Adapter")
        plt.colorbar(im2, ax=axes[2])
        
        plt.suptitle("Comparison: Baseline vs Fourier Adapter", fontsize=14)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"[OK] Saved: {save_path}")
        plt.show()
        
        return fig


def demo_visualization():
    """演示可视化功能"""
    print("="*60)
    print("Feature Visualization Demo")
    print("="*60)
    
    visualizer = FeatureVisualizer()
    
    # 1. 测试音频频谱可视化
    print("\n[1/2] Generating audio spectrum visualization...")
    duration = 3
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    
    # 干净信号
    clean = np.sin(2 * np.pi * 440 * t) * 0.3
    clean += np.sin(2 * np.pi * 880 * t) * 0.2
    
    # 添加高频噪声
    noise = np.random.randn(len(t)) * 0.1
    noisy = clean + noise
    
    # 可视化
    visualizer.visualize_spectrum(
        clean, sr, "Clean Signal", "spectrum_clean.png"
    )
    visualizer.visualize_spectrum(
        noisy, sr, "Noisy Signal", "spectrum_noisy.png"
    )
    
    # 2. 2D-FFT 可视化（使用模拟的 hidden states）
    print("\n[2/2] Generating 2D-FFT visualization...")
    
    # 模拟 hidden states [seq_len=100, hidden_dim=2048]
    np.random.seed(42)
    seq_len = 100
    hidden_dim = 512  # 用 512 便于可视化
    
    # 创建有结构的特征（低频信号 + 高频噪声）
    hidden_states = np.zeros((seq_len, hidden_dim))
    for i in range(seq_len):
        # 低频成分（稳定特征）
        hidden_states[i, :100] = np.sin(np.linspace(0, 4*np.pi, 100)) * 0.5
        # 高频成分（噪声）
        hidden_states[i, 100:] = np.random.randn(hidden_dim-100) * 0.1
    
    hidden_states = torch.from_numpy(hidden_states).float().unsqueeze(0)
    
    visualizer.visualize_fourier_transform_2d(
        hidden_states, "fourier_2d_analysis.png"
    )
    
    print("\n" + "="*60)
    print("Visualization Demo Complete!")
    print(f"All figures saved to: {visualizer.output_dir}")
    print("="*60)


if __name__ == "__main__":
    demo_visualization()
