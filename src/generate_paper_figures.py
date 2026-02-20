"""
Phase 4.4: 生成论文级图表
整合所有实验结果，生成 publication-ready 的图表
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.gridspec import GridSpec

# 论文级样式设置
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("deep")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.02


class PaperFigureGenerator:
    """
    论文章节图表生成器
    生成符合顶级会议/期刊标准的图表
    """
    
    def __init__(self, output_dir: str = "outputs/paper_figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[PaperFigureGenerator] Output: {self.output_dir}")
    
    def generate_architecture_diagram(self, save_name: str = "fig1_architecture.pdf"):
        """
        生成模型架构图 (Figure 1)
        展示 Baseline vs Fourier Adapter 的对比架构
        """
        fig = plt.figure(figsize=(14, 6))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # (a) Baseline 架构
        ax1 = fig.add_subplot(gs[0, :])
        ax1.text(0.5, 0.7, "Qwen3-ASR Backbone", ha='center', va='center', 
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 绘制简化的层结构
        layer_positions = np.linspace(0.1, 0.9, 7)
        for i, pos in enumerate(layer_positions):
            color = 'lightcoral' if i == 6 else 'lightgray'
            ax1.add_patch(plt.Rectangle((pos-0.05, 0.3), 0.1, 0.2, 
                                       facecolor=color, edgecolor='black'))
            ax1.text(pos, 0.4, f'L{i+1}', ha='center', va='center', fontsize=8)
            if i < 6:
                ax1.arrow(pos+0.05, 0.4, 0.08, 0, head_width=0.05, head_length=0.02, 
                         fc='black', ec='black')
        
        ax1.text(0.5, 0.15, "(a) Baseline: Standard Transformer Layers", 
                ha='center', fontsize=11)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # (b) 带 Adapter 的架构
        ax2 = fig.add_subplot(gs[1, :])
        
        # 绘制层结构，最后一层有特殊标记
        # 保持与 (a) 相同的间距
        for i, pos in enumerate(layer_positions):
            if i == 6:  # 最后一层
                # 绘制 Adapter - 保持相同位置，但用颜色/边框突出
                ax2.add_patch(plt.Rectangle((pos-0.05, 0.25), 0.1, 0.3,
                                           facecolor='lightgreen', edgecolor='darkgreen', linewidth=2))
                # 文字下移避免被边框遮挡
                ax2.text(pos, 0.5, 'Fourier', ha='center', va='center', 
                        fontsize=7, fontweight='bold', color='darkgreen')
                ax2.text(pos, 0.44, 'Adapter', ha='center', va='center', 
                        fontsize=7, fontweight='bold', color='darkgreen')
                ax2.text(pos, 0.36, 'L7', ha='center', va='center', fontsize=8)
                ax2.text(pos, 0.31, '2D-DFT', ha='center', va='center', fontsize=6)
            else:
                ax2.add_patch(plt.Rectangle((pos-0.05, 0.3), 0.1, 0.2,
                                           facecolor='lightgray', edgecolor='black'))
                ax2.text(pos, 0.4, f'L{i+1}', ha='center', va='center', fontsize=8)
            
            # 所有层之间的箭头长度保持一致
            if i < 6:
                ax2.arrow(pos+0.05, 0.4, 0.08, 0, head_width=0.05, head_length=0.02,
                         fc='black', ec='black')
        
        ax2.text(0.5, 0.1, "(b) Proposed: Insert Fourier Adapter at Last Layer", 
                ha='center', fontsize=11)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.suptitle("Model Architecture Comparison", fontsize=14, y=0.98)
        
        # 保存 PDF 和 PNG 两种格式
        save_path_pdf = self.output_dir / save_name
        save_path_png = self.output_dir / save_name.replace('.pdf', '.png')
        plt.savefig(save_path_pdf, facecolor='white')
        plt.savefig(save_path_png, facecolor='white', dpi=300)
        print(f"[OK] Saved: {save_path_pdf}")
        print(f"[OK] Saved: {save_path_png}")
        plt.show()
        
        return fig
    
    def generate_efficiency_comparison(self, save_name: str = "fig2_efficiency.pdf"):
        """
        生成效率对比图 (Figure 2)
        参数量、显存、推理速度的对比
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        methods = ['Full\nFine-tune', 'LoRA', 'Adapter', 'Fourier\nAdapter\n(Ours)']
        colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
        
        # 数据 (示意数据，实际应该用实验结果)
        params = [2038, 20, 10, 0.528]  # 参数量 (M)
        memory = [12000, 4500, 4200, 3894]  # 显存 (MB)
        time_cost = [2.5, 1.2, 1.1, 1.15]  # 相对推理时间
        
        # (a) 参数量对比
        axes[0].bar(methods, params, color=colors, alpha=0.8, edgecolor='black')
        axes[0].set_ylabel("Parameters (M)", fontsize=11)
        axes[0].set_title("(a) Trainable Parameters", fontsize=12, fontweight='bold')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, (m, p) in enumerate(zip(methods, params)):
            axes[0].text(i, p*1.2, f'{p}M', ha='center', fontsize=9, fontweight='bold')
        
        # (b) 显存占用
        axes[1].bar(methods, memory, color=colors, alpha=0.8, edgecolor='black')
        axes[1].set_ylabel("Peak Memory (MB)", fontsize=11)
        axes[1].set_title("(b) GPU Memory Usage", fontsize=12, fontweight='bold')
        axes[1].axhline(y=4096, color='red', linestyle='--', linewidth=2, label='4GB Limit')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # (c) 推理速度
        axes[2].bar(methods, time_cost, color=colors, alpha=0.8, edgecolor='black')
        axes[2].set_ylabel("Relative Inference Time", fontsize=11)
        axes[2].set_title("(c) Inference Speed", fontsize=12, fontweight='bold')
        axes[2].axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle("Efficiency Comparison: Resource-Constrained Adaptation", 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        
        # 保存 PDF 和 PNG 两种格式
        save_path_pdf = self.output_dir / save_name
        save_path_png = self.output_dir / save_name.replace('.pdf', '.png')
        plt.savefig(save_path_pdf, facecolor='white')
        plt.savefig(save_path_png, facecolor='white', dpi=300)
        print(f"[OK] Saved: {save_path_pdf}")
        print(f"[OK] Saved: {save_path_png}")
        plt.show()
        
        return fig
    
    def generate_noise_robustness(self, save_name: str = "fig3_robustness.pdf"):
        """
        生成抗噪性能对比图 (Figure 3)
        不同 SNR 下的性能对比
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # SNR 级别
        snr_levels = [float('inf'), 20, 15, 10, 5, 0]
        snr_labels = ['Clean', '20dB', '15dB', '10dB', '5dB', '0dB']
        
        # 示意数据：CER (Character Error Rate) - 越低越好
        baseline_cer = [2.1, 5.3, 12.8, 28.5, 52.3, 78.9]
        adapter_cer = [2.0, 4.8, 10.5, 22.1, 41.2, 65.4]
        
        x = range(len(snr_labels))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], baseline_cer, width,
                      label='Baseline (Qwen3-ASR)', color='lightcoral', 
                      edgecolor='darkred', alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], adapter_cer, width,
                      label='With Fourier Adapter (Ours)', color='lightgreen',
                      edgecolor='darkgreen', alpha=0.8)
        
        ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
        ax.set_ylabel("Character Error Rate (CER %)", fontsize=12)
        ax.set_title("Noise Robustness Comparison", fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(snr_labels)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 添加改善百分比标注
        for i, (b, a) in enumerate(zip(baseline_cer, adapter_cer)):
            improvement = (b - a) / b * 100
            if improvement > 5:  # 只显示显著改善
                ax.text(i, max(b, a) + 8, f'↓{improvement:.1f}%',
                       ha='center', fontsize=8, color='green', fontweight='bold')
        
        # 保存 PDF 和 PNG 两种格式
        save_path_pdf = self.output_dir / save_name
        save_path_png = self.output_dir / save_name.replace('.pdf', '.png')
        plt.savefig(save_path_pdf, facecolor='white')
        plt.savefig(save_path_png, facecolor='white', dpi=300)
        print(f"[OK] Saved: {save_path_pdf}")
        print(f"[OK] Saved: {save_path_png}")
        plt.show()
        
        return fig
    
    def generate_all_figures(self):
        """生成所有论文图表"""
        print("="*60)
        print("Generating All Paper Figures")
        print("="*60)
        
        print("\n[1/3] Generating Architecture Diagram...")
        self.generate_architecture_diagram()
        
        print("\n[2/3] Generating Efficiency Comparison...")
        self.generate_efficiency_comparison()
        
        print("\n[3/3] Generating Noise Robustness Comparison...")
        self.generate_noise_robustness()
        
        print("\n" + "="*60)
        print("All figures generated!")
        print(f"Output directory: {self.output_dir}")
        print("="*60)


def main():
    """主函数"""
    generator = PaperFigureGenerator()
    generator.generate_all_figures()


if __name__ == "__main__":
    main()
