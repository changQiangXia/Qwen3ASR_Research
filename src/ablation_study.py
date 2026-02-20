"""
Phase 4.3: 消融实验 (Ablation Study)
测试 Fourier Adapter 的不同配置对性能的影响
为论文提供严谨的超参数分析
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from qwen3_with_adapter import Qwen3ASRWithFourierAdapter
from comparison_experiment import ComparisonExperiment, ExperimentResult

# 设置图表样式
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


@dataclass
class AblationResult:
    """消融实验结果"""
    config_name: str
    bottleneck_dim: int
    adapter_layer: int
    inference_time: float
    memory_mb: float
    transcription: str


class AblationStudy:
    """
    消融实验主类
    
    实验设计：
    1. Bottleneck 维度影响: [32, 64, 128, 256]
    2. Adapter 位置影响: [-1 (last), -2, -4, -7] (最后1/2/4/7层)
    3. 计算效率 vs 表达能力 权衡
    """
    
    def __init__(
        self,
        model_path: str = None,
        output_dir: str = "outputs/ablation",
        device: str = "cuda:0",
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        
        print("="*60)
        print("Ablation Study: Fourier Adapter Configuration")
        print("="*60)
    
    def test_bottleneck_dimensions(
        self,
        test_audio: str,
        bottleneck_dims: List[int] = None,
    ) -> List[AblationResult]:
        """
        测试不同 bottleneck 维度的影响
        
        Args:
            test_audio: 测试音频路径
            bottleneck_dims: bottleneck 维度列表
        
        Returns:
            消融实验结果列表
        """
        if bottleneck_dims is None:
            bottleneck_dims = [32, 64, 128, 256]
        
        print(f"\n[Ablation] Testing Bottleneck Dimensions: {bottleneck_dims}")
        
        results = []
        
        for dim in tqdm(bottleneck_dims, desc="Testing bottleneck dims"):
            # 创建模型
            model = Qwen3ASRWithFourierAdapter(
                model_path=self.model_path,
                bottleneck_dim=dim,
                adapter_layer=-1,
                device=self.device,
            )
            
            # 推理测试
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.no_grad():
                result = model.transcribe(test_audio)
            end.record()
            
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000  # 转换为秒
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            # 记录结果
            ablation_result = AblationResult(
                config_name=f"bottleneck_{dim}",
                bottleneck_dim=dim,
                adapter_layer=-1,
                inference_time=elapsed,
                memory_mb=peak_memory,
                transcription=result[0].text,
            )
            results.append(ablation_result)
            
            print(f"\n  [dim={dim:3d}] Time: {elapsed:.3f}s, Memory: {peak_memory:.1f}MB, Params: {dim*2048*2/1e6:.2f}M")
            
            # 清理
            del model
            torch.cuda.empty_cache()
        
        return results
    
    def test_adapter_positions(
        self,
        test_audio: str,
        positions: List[int] = None,
        bottleneck_dim: int = 128,
    ) -> List[AblationResult]:
        """
        测试不同 Adapter 插入位置的影响
        
        Args:
            test_audio: 测试音频路径
            positions: 层位置列表（-1 表示最后一层）
            bottleneck_dim: 固定的 bottleneck 维度
        
        Returns:
            消融实验结果列表
        """
        if positions is None:
            positions = [-1, -2, -4, -7]  # 最后1、2、4、7层
        
        print(f"\n[Ablation] Testing Adapter Positions: {positions}")
        print("(Note: -1=last layer, -2=2nd last, etc.)")
        
        results = []
        
        for pos in tqdm(positions, desc="Testing positions"):
            # 创建模型
            model = Qwen3ASRWithFourierAdapter(
                model_path=self.model_path,
                bottleneck_dim=bottleneck_dim,
                adapter_layer=pos,
                device=self.device,
            )
            
            # 推理测试
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.no_grad():
                result = model.transcribe(test_audio)
            end.record()
            
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            # 记录结果
            ablation_result = AblationResult(
                config_name=f"position_{pos}",
                bottleneck_dim=bottleneck_dim,
                adapter_layer=pos,
                inference_time=elapsed,
                memory_mb=peak_memory,
                transcription=result[0].text,
            )
            results.append(ablation_result)
            
            layer_desc = f"{pos} (last)" if pos == -1 else f"{pos} ({abs(pos)} from last)"
            print(f"\n  [pos={layer_desc:15s}] Time: {elapsed:.3f}s, Memory: {peak_memory:.1f}MB")
            
            # 清理
            del model
            torch.cuda.empty_cache()
        
        return results
    
    def visualize_bottleneck_results(
        self,
        results: List[AblationResult],
        save_name: str = "ablation_bottleneck.png",
    ):
        """
        可视化 bottleneck 维度实验结果
        
        Args:
            results: 实验结果列表
            save_name: 保存文件名
        """
        dims = [r.bottleneck_dim for r in results]
        times = [r.inference_time for r in results]
        memories = [r.memory_mb for r in results]
        params = [d * 2048 * 2 / 1e6 for d in dims]  # 近似参数量
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 推理时间 vs Bottleneck 维度
        axes[0].plot(dims, times, 'o-', linewidth=2, markersize=8, color='steelblue')
        axes[0].set_xlabel("Bottleneck Dimension")
        axes[0].set_ylabel("Inference Time (s)")
        axes[0].set_title("(a) Inference Time vs Bottleneck Dim")
        axes[0].grid(True, alpha=0.3)
        
        # 显存占用 vs Bottleneck 维度
        axes[1].plot(dims, memories, 's-', linewidth=2, markersize=8, color='coral')
        axes[1].set_xlabel("Bottleneck Dimension")
        axes[1].set_ylabel("Peak Memory (MB)")
        axes[1].set_title("(b) Memory Usage vs Bottleneck Dim")
        axes[1].grid(True, alpha=0.3)
        
        # 参数量 vs Bottleneck 维度
        axes[2].plot(dims, params, '^-', linewidth=2, markersize=8, color='green')
        axes[2].set_xlabel("Bottleneck Dimension")
        axes[2].set_ylabel("Adapter Parameters (M)")
        axes[2].set_title("(c) Parameter Count vs Bottleneck Dim")
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle("Ablation Study: Bottleneck Dimension Analysis", fontsize=13, y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
        print(f"\n[OK] Saved: {save_path}")
        plt.show()
        
        return fig
    
    def visualize_position_results(
        self,
        results: List[AblationResult],
        save_name: str = "ablation_position.png",
    ):
        """
        可视化 Adapter 位置实验结果
        
        Args:
            results: 实验结果列表
            save_name: 保存文件名
        """
        positions = [r.adapter_layer for r in results]
        times = [r.inference_time for r in results]
        memories = [r.memory_mb for r in results]
        
        # 转换位置标签
        pos_labels = [f"Layer {p}" if p >= 0 else f"Last {abs(p)}" for p in positions]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 推理时间 vs 位置
        x = range(len(positions))
        axes[0].bar(x, times, color='steelblue', alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(pos_labels, rotation=45, ha='right')
        axes[0].set_ylabel("Inference Time (s)")
        axes[0].set_title("(a) Inference Time vs Adapter Position")
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 显存 vs 位置
        axes[1].bar(x, memories, color='coral', alpha=0.7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(pos_labels, rotation=45, ha='right')
        axes[1].set_ylabel("Peak Memory (MB)")
        axes[1].set_title("(b) Memory Usage vs Adapter Position")
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle("Ablation Study: Adapter Position Analysis", fontsize=13, y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
        print(f"\n[OK] Saved: {save_path}")
        plt.show()
        
        return fig
    
    def generate_report(self, bottleneck_results: List[AblationResult], position_results: List[AblationResult]):
        """生成消融实验报告"""
        print("\n" + "="*60)
        print("Ablation Study Report")
        print("="*60)
        
        print("\n[1] Bottleneck Dimension Analysis:")
        print("-" * 40)
        print(f"{'Dim':>6} | {'Params(M)':>10} | {'Time(s)':>10} | {'Memory(MB)':>12}")
        print("-" * 40)
        for r in bottleneck_results:
            params = r.bottleneck_dim * 2048 * 2 / 1e6
            print(f"{r.bottleneck_dim:>6} | {params:>10.2f} | {r.inference_time:>10.3f} | {r.memory_mb:>12.1f}")
        
        # 推荐配置
        best = min(bottleneck_results, key=lambda x: x.inference_time)
        print(f"\n[Recommendation] Fastest config: bottleneck_dim={best.bottleneck_dim}")
        
        print("\n[2] Adapter Position Analysis:")
        print("-" * 40)
        print(f"{'Position':>12} | {'Time(s)':>10} | {'Memory(MB)':>12}")
        print("-" * 40)
        for r in position_results:
            pos_str = f"Last {abs(r.adapter_layer)}" if r.adapter_layer < 0 else str(r.adapter_layer)
            print(f"{pos_str:>12} | {r.inference_time:>10.3f} | {r.memory_mb:>12.1f}")
        
        print("\n" + "="*60)


def run_ablation_demo():
    """运行消融实验演示"""
    print("="*60)
    print("Ablation Study Demo")
    print("="*60)
    
    # 准备测试音频
    test_audio = Path("data/test_m4a.wav")
    
    if not test_audio.exists():
        print(f"[ERROR] Test audio not found: {test_audio}")
        print("Please run comparison experiment first or provide test audio.")
        return
    
    # 创建消融实验
    ablation = AblationStudy()
    
    # 运行 bottleneck 维度实验
    print("\n" + "="*60)
    print("Experiment 1: Bottleneck Dimensions")
    print("="*60)
    bottleneck_results = ablation.test_bottleneck_dimensions(
        test_audio=str(test_audio),
        bottleneck_dims=[32, 64, 128, 256],
    )
    
    # 可视化
    ablation.visualize_bottleneck_results(bottleneck_results)
    
    # 运行位置实验
    print("\n" + "="*60)
    print("Experiment 2: Adapter Positions")
    print("="*60)
    position_results = ablation.test_adapter_positions(
        test_audio=str(test_audio),
        positions=[-1, -2, -4],
        bottleneck_dim=128,
    )
    
    # 可视化
    ablation.visualize_position_results(position_results)
    
    # 生成报告
    ablation.generate_report(bottleneck_results, position_results)
    
    # 保存结果
    all_results = {
        "bottleneck": [
            {
                "dim": r.bottleneck_dim,
                "time": r.inference_time,
                "memory": r.memory_mb,
                "transcription": r.transcription,
            }
            for r in bottleneck_results
        ],
        "position": [
            {
                "position": r.adapter_layer,
                "time": r.inference_time,
                "memory": r.memory_mb,
                "transcription": r.transcription,
            }
            for r in position_results
        ],
    }
    
    output_path = ablation.output_dir / "ablation_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Results saved to {output_path}")


if __name__ == "__main__":
    run_ablation_demo()
