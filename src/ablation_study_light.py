"""
Phase 4.3: 消融实验 (轻量版)
只测试 bottleneck 维度，减少模型加载次数
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from fourier_adapter import FourierAdapter

# 设置图表样式
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


def test_adapter_efficiency(
    hidden_dim: int = 2048,
    bottleneck_dims: List[int] = None,
) -> Dict:
    """
    测试不同 bottleneck 维度的效率（不加载大模型）
    
    只测试 Adapter 本身的计算效率
    """
    if bottleneck_dims is None:
        bottleneck_dims = [32, 64, 128, 256, 512]
    
    print("="*60)
    print("Ablation Study: Bottleneck Dimension Efficiency")
    print("="*60)
    print(f"Hidden dim: {hidden_dim}")
    print(f"Testing bottleneck dims: {bottleneck_dims}")
    
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模拟输入 [batch=1, seq_len=50, hidden_dim=2048]
    seq_len = 50
    dummy_input = torch.randn(1, seq_len, hidden_dim).to(device)
    
    for dim in tqdm(bottleneck_dims, desc="Testing"):
        # 创建 Adapter
        adapter = FourierAdapter(
            hidden_dim=hidden_dim,
            bottleneck_dim=dim,
            dropout=0.1,
        ).to(device)
        
        # 计算参数量
        params = sum(p.numel() for p in adapter.parameters())
        
        # 预热
        with torch.no_grad():
            for _ in range(5):
                _ = adapter(dummy_input)
        
        # 测试推理时间
        torch.cuda.synchronize() if device == "cuda" else None
        start = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        
        if device == "cuda":
            start.record()
        else:
            import time
            t0 = time.time()
        
        with torch.no_grad():
            for _ in range(100):  # 100次取平均
                _ = adapter(dummy_input)
        
        if device == "cuda":
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000 / 100  # 单次时间(秒)
        else:
            elapsed = (time.time() - t0) / 100
        
        # 测试显存
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = adapter(dummy_input)
            memory = torch.cuda.max_memory_allocated() / 1024**2
        else:
            memory = 0
        
        results.append({
            "bottleneck_dim": dim,
            "parameters": params,
            "parameters_mb": params * 4 / 1024**2,  # FP32 字节数
            "inference_time_ms": elapsed * 1000,  # 毫秒
            "memory_mb": memory,
        })
        
        print(f"\n  [dim={dim:3d}] Params: {params/1e6:.3f}M, Time: {elapsed*1000:.2f}ms, Mem: {memory:.1f}MB")
        
        # 清理
        del adapter
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return results


def visualize_results(results: List[Dict], output_dir: str = "outputs/ablation"):
    """可视化消融实验结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dims = [r["bottleneck_dim"] for r in results]
    params = [r["parameters_mb"] for r in results]
    times = [r["inference_time_ms"] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 参数量
    axes[0].plot(dims, params, 'o-', linewidth=2, markersize=8, color='steelblue')
    axes[0].set_xlabel("Bottleneck Dimension")
    axes[0].set_ylabel("Parameters (MB)")
    axes[0].set_title("(a) Adapter Parameter Count")
    axes[0].grid(True, alpha=0.3)
    
    # 推理时间
    axes[1].plot(dims, times, 's-', linewidth=2, markersize=8, color='coral')
    axes[1].set_xlabel("Bottleneck Dimension")
    axes[1].set_ylabel("Inference Time (ms)")
    axes[1].set_title("(b) Adapter Inference Speed")
    axes[1].grid(True, alpha=0.3)
    
    # 计算效率 vs 容量权衡
    # 假设表达能力与 bottleneck_dim 成正比，效率与 time 成反比
    efficiency = [dim / (t + 1e-6) for dim, t in zip(dims, times)]
    axes[2].plot(dims, efficiency, '^-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel("Bottleneck Dimension")
    axes[2].set_ylabel("Efficiency Score (dim/time)")
    axes[2].set_title("(c) Capacity-Speed Trade-off")
    axes[2].grid(True, alpha=0.3)
    
    # 标记推荐配置（效率最高点）
    best_idx = efficiency.index(max(efficiency))
    axes[2].plot(dims[best_idx], efficiency[best_idx], 'r*', markersize=15, 
                label=f'Recommended: dim={dims[best_idx]}')
    axes[2].legend()
    
    plt.suptitle("Ablation Study: Bottleneck Dimension Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    
    save_path = output_dir / "ablation_bottleneck.png"
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    print(f"\n[OK] Saved: {save_path}")
    plt.show()


def generate_report(results: List[Dict]):
    """生成报告"""
    print("\n" + "="*60)
    print("Ablation Study Report")
    print("="*60)
    print(f"\n{'Dim':>6} | {'Params(M)':>10} | {'Time(ms)':>10} | {'Efficiency':>12}")
    print("-" * 50)
    
    for r in results:
        efficiency = r["bottleneck_dim"] / r["inference_time_ms"]
        print(f"{r['bottleneck_dim']:>6} | {r['parameters_mb']/4:>10.3f} | {r['inference_time_ms']:>10.2f} | {efficiency:>12.2f}")
    
    # 推荐配置
    efficiencies = [r["bottleneck_dim"] / r["inference_time_ms"] for r in results]
    best_idx = efficiencies.index(max(efficiencies))
    best = results[best_idx]
    
    print(f"\n[Recommendation]")
    print(f"  Best bottleneck_dim: {best['bottleneck_dim']}")
    print(f"  Params: {best['parameters_mb']/4:.3f}M")
    print(f"  Inference time: {best['inference_time_ms']:.2f}ms")
    print(f"  Efficiency score: {efficiencies[best_idx]:.2f}")
    
    print("="*60)


def main():
    """主函数"""
    # 运行消融实验
    results = test_adapter_efficiency(
        hidden_dim=2048,
        bottleneck_dims=[32, 64, 128, 256],
    )
    
    # 保存结果
    output_dir = Path("outputs/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "ablation_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # 可视化
    visualize_results(results, str(output_dir))
    
    # 生成报告
    generate_report(results)


if __name__ == "__main__":
    main()
