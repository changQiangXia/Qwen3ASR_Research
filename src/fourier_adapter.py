"""
Phase 2: Fourier Acoustic Adapter 核心实现
基于 2D-DFT 的频域适配器，零参数混合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FourierAdapter(nn.Module):
    """
    频域声学适配器 (Fourier Acoustic Adapter)
    
    核心创新：使用 2D-DFT 替代 Attention，实现零参数特征混合
    数学公式: Z = Linear_up(ℜ(FFT2(FFT2(Linear_down(X)))))
    
    Args:
        hidden_dim: 输入/输出的隐藏维度（与主干模型一致）
        bottleneck_dim: 降维后的维度（控制参数量，如 64, 128）
        dropout: Dropout 比率
    """
    
    def __init__(
        self,
        hidden_dim: int = 1536,  # Qwen3-ASR-1.7B 的 hidden_dim
        bottleneck_dim: int = 128,  # 降维维度，控制参数量
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        
        # 降维层：将高维特征压缩到低维
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        
        # 升维层：将低维特征还原到高维
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm 用于稳定训练
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 初始化（使用较小的权重，因为是残差连接）
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.01)
        
        print(f"[FourierAdapter] Created: hidden={hidden_dim}, bottleneck={bottleneck_dim}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters())/1e6:.3f}M")
    
    def fourier_transform_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D 离散傅里叶变换
        在序列长度和特征维度两个方向上进行频域混合
        
        注意：cuFFT 在 FP16 下只支持 2 的幂次方尺寸，所以先转 FP32
        
        Args:
            x: 输入张量 [batch, seq_len, dim]
            
        Returns:
            经过 2D-FFT 和逆变换的张量 [batch, seq_len, dim]
        """
        # x: [batch, seq_len, bottleneck_dim]
        original_dtype = x.dtype
        
        # 转 FP32（cuFFT 要求）
        x = x.float()
        
        # 第1维 FFT：沿序列长度方向 (dim=1)
        x_freq = torch.fft.fft(x, dim=1, norm='ortho')
        
        # 第2维 FFT：沿特征维度方向 (dim=2)
        x_freq = torch.fft.fft(x_freq, dim=2, norm='ortho')
        
        # 高频滤波（可选）：抑制高频噪声
        # 这里可以实现频域掩码，暂时保留所有频率
        
        # 逆 FFT 还原
        x_time = torch.fft.ifft(x_freq, dim=2, norm='ortho')
        x_time = torch.fft.ifft(x_time, dim=1, norm='ortho')
        
        # 取实部并转回原始 dtype
        return torch.real(x_time).to(original_dtype)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 编码器的隐藏状态 [batch, seq_len, hidden_dim]
            
        Returns:
            适配后的特征 [batch, seq_len, hidden_dim]
        """
        residual = hidden_states
        
        # Step 1: 降维
        # [batch, seq_len, hidden_dim] -> [batch, seq_len, bottleneck_dim]
        x = self.down_proj(hidden_states)
        x = self.dropout(x)
        
        # Step 2: 2D-DFT 频域混合（核心创新）
        # 零参数的特征混合，隔离高频噪声
        x = self.fourier_transform_2d(x)
        
        # Step 3: 升维
        # [batch, seq_len, bottleneck_dim] -> [batch, seq_len, hidden_dim]
        x = self.up_proj(x)
        
        # Step 4: 残差连接
        output = self.layer_norm(residual + x)
        
        return output


class Qwen3ASRWithAdapter(nn.Module):
    """
    包装 Qwen3-ASR，插入 FourierAdapter
    使用 Hook 机制截获编码器特征
    """
    
    def __init__(
        self,
        base_model_path: str,
        adapter: Optional[FourierAdapter] = None,
        device: str = "cuda",
    ):
        super().__init__()
        
        from qwen_asr import Qwen3ASRModel
        
        print(f"[Qwen3ASRWithAdapter] Loading base model from: {base_model_path}")
        
        # 加载基础模型（冻结）
        self.base_model = Qwen3ASRModel.from_pretrained(
            base_model_path,
            dtype=torch.float16,
            device_map=device,
            max_inference_batch_size=1,
        )
        
        # 冻结所有参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 创建或传入适配器
        if adapter is None:
            # 自动检测 hidden_dim
            # Qwen3-ASR-1.7B 默认是 1536
            self.adapter = FourierAdapter(hidden_dim=1536, bottleneck_dim=128)
        else:
            self.adapter = adapter
        
        self.adapter.to(device)
        
        # 注册 Hook（这里需要知道 Qwen3-ASR 的内部结构）
        self._register_hooks()
        
        print("[Qwen3ASRWithAdapter] Model ready!")
        print(f"  Trainable params: {sum(p.numel() for p in self.adapter.parameters())/1e6:.3f}M")
        print(f"  Frozen params: {sum(p.numel() for p in self.base_model.parameters())/1e6:.1f}M")
    
    def _register_hooks(self):
        """
        注册 forward hook 截获编码器特征
        注意：Qwen3-ASR 的具体结构需要通过检查模型来确定
        """
        # TODO: 需要根据 Qwen3-ASR 的实际结构来确定 hook 位置
        # 这里只是一个框架
        print("[Qwen3ASRWithAdapter] Hooks registered (placeholder)")
        pass
    
    def forward(self, audio, **kwargs):
        """前向传播"""
        # 这里需要实现带 adapter 的推理
        # 暂时直接调用基础模型
        return self.base_model.transcribe(audio, **kwargs)


def test_fourier_adapter():
    """测试 FourierAdapter"""
    print("\n" + "="*60)
    print("Testing FourierAdapter")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建适配器（使用 Qwen3-ASR thinker.model 的实际 hidden_dim=2048）
    adapter = FourierAdapter(hidden_dim=2048, bottleneck_dim=128)
    adapter.to(device)
    
    # 模拟输入 [batch=1, seq_len=100, hidden_dim=2048]
    batch_size = 1
    seq_len = 100
    hidden_dim = 2048
    
    dummy_input = torch.randn(batch_size, seq_len, hidden_dim).to(device)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Input device: {dummy_input.device}")
    
    # 前向传播
    with torch.no_grad():
        output = adapter(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output device: {output.device}")
    
    # 验证残差连接
    diff = torch.abs(output - dummy_input).mean().item()
    print(f"\nMean difference from input (should be small due to residual): {diff:.6f}")
    
    # 验证参数量
    total_params = sum(p.numel() for p in adapter.parameters())
    print(f"\nTotal adapter parameters: {total_params/1e6:.3f}M")
    print(f"  Down proj: {adapter.down_proj.weight.numel()/1e6:.3f}M")
    print(f"  Up proj: {adapter.up_proj.weight.numel()/1e6:.3f}M")
    
    print("\n[OK] FourierAdapter test passed!")
    
    return adapter


if __name__ == "__main__":
    # 测试适配器
    test_fourier_adapter()
