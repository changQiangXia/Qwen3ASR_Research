"""
Phase 2.2 + 2.3: Qwen3-ASR + Fourier Adapter 完整实现
使用 Hook 机制在编码器深层插入频域适配器
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path

from fourier_adapter import FourierAdapter


class Qwen3ASRWithFourierAdapter:
    """
    Qwen3-ASR 包装类，集成 Fourier Adapter
    
    架构：
    1. 冻结 Qwen3-ASR 所有参数
    2. 在 thinker.model.layers[-1] 注册 forward hook
    3. Hook 中截获 hidden states -> FourierAdapter -> 返回适配后的特征
    4. 残差融合：adapter_output + original_hidden_states
    """
    
    def __init__(
        self,
        model_path: str = None,
        bottleneck_dim: int = 128,
        adapter_layer: int = -1,  # -1 表示最后一层
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            model_path: 模型路径
            bottleneck_dim: Adapter 瓶颈维度
            adapter_layer: 插入 adapter 的层索引，-1 表示最后一层
            device: 设备
            dtype: 数据类型
        """
        from qwen_asr import Qwen3ASRModel
        
        # 自动查找模型路径
        if model_path is None:
            model_path = self._find_model_path()
        
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.adapter_layer_idx = adapter_layer
        
        print("="*60)
        print("Qwen3-ASR with Fourier Adapter")
        print("="*60)
        print(f"Model: {model_path}")
        print(f"Adapter bottleneck: {bottleneck_dim}")
        print(f"Adapter layer: {adapter_layer} (last layer)")
        
        # 加载基础模型
        print("\nLoading base model...")
        self.base_model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )
        
        # 冻结所有参数
        self._freeze_base_model()
        
        # 获取 hidden_dim（从模型配置或探索得知）
        self.hidden_dim = 2048  # Qwen3-ASR-1.7B thinker.model 的 hidden_dim
        
        # 创建 Adapter
        self.adapter = FourierAdapter(
            hidden_dim=self.hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=0.1,
        ).to(device).to(dtype)
        
        # 注册 Hook
        self.hook_handle = None
        self._register_adapter_hook()
        
        # 统计参数量
        self._print_stats()
    
    def _find_model_path(self) -> str:
        """自动查找模型路径"""
        possible_paths = [
            "checkpoints/qwen/Qwen3-ASR-1.7B",
            "checkpoints/qwen/Qwen3-ASR-1___7B",
        ]
        for path in possible_paths:
            if Path(path).exists():
                return path
        return "Qwen/Qwen3-ASR-1.7B"
    
    def _freeze_base_model(self):
        """冻结基础模型所有参数"""
        # Qwen3ASRModel 包装了内部模型，需要访问 .model
        inner_model = self.base_model.model if hasattr(self.base_model, 'model') else self.base_model
        for param in inner_model.parameters():
            param.requires_grad = False
        print("[OK] Base model frozen")
    
    def _register_adapter_hook(self):
        """注册 Adapter Hook 到指定层"""
        # 获取 thinker.model.layers
        thinker = self.base_model.model.thinker
        text_model = thinker.model
        layers = text_model.layers
        
        # 确定层索引
        if self.adapter_layer_idx == -1:
            self.adapter_layer_idx = len(layers) - 1
        
        target_layer = layers[self.adapter_layer_idx]
        
        print(f"\nRegistering hook to layer {self.adapter_layer_idx}...")
        
        # 定义 hook 函数
        def adapter_hook(module, input, output):
            """
            Hook 函数：截获 hidden states，应用 adapter，返回残差融合结果
            
            Args:
                module: 被 hook 的层
                input: 输入元组
                output: 原始输出（可能是 tensor 或 tuple）
            
            Returns:
                适配后的输出
            """
            # 提取 hidden states
            if isinstance(output, torch.Tensor):
                hidden_states = output
                return_tuple = False
            elif isinstance(output, tuple):
                hidden_states = output[0]
                return_tuple = True
            else:
                # 其他类型，不处理
                return output
            
            # 确保是 float32 或 float16（adapter 需要）
            original_dtype = hidden_states.dtype
            
            # 应用 Adapter
            adapted_states = self.adapter(hidden_states.to(self.dtype))
            
            # 还原 dtype
            adapted_states = adapted_states.to(original_dtype)
            
            # 返回适配后的结果
            if return_tuple:
                return (adapted_states,) + output[1:]
            else:
                return adapted_states
        
        # 注册 hook
        self.hook_handle = target_layer.register_forward_hook(adapter_hook)
        print(f"[OK] Hook registered to layer {self.adapter_layer_idx}")
    
    def _print_stats(self):
        """打印统计信息"""
        # 计算参数量
        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        inner_model = self.base_model.model if hasattr(self.base_model, 'model') else self.base_model
        base_params = sum(p.numel() for p in inner_model.parameters())
        
        print("\n" + "="*60)
        print("Model Statistics")
        print("="*60)
        print(f"Base model params:     {base_params/1e6:.1f}M (frozen)")
        print(f"Adapter params:        {adapter_params/1e6:.3f}M (trainable)")
        print(f"Adapter / Base ratio:  {adapter_params/base_params*100:.4f}%")
        print("="*60)
    
    def transcribe(self, audio, language=None, **kwargs):
        """
        语音识别（带 adapter）
        
        Args:
            audio: 音频路径或 (array, sr) 元组
            language: 指定语言
            **kwargs: 其他参数
        
        Returns:
            识别结果
        """
        return self.base_model.transcribe(audio=audio, language=language, **kwargs)
    
    def transcribe_with_comparison(self, audio, language=None):
        """
        对比模式：同时输出原版和带 adapter 的结果
        
        Args:
            audio: 音频
            language: 指定语言
            
        Returns:
            dict: 包含 baseline 和 adapter 的结果
        """
        # 先移除 hook，获取 baseline 结果
        self.hook_handle.remove()
        
        with torch.no_grad():
            baseline_result = self.base_model.transcribe(audio=audio, language=language)
        
        # 重新注册 hook，获取 adapter 结果
        self._register_adapter_hook()
        
        with torch.no_grad():
            adapter_result = self.base_model.transcribe(audio=audio, language=language)
        
        return {
            "baseline": {
                "text": baseline_result[0].text,
                "language": baseline_result[0].language,
            },
            "with_adapter": {
                "text": adapter_result[0].text,
                "language": adapter_result[0].language,
            }
        }
    
    def get_trainable_params(self):
        """获取可训练参数（仅 adapter）"""
        return [p for p in self.adapter.parameters() if p.requires_grad]
    
    def save_adapter(self, path: str):
        """保存 adapter 权重"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.adapter.state_dict(), path)
        print(f"[OK] Adapter saved to {path}")
    
    def load_adapter(self, path: str):
        """加载 adapter 权重"""
        self.adapter.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[OK] Adapter loaded from {path}")
    
    def __del__(self):
        """析构时移除 hook"""
        if self.hook_handle is not None:
            self.hook_handle.remove()


def test_adapter_integration():
    """测试 Adapter 集成"""
    print("\n" + "="*60)
    print("Testing Qwen3ASRWithFourierAdapter")
    print("="*60)
    
    # 创建模型
    model = Qwen3ASRWithFourierAdapter(
        bottleneck_dim=128,  # 小维度节省显存
    )
    
    # 测试推理
    print("\n--- Testing inference ---")
    import numpy as np
    
    # 3秒静音测试
    dummy_audio = np.zeros(16000 * 3, dtype=np.float32)
    
    print("Running inference with adapter...")
    with torch.no_grad():
        result = model.transcribe(audio=(dummy_audio, 16000))
    
    print(f"Transcription: '{result[0].text}'")
    print(f"Language: {result[0].language}")
    
    print("\n[OK] Integration test passed!")
    
    return model


if __name__ == "__main__":
    test_adapter_integration()
