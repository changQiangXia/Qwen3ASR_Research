"""
Phase 3.2: 低资源微调 - 只训练 Fourier Adapter
冻结 Qwen3-ASR，只更新 Adapter 参数
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

from qwen3_with_adapter import Qwen3ASRWithFourierAdapter


class NoisyAudioDataset(Dataset):
    """
    带噪音频数据集
    用于加载 (noisy_audio, clean_transcription) 对
    """
    
    def __init__(
        self,
        audio_dir: str,
        manifest_file: str = None,
        max_duration: float = 30.0,  # 最大音频长度（秒）
    ):
        """
        Args:
            audio_dir: 音频文件目录
            manifest_file: manifest JSON 文件（包含 clean_transcription）
            max_duration: 最大音频长度
        """
        self.audio_dir = Path(audio_dir)
        self.max_duration = max_duration
        
        # 如果没有 manifest，自动扫描目录
        if manifest_file and Path(manifest_file).exists():
            with open(manifest_file, 'r', encoding='utf-8') as f:
                self.manifest = json.load(f)
        else:
            # 自动扫描 wav 文件
            audio_files = list(self.audio_dir.glob("*.wav"))
            self.manifest = [
                {"noisy_file": f.name, "clean_transcription": ""}
                for f in audio_files
            ]
        
        # 过滤掉太长的音频（避免 OOM）
        self.samples = []
        for item in self.manifest:
            audio_path = self.audio_dir / item["noisy_file"]
            if audio_path.exists():
                self.samples.append(item)
        
        print(f"[NoisyAudioDataset] Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        audio_path = self.audio_dir / item["noisy_file"]
        
        return {
            "audio_path": str(audio_path),
            "transcription": item.get("clean_transcription", ""),
            "noise_type": item.get("noise_type", "unknown"),
            "snr_db": item.get("snr_db", 0),
        }


class AdapterTrainer:
    """
    Adapter 训练器 - 只训练 Fourier Adapter
    """
    
    def __init__(
        self,
        model: Qwen3ASRWithFourierAdapter,
        learning_rate: float = 1e-4,
        device: str = "cuda:0",
    ):
        self.model = model
        self.device = device
        
        # 只优化 Adapter 参数
        self.optimizer = torch.optim.AdamW(
            model.get_trainable_params(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        # 学习率调度器
        self.scheduler = None  # 可以添加 warmup + cosine
        
        print("="*60)
        print("Adapter Trainer")
        print("="*60)
        print(f"Learning rate: {learning_rate}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.get_trainable_params())/1e6:.3f}M")
    
    def compute_loss(self, predicted_text: str, target_text: str) -> torch.Tensor:
        """
        计算损失 - 使用字符级匹配作为简单损失
        
        注意：这里使用简化版本，实际应该使用模型的 logits 计算交叉熵
        但由于 qwen-asr 的封装，我们需要更复杂的集成
        
        Args:
            predicted_text: 模型预测的文本
            target_text: 目标文本
        
        Returns:
            损失值（这里用占位符，实际实现需要模型输出 logits）
        """
        # 简化版本：这里只是占位符
        # 实际实现需要修改 qwen3_with_adapter 以获取 logits
        
        # 为了演示，返回一个虚拟损失
        return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def train_step(self, batch: Dict) -> Dict:
        """
        单步训练
        
        Args:
            batch: 数据批次
        
        Returns:
            训练指标
        """
        self.model.adapter.train()
        
        # 注意：这里需要实现实际的训练逻辑
        # 由于 qwen-asr 的封装，直接训练需要修改内部逻辑
        
        # 简化演示：只是跑一遍前向传播
        results = []
        for audio_path in batch["audio_path"]:
            result = self.model.transcribe(audio_path)
            results.append(result[0].text)
        
        # 这里应该计算损失并反向传播
        # 但由于模型封装，我们需要更深入的集成
        
        return {
            "predictions": results,
            "loss": 0.0,  # 占位符
        }
    
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """训练一个 epoch"""
        self.model.adapter.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        total_loss = 0
        
        for batch in pbar:
            metrics = self.train_step(batch)
            
            # 更新进度条
            pbar.set_postfix({"loss": metrics["loss"]})
            total_loss += metrics["loss"]
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """评估模型"""
        self.model.adapter.eval()
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                metrics = self.train_step(batch)
                all_predictions.extend(metrics["predictions"])
        
        return {
            "predictions": all_predictions,
        }
    
    def save_checkpoint(self, path: str, epoch: int):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "adapter_state_dict": self.model.adapter.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"[OK] Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.adapter.load_state_dict(checkpoint["adapter_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[OK] Checkpoint loaded from {path}")
        return checkpoint.get("epoch", 0)


def simple_finetune_demo():
    """
    简单的微调演示
    由于模型封装限制，这里展示框架，实际训练需要更深入集成
    """
    print("="*60)
    print("Adapter Fine-tuning Demo")
    print("="*60)
    
    # 创建模型
    model = Qwen3ASRWithFourierAdapter(
        bottleneck_dim=128,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    
    # 检查数据集是否存在
    noisy_dir = Path("data/noisy")
    if not noisy_dir.exists():
        print(f"\n[ERROR] Noisy dataset not found at {noisy_dir}")
        print("Please run: python src/data_preparation.py --test")
        print("Or prepare your noisy audio dataset first.")
        return
    
    # 创建数据集
    dataset = NoisyAudioDataset(
        audio_dir=str(noisy_dir),
    )
    
    if len(dataset) == 0:
        print(f"\n[ERROR] No audio files found in {noisy_dir}")
        return
    
    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 4GB 显存只能 batch=1
        shuffle=True,
        num_workers=0,
    )
    
    # 创建训练器
    trainer = AdapterTrainer(model, learning_rate=1e-4)
    
    # 运行几个示例（演示用）
    print("\n--- Running demo inference on noisy samples ---")
    
    for i, batch in enumerate(dataloader):
        if i >= 3:  # 只演示 3 个样本
            break
        
        print(f"\nSample {i+1}:")
        print(f"  Audio: {batch['audio_path'][0]}")
        print(f"  Noise: {batch['noise_type'][0]}, SNR: {batch['snr_db'][0]} dB")
        
        # 推理
        result = model.transcribe(batch["audio_path"][0])
        print(f"  Prediction: '{result[0].text}'")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("\nNote: Full training loop requires deeper integration with")
    print("qwen-asr to access model logits for loss computation.")
    print("="*60)
    
    # 保存 adapter
    model.save_adapter("checkpoints/adapter_demo.pt")


if __name__ == "__main__":
    simple_finetune_demo()
