"""
探索 Qwen3-ASR 模型结构，找到可以插入 Adapter 的位置
"""

import torch
from qwen_asr import Qwen3ASRModel
from pathlib import Path


def find_model_path():
    """自动查找模型路径"""
    possible_paths = [
        "checkpoints/qwen/Qwen3-ASR-1.7B",
        "checkpoints/qwen/Qwen3-ASR-1___7B",
    ]
    for path in possible_paths:
        if Path(path).exists():
            return path
    return "Qwen/Qwen3-ASR-1.7B"


def explore_text_model(text_model, name="text_model"):
    """探索文本模型结构"""
    print(f"\n{'='*60}")
    print(f"Exploring {name}")
    print(f"{'='*60}")
    
    print(f"Type: {text_model.__class__.__name__}")
    
    # 列出子模块
    print("\nDirect children:")
    for child_name, child_module in text_model.named_children():
        print(f"  - {child_name}: {child_module.__class__.__name__}")
        
        # 如果是 layers，探索层数
        if child_name == 'layers' and hasattr(child_module, '__len__'):
            num_layers = len(child_module)
            print(f"    Number of layers: {num_layers}")
            
            # 显示第一层和最后一层的结构
            if num_layers > 0:
                print(f"\n    First layer (0): {child_module[0].__class__.__name__}")
                for sub_name, sub_module in child_module[0].named_children():
                    print(f"      - {sub_name}: {sub_module.__class__.__name__}")
                
                print(f"\n    Last layer ({num_layers-1}): {child_module[-1].__class__.__name__}")
                for sub_name, sub_module in child_module[-1].named_children():
                    print(f"      - {sub_name}: {sub_module.__class__.__name__}")
                
                return child_module
    
    return None


def test_hook_on_layer(model, layer, layer_name="layer"):
    """在指定层测试 hook"""
    print(f"\n{'='*60}")
    print(f"Testing hook on: {layer_name}")
    print(f"{'='*60}")
    print(f"Layer type: {layer.__class__.__name__}")
    
    captured_output = {}
    
    def hook_fn(module, input, output):
        print(f"  [Hook triggered on {layer_name}]")
        
        # 处理不同类型的输出
        if isinstance(output, torch.Tensor):
            print(f"  Output tensor shape: {output.shape}")
            captured_output['hidden_states'] = output.detach()
        elif isinstance(output, tuple):
            print(f"  Output is tuple, len={len(output)}")
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    print(f"    output[{i}] tensor shape: {o.shape}")
                    # 通常第一个是 hidden states
                    if i == 0 and 'hidden_states' not in captured_output:
                        captured_output['hidden_states'] = o.detach()
                else:
                    print(f"    output[{i}] type: {type(o)}")
        else:
            print(f"  Output type: {type(output)}")
            # 检查是否有 last_hidden_state 属性
            if hasattr(output, 'last_hidden_state'):
                print(f"  has last_hidden_state: {output.last_hidden_state.shape}")
                captured_output['hidden_states'] = output.last_hidden_state.detach()
            if hasattr(output, 'hidden_states'):
                print(f"  has hidden_states (tuple/list): {len(output.hidden_states)}")
    
    handle = layer.register_forward_hook(hook_fn)
    
    # 测试前向传播
    print("  Running test inference...")
    import numpy as np
    dummy_audio = np.zeros(16000 * 3, dtype=np.float32)  # 3秒静音
    
    with torch.no_grad():
        try:
            results = model.transcribe(audio=(dummy_audio, 16000))
            print(f"  Transcription: '{results[0].text}'")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    handle.remove()
    
    if 'hidden_states' in captured_output:
        tensor = captured_output['hidden_states']
        print(f"\n  [SUCCESS] Captured hidden states:")
        print(f"    Shape: {tensor.shape}")
        print(f"    Dtype: {tensor.dtype}")
        print(f"    Device: {tensor.device}")
        return tensor
    else:
        print(f"\n  [FAILED] No hidden states captured")
        return None


def explore_model_structure():
    """探索模型结构"""
    
    print("="*60)
    print("Exploring Qwen3-ASR Model Structure")
    print("="*60)
    
    model_path = find_model_path()
    print(f"\nModel path: {model_path}")
    
    print("\nLoading model...")
    model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    
    inner_model = model.model
    
    # 获取 thinker
    if not hasattr(inner_model, 'thinker'):
        print("No thinker found!")
        return
    
    thinker = inner_model.thinker
    
    # 探索 thinker.model (文本模型)
    if hasattr(thinker, 'model'):
        text_model = thinker.model
        layers = explore_text_model(text_model, "thinker.model")
        
        if layers is not None and len(layers) > 0:
            # 测试在最后一层 hook
            last_layer_idx = len(layers) - 1
            last_layer = layers[last_layer_idx]
            
            captured = test_hook_on_layer(
                model, 
                last_layer, 
                layer_name=f"thinker.model.layers[{last_layer_idx}] (last layer)"
            )
            
            if captured is not None:
                print("\n" + "="*60)
                print("RECOMMENDATION")
                print("="*60)
                print(f"Best hook location: thinker.model.layers[{last_layer_idx}]")
                print(f"  - This is the LAST transformer layer")
                print(f"  - Hidden states shape: {captured.shape}")
                print(f"  - Hidden dim: {captured.shape[-1]} (for FourierAdapter)")
                print(f"\nInsertion strategy:")
                print(f"  1. Hook: thinker.model.layers[{last_layer_idx}]")
                print(f"  2. Apply FourierAdapter to hidden states")
                print(f"  3. Return adapted hidden states")
    
    # 也探索 audio_tower
    if hasattr(thinker, 'audio_tower'):
        print("\n" + "="*60)
        print("Exploring Audio Tower (optional)")
        print("="*60)
        audio_tower = thinker.audio_tower
        print(f"Type: {audio_tower.__class__.__name__}")
        
        # 可以探索 audio_tower 的结构
        print("\nDirect children:")
        for child_name, child_module in audio_tower.named_children():
            print(f"  - {child_name}: {child_module.__class__.__name__}")
    
    print("\n" + "="*60)
    print("Exploration Complete")
    print("="*60)


if __name__ == "__main__":
    explore_model_structure()
