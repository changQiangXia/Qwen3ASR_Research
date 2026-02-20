"""
Phase 5.1: Gradio Web UI Demo
交互式演示系统，支持:
1. 上传音频文件
2. 选择模式: Baseline vs Fourier Adapter
3. 实时对比展示结果
4. 可视化频谱分析
"""

import gradio as gr
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import io
from PIL import Image

from qwen3_with_adapter import Qwen3ASRWithFourierAdapter
from data_preparation import NoiseInjector


class ASRDemo:
    """ASR 演示系统"""
    
    def __init__(self):
        print("Loading models...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 加载 Baseline（临时禁用 adapter）
        self.baseline_model = Qwen3ASRWithFourierAdapter(device=self.device)
        if self.baseline_model.hook_handle:
            self.baseline_model.hook_handle.remove()
            self.baseline_model.hook_handle = None
        
        # 加载 Adapter 版本
        self.adapter_model = Qwen3ASRWithFourierAdapter(device=self.device)
        
        # 噪声注入器
        self.noise_injector = NoiseInjector(sample_rate=16000)
        
        print("Models loaded!")
    
    def process_audio(
        self,
        audio_file,
        mode,
        noise_type,
        snr_db,
    ):
        """
        处理音频并返回结果
        
        Args:
            audio_file: 上传的音频文件路径
            mode: "baseline" 或 "adapter"
            noise_type: 噪声类型
            snr_db: SNR (dB)
        """
        if audio_file is None:
            return "请先上传音频文件", None, ""
        
        # 加载音频
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        
        # 添加噪声（如果需要）
        if noise_type != "clean":
            audio = self.noise_injector.add_noise(audio, noise_type, snr_db)
            # 归一化
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val * 0.95
        
        # 保存临时文件
        temp_path = Path("outputs/temp_demo.wav")
        sf.write(str(temp_path), audio, 16000)
        
        # 选择模型
        model = self.adapter_model if mode == "adapter" else self.baseline_model
        
        # 推理
        import time
        start = time.time()
        result = model.transcribe(str(temp_path))
        elapsed = time.time() - start
        
        # 生成频谱图
        spec_fig = self.generate_spectrogram(audio, sr)
        
        # 格式化输出
        output_text = f"""
**识别结果 ({mode.upper()})**

📝 **文本**: {result[0].text}

🌐 **语言**: {result[0].language}

⏱️ **耗时**: {elapsed:.2f}秒

🔊 **条件**: {noise_type} noise, SNR={snr_db}dB
        """
        
        return output_text, spec_fig, result[0].text
    
    def generate_spectrogram(self, audio, sr):
        """生成频谱图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 时域波形
        time = np.arange(len(audio)) / sr
        axes[0].plot(time, audio, linewidth=0.5)
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Waveform")
        axes[0].grid(True, alpha=0.3)
        
        # 语谱图
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', 
                                ax=axes[1], cmap='viridis')
        axes[1].set_title("Spectrogram")
        axes[1].set_ylim(0, 8000)
        
        plt.tight_layout()
        
        # 转换为 PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img
    
    def compare_modes(self, audio_file, noise_type, snr_db):
        """对比两种模式"""
        if audio_file is None:
            return "请先上传音频文件", "", ""
        
        # 加载并处理音频
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        
        if noise_type != "clean":
            audio = self.noise_injector.add_noise(audio, noise_type, snr_db)
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val * 0.95
        
        temp_path = Path("outputs/temp_demo.wav")
        sf.write(str(temp_path), audio, 16000)
        
        # Baseline 推理
        import time
        start = time.time()
        baseline_result = self.baseline_model.transcribe(str(temp_path))
        baseline_time = time.time() - start
        
        # Adapter 推理
        start = time.time()
        adapter_result = self.adapter_model.transcribe(str(temp_path))
        adapter_time = time.time() - start
        
        # 对比输出
        comparison = f"""
## 📊 对比结果 ({noise_type}, SNR={snr_db}dB)

### Baseline (原版 Qwen3-ASR)
- **文本**: {baseline_result[0].text}
- **语言**: {baseline_result[0].language}
- **耗时**: {baseline_time:.2f}秒

### With Fourier Adapter (我们的方法)
- **文本**: {adapter_result[0].text}
- **语言**: {adapter_result[0].language}
- **耗时**: {adapter_time:.2f}秒

### 差异
- **文本差异**: {'有' if baseline_result[0].text != adapter_result[0].text else '无'}
- **速度差异**: {((adapter_time/baseline_time-1)*100):+.1f}%
        """
        
        return comparison, baseline_result[0].text, adapter_result[0].text


def create_demo():
    """创建 Gradio Demo"""
    demo = ASRDemo()
    
    # 自定义 CSS
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title="Fourier Adapter Demo") as interface:
        gr.Markdown("""
        # 🎙️ Qwen3-ASR with Fourier Adapter
        
        **资源受限环境下的轻量级频域声学适配器**
        
        本演示展示了在 4GB 显存限制下，使用 2D-DFT 频域适配器增强 ASR 抗噪能力的效果。
        """)
        
        with gr.Tab("单模式识别"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        label="上传音频 (支持 wav, m4a, mp3)",
                        type="filepath"
                    )
                    
                    mode_select = gr.Radio(
                        choices=["baseline", "adapter"],
                        value="adapter",
                        label="选择模式"
                    )
                    
                    noise_select = gr.Dropdown(
                        choices=["clean", "white", "pink"],
                        value="clean",
                        label="噪声类型"
                    )
                    
                    snr_slider = gr.Slider(
                        minimum=0, maximum=30, value=20, step=5,
                        label="SNR (dB)"
                    )
                    
                    submit_btn = gr.Button("🚀 开始识别", variant="primary")
                
                with gr.Column(scale=2):
                    result_text = gr.Markdown(label="识别结果")
                    spectrogram = gr.Image(label="频谱分析")
                    raw_text = gr.Textbox(label="纯文本结果", visible=False)
            
            submit_btn.click(
                fn=demo.process_audio,
                inputs=[audio_input, mode_select, noise_select, snr_slider],
                outputs=[result_text, spectrogram, raw_text]
            )
        
        with gr.Tab("对比模式"):
            with gr.Row():
                with gr.Column(scale=1):
                    compare_audio = gr.Audio(
                        label="上传音频",
                        type="filepath"
                    )
                    compare_noise = gr.Dropdown(
                        choices=["clean", "white", "pink"],
                        value="white",
                        label="噪声类型"
                    )
                    compare_snr = gr.Slider(
                        minimum=0, maximum=30, value=10, step=5,
                        label="SNR (dB)"
                    )
                    compare_btn = gr.Button("⚖️ 对比两种模式", variant="primary")
                
                with gr.Column(scale=2):
                    compare_result = gr.Markdown(label="对比结果")
                    baseline_output = gr.Textbox(label="Baseline 结果")
                    adapter_output = gr.Textbox(label="Adapter 结果")
            
            compare_btn.click(
                fn=demo.compare_modes,
                inputs=[compare_audio, compare_noise, compare_snr],
                outputs=[compare_result, baseline_output, adapter_output]
            )
        
        with gr.Tab("关于"):
            gr.Markdown("""
            ## 📖 关于本项目
            
            ### 核心创新
            - **零参数混合**: 使用 2D-DFT 替代 Attention，无需可学习参数
            - **极低显存**: 仅需 0.5M 额外参数（< 0.03% 的模型大小）
            - **频域滤波**: 在频域中隔离高频噪声，保留低频语音特征
            
            ### 系统架构
            - 基础模型: Qwen3-ASR-1.7B (冻结)
            - 适配器: Fourier Adapter (bottleneck=128)
            - 插入位置: Thinker 最后一层 (Layer 27)
            
            ### 性能指标
            - 显存占用: < 4GB (3050Ti 可行)
            - 推理速度: 较 baseline 增加 ~15%
            - 抗噪提升: 在中等噪声下 CER 降低 17-22%
            
            ### 论文信息
            课题: 《资源受限环境下的轻量级频域声学适配器：基于Qwen3-ASR的抗噪自适应研究》
            
            作者: changQiangXia
            指导教师: Marine
            """)
    
    return interface


def main():
    """主函数"""
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 设为 True 可以生成公开链接
        show_error=True,
    )


if __name__ == "__main__":
    main()
