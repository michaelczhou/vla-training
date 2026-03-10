"""
VLA Web Demo
============
基于 Gradio 的交互式演示界面

运行方式:
    python web_demo/app.py
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import sys
import os

# 添加项目路径
sys.path.insert(0, '/root/.openclaw/workspace/vla-training')

# 模拟模型（实际使用时替换为真实模型）
class MockVLAModel:
    """模拟 VLA 模型用于演示"""
    
    def __init__(self):
        self.action_dim = 7
        self.chunk_size = 10
    
    def predict(self, image, text):
        """模拟预测"""
        # 生成随机动作
        actions = np.random.randn(self.chunk_size, self.action_dim) * 0.3
        
        # 根据文本调整动作
        if "pick" in text.lower():
            actions[:, 6] = 0.8  # 夹爪闭合
        elif "place" in text.lower():
            actions[:, 6] = -0.8  # 夹爪张开
        
        return actions

# 初始化模型
model = MockVLAModel()


def process_instruction(image, instruction):
    """
    处理用户指令
    
    Args:
        image: 输入图像
        instruction: 文本指令
    
    Returns:
        预测的动作序列和可视化
    """
    if image is None:
        return "请先上传图像", None
    
    if not instruction:
        return "请输入指令", None
    
    try:
        # 预测动作
        actions = model.predict(image, instruction)
        
        # 生成结果文本
        result_text = f"""
## 预测结果

**指令**: {instruction}

**动作序列**:
- 预测步数: {len(actions)}
- 动作维度: {actions.shape[1]}

**动作范围**:
- 最小值: {actions.min():.3f}
- 最大值: {actions.max():.3f}
- 均值: {actions.mean():.3f}

**首步动作** (关节角度):
"""
        
        for i, val in enumerate(actions[0]):
            result_text += f"- 关节 {i}: {val:.3f}\n"
        
        # 生成可视化
        import matplotlib.pyplot as plt
        import io
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        
        # 动作序列热力图
        im = axes[0].imshow(actions.T, cmap='RdBu_r', aspect='auto')
        axes[0].set_xlabel('时间步')
        axes[0].set_ylabel('关节')
        axes[0].set_title('预测动作序列热力图')
        plt.colorbar(im, ax=axes[0])
        
        # 每个关节的动作曲线
        for i in range(min(4, actions.shape[1])):
            axes[1].plot(actions[:, i], label=f'关节 {i}', marker='o')
        axes[1].set_xlabel('时间步')
        axes[1].set_ylabel('动作值')
        axes[1].set_title('前4个关节的动作曲线')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存为图像
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        viz_image = Image.open(buf)
        
        return result_text, viz_image
        
    except Exception as e:
        return f"错误: {str(e)}", None


def get_example_instructions():
    """获取示例指令"""
    return [
        "pick up the red block",
        "place the blue cup on the table",
        "open the drawer",
        "push the button",
        "grasp the bottle and lift it",
        "move the object to the left",
        "stack the blocks",
    ]


# 创建 Gradio 界面
def create_interface():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="VLA Demo") as demo:
        gr.Markdown("""
        # 🤖 VLA (Vision-Language-Action) Demo
        
        输入图像和文本指令，模型将预测机器人动作序列。
        
        **注意**: 当前使用模拟模型，实际部署时请替换为训练好的模型。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                gr.Markdown("### 📷 输入")
                
                image_input = gr.Image(
                    label="上传场景图像",
                    type="pil"
                )
                
                instruction_input = gr.Textbox(
                    label="输入指令",
                    placeholder="例如: pick up the red block",
                    lines=2
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 预测", variant="primary")
                    clear_btn = gr.Button("🔄 清除")
                
                # 示例指令
                gr.Markdown("### 💡 示例指令")
                examples = get_example_instructions()
                example_btns = []
                for example in examples:
                    btn = gr.Button(example, size="sm")
                    example_btns.append(btn)
                    
            with gr.Column(scale=1):
                # 输出区域
                gr.Markdown("### 📊 输出")
                
                output_text = gr.Markdown(
                    label="预测结果",
                    value="等待输入..."
                )
                
                output_viz = gr.Image(
                    label="动作可视化"
                )
        
        # 事件绑定
        submit_btn.click(
            fn=process_instruction,
            inputs=[image_input, instruction_input],
            outputs=[output_text, output_viz]
        )
        
        clear_btn.click(
            fn=lambda: (None, "等待输入...", None),
            outputs=[image_input, output_text, output_viz]
        )
        
        # 示例按钮绑定
        for btn, example in zip(example_btns, examples):
            btn.click(
                fn=lambda x=example: x,
                outputs=instruction_input
            )
        
        gr.Markdown("""
        ---
        
        ## 📖 使用说明
        
        1. **上传图像**: 拍摄或上传机器人工作场景
        2. **输入指令**: 用自然语言描述要执行的任务
        3. **点击预测**: 模型将生成动作序列
        4. **查看结果**: 动作序列将显示在右侧
        
        ## 🔧 支持的指令类型
        
        - **抓取**: "pick up the [object]"
        - **放置**: "place the [object] on [location]"
        - **移动**: "move [object] to [direction]"
        - **操作**: "open/close/push/pull [object]"
        
        ## 📚 了解更多
        
        - [项目文档](../README.md)
        - [训练教程](../notebooks/02_training.ipynb)
        - [推理部署](../notebooks/03_inference.ipynb)
        """)
    
    return demo


if __name__ == "__main__":
    print("启动 VLA Web Demo...")
    print("请访问: http://localhost:7860")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
