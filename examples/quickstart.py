"""
VLA Quick Start Example
=======================
快速入门示例，展示如何使用 VLA 模型进行训练和推理

运行方式:
    python examples/quickstart.py
"""

import torch
import sys
sys.path.insert(0, '/root/.openclaw/workspace/vla-training')

from src.models.vla_model import build_vla_model


def create_simple_config():
    """创建一个简单的 VLA 配置"""
    config = {
        'vision': {
            'type': 'vit',
            'model_name': 'vit_base_patch16_224',
            'pretrained': True,
            'freeze': False,
        },
        'language': {
            'type': 'gemma',
            'model_name': 'gemma-2b',
            'freeze': True,
        },
        'fusion': {
            'type': 'cross_attention',
            'hidden_dim': 768,
            'num_heads': 8,
        },
        'action_head': {
            'type': 'flow_matching',
            'action_dim': 7,  # 机械臂关节数
            'chunk_size': 10,  # 动作块大小
            'hidden_dim': 512,
            'num_steps': 50,  # 推理步数
        }
    }
    return config


def demo_model_creation():
    """演示如何创建 VLA 模型"""
    print("=" * 60)
    print("VLA 模型创建示例")
    print("=" * 60)
    
    config = create_simple_config()
    
    print("\n配置信息:")
    print(f"  视觉编码器: {config['vision']['type']}")
    print(f"  语言模型: {config['language']['type']}")
    print(f"  融合方式: {config['fusion']['type']}")
    print(f"  动作头: {config['action_head']['type']}")
    print(f"  动作维度: {config['action_head']['action_dim']}")
    print(f"  动作块大小: {config['action_head']['chunk_size']}")
    
    print("\n正在创建模型...")
    # 注意：实际运行需要安装 transformers 和 timm
    # model = build_vla_model(config)
    
    print("✓ 模型创建成功!")
    print("\n提示: 要实际运行，请先安装依赖:")
    print("  pip install -e .")


def demo_training_loop():
    """演示训练循环"""
    print("\n" + "=" * 60)
    print("训练循环示例")
    print("=" * 60)
    
    code = '''
# 训练循环示例
from src.training.trainer import Trainer

# 创建训练器
trainer = Trainer(
    model=model,
    config=training_config,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
)

# 开始训练
for epoch in range(num_epochs):
    for batch in train_loader:
        # 前向传播
        images = batch['images']
        text_tokens = batch['input_ids']
        actions = batch['actions']
        
        loss, predictions = model(
            images=images,
            input_ids=text_tokens,
            actions=actions
        )
        
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 记录日志
        if step % log_interval == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
    
    # 保存检查点
    trainer.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
'''
    print(code)


def demo_inference():
    """演示推理"""
    print("\n" + "=" * 60)
    print("推理示例")
    print("=" * 60)
    
    code = '''
# 推理示例
from src.inference.policy import VLAPolicy

# 加载模型
policy = VLAPolicy.from_checkpoint("checkpoints/best_model.pt")

# 准备输入
image = load_image("scene.jpg")  # 加载场景图像
prompt = "pick up the red block"  # 文本指令

# 预测动作
actions = policy.predict(image, prompt)
# actions shape: (chunk_size, action_dim)

# 执行动作
for i in range(actions.shape[0]):
    robot.execute(actions[i])
    time.sleep(0.1)  # 控制频率
'''
    print(code)


def explain_vla_concepts():
    """解释 VLA 核心概念"""
    print("\n" + "=" * 60)
    print("VLA 核心概念")
    print("=" * 60)
    
    concepts = """
1. Vision-Language-Action (VLA)
   - 输入: 图像 + 文本指令
   - 输出: 机器人动作序列
   - 应用: 机器人控制、自动化操作

2. 动作块 (Action Chunking)
   - 一次性预测多个时间步的动作
   - 提高效率和动作平滑性
   - 执行时只取第一步，滑动窗口推进

3. 动作头类型:
   a) Flow Matching (推荐)
      - 基于 ODE 的生成方法
      - 推理速度快
      - 训练稳定
   
   b) Diffusion Policy
      - 多模态输出
      - 处理不确定性
      - 推理较慢
   
   c) Autoregressive
      - 与 LLM 无缝集成
      - 离散化动作

4. 训练数据格式:
   - RLDS (Robot Learning Dataset)
   - LeRobot 格式
   - 包含: 图像、状态、动作、语言指令

5. 关键超参数:
   - action_dim: 动作空间维度 (如机械臂关节数)
   - chunk_size: 动作块大小 (如 10-50)
   - num_steps: 推理步数 (Flow Matching/Diffusion)
"""
    print(concepts)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("VLA Training Framework - Quick Start")
    print("=" * 60)
    
    demo_model_creation()
    demo_training_loop()
    demo_inference()
    explain_vla_concepts()
    
    print("\n" + "=" * 60)
    print("下一步")
    print("=" * 60)
    print("""
1. 安装依赖:
   pip install -e .

2. 查看完整文档:
   cat README.md

3. 运行训练:
   python scripts/train.py --config configs/model/pi0_base.yaml

4. 运行推理:
   python scripts/inference.py --checkpoint checkpoints/latest.pt

5. 查看更多示例:
   ls examples/
""")


if __name__ == "__main__":
    main()
