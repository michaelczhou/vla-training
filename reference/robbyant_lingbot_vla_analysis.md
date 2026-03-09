# LingBot-VLA 详细解读

## 1. 代码框架和项目结构

### 1.1 项目概述

LingBot-VLA 是一个实用的 VLA 基础模型，由 Robbyant 团队开发。其核心特点包括：
- **大规模预训练数据**: 20,000 小时真实世界数据，来自 9 种流行的双臂机器人配置
- **强大性能**: 在仿真和真实世界基准测试中明显优于竞争对手
- **训练效率**: 相比现有 VLA 代码库提升 1.5~2.8 倍训练速度
- **深度支持**: 可选的深度信息蒸馏版本

### 1.2 目录结构

```
lingbot-vla/
├── lingbotvla/                 # 核心代码
│   ├── models/                 # 模型定义
│   │   ├── vla/                # VLA 模型
│   │   │   ├── modeling_lingbotvla.py  # 主模型
│   │   │   ├── vision_models/  # 视觉模型
│   │   │   │   ├── lingbot-depth/  # 深度模型
│   │   │   │   └── MoGe/       # MoGe 模型
│   │   │   ├── vlm_adapter.py  # VLM 适配器
│   │   │   ├── flow_head.py    # 流匹配头
│   │   │   └── fusion_module.py # 融合模块
│   │   └── tokenizer/          # 分词器
│   │       └── fast_tokenizer.py
│   │
│   ├── data/                   # 数据模块
│   │   ├── vla_data/           # VLA 数据
│   │   │   ├── dataset.py      # 数据集类
│   │   │   ├── transforms.py   # 数据变换
│   │   │   └── norm_stats.py   # 归一化统计
│   │   └── lerobot_dataset.py  # LeRobot 数据集
│   │
│   ├── training/               # 训练模块
│   │   ├── trainer.py          # 训练器
│   │   ├── optimizer.py        # 优化器
│   │   ├── loss.py             # 损失函数
│   │   └── checkpoint.py       # 检查点管理
│   │
│   ├── inference/              # 推理模块
│   │   ├── policy.py           # 策略接口
│   │   └── deploy.py           # 部署工具
│   │
│   └── utils/                  # 工具函数
│       ├── logging.py
│       ├── visualization.py
│       └── distributed.py      # 分布式训练
│
├── scripts/                    # 脚本
│   ├── download_hf_model.py    # 下载模型
│   ├── train.sh                # 训练脚本
│   └── eval.py                 # 评估脚本
│
├── configs/                    # 配置
│   ├── vla/                    # VLA 配置
│   │   └── robotwin_50.yaml    # RoboTwin 配置
│   └── model/                  # 模型配置
│
├── experiment/                 # 实验
│   └── robotwin/               # RoboTwin 实验
│       └── README.md
│
├── deploy/                     # 部署
│   └── lingbot_robotwin_policy.py
│
├── assets/                     # 资源
│   ├── norm_stats/             # 归一化统计
│   └── LingBot-VLA.pdf         # 技术报告
│
└── requirements.txt            # 依赖
```

### 1.3 核心模块说明

**lingbotvla/models/vla/modeling_lingbotvla.py** - 主模型:
```python
"""
LingBot-VLA 模型
基于 Qwen2.5-VL + 流匹配头 + 可选深度蒸馏
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Optional

class LingBotVLA(nn.Module):
    """
    LingBot-VLA 模型
    
    架构:
    - Qwen2.5-VL 主干 (3B 或 7B)
    - 可选深度蒸馏 (MoGe + LingBot-Depth)
    - 流匹配动作头
    - FSDP2 分布式训练支持
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        action_dim: int = 14,
        max_action_dim: int = 75,
        max_state_dim: int = 75,
        use_depth: bool = False,
        post_training: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.use_depth = use_depth
        self.post_training = post_training
        
        # Qwen2.5-VL 主干
        self.vlm = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        vlm_hidden = self.vlm.config.hidden_size
        
        # 分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        
        # 深度蒸馏模块 (可选)
        if use_depth:
            self.depth_module = DepthDistillationModule(
                moge_path=kwargs.get("moge_path"),
                morgbd_path=kwargs.get("morgbd_path")
            )
            # 深度投影
            self.depth_proj = nn.Linear(1024, vlm_hidden)
        
        # VLM 投影
        self.vlm_proj = nn.Linear(vlm_hidden, 1024)
        
        # 流匹配动作头
        self.flow_head = FlowMatchingHead(
            input_dim=1024,
            action_dim=max_action_dim,
            action_horizon=50,  # 默认 50 步
            hidden_dim=1024,
            num_blocks=8
        )
        
        # 状态投影
        self.state_proj = nn.Linear(max_state_dim, 1024)
    
    def forward(
        self,
        images: torch.Tensor,       # [B, N, C, H, W]
        text: str,
        actions: Optional[torch.Tensor] = None,  # [B, T, D]
        timestep: Optional[torch.Tensor] = None,  # [B]
        depth_images: Optional[torch.Tensor] = None,  # [B, N, C, H, W]
        train: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: RGB 图像
            text: 语言指令
            actions: 真实动作 (训练时)
            timestep: 流匹配时间步 (训练时)
            depth_images: 深度图像 (可选)
            train: 是否训练模式
        
        Returns:
            outputs: 包含 loss (训练) 或 actions (推理)
        """
        # 1. 编码视觉 (RGB + 可选深度)
        vlm_features = self.encode_vision(images, depth_images)
        
        # 2. 编码语言
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=24  # tokenizer_max_length
        ).to(images.device)
        
        lang_features = self.vlm.language_model(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        ).last_hidden_state
        
        # 3. 融合特征
        fused_features = self.fuse(vlm_features, lang_features)
        
        if train and actions is not None:
            # 训练：流匹配
            loss = self._flow_matching_train(
                fused_features, actions, timestep
            )
            return {"loss": loss}
        else:
            # 推理：采样动作
            actions = self._flow_matching_sample(
                fused_features, num_steps=10
            )
            # 截断到实际 action_dim
            actions = actions[:, :, :self.action_dim]
            return {"actions": actions}
    
    def encode_vision(
        self,
        images: torch.Tensor,
        depth_images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """编码视觉特征"""
        # RGB 编码
        vlm_outputs = self.vlm.vision_encoder(images)
        vision_features = vlm_outputs.last_hidden_state
        
        # 深度编码 (可选)
        if self.use_depth and depth_images is not None:
            depth_features = self.depth_module(depth_images)
            depth_proj = self.depth_proj(depth_features)
            
            # 融合 RGB 和深度
            vision_features = vision_features + depth_proj
        
        return vision_features
    
    def fuse(
        self,
        vision_features: torch.Tensor,
        lang_features: torch.Tensor
    ) -> torch.Tensor:
        """融合视觉和语言特征"""
        # 简单拼接 + 投影
        # 也可以使用 Cross-Attention
        
        # 池化视觉特征
        vision_pooled = vision_features.mean(dim=1)
        
        # 池化语言特征
        lang_pooled = lang_features.mean(dim=1)
        
        # 拼接
        combined = torch.cat([vision_pooled, lang_pooled], dim=-1)
        
        # 投影
        fused = self.vlm_proj(combined)
        
        return fused
    
    def _flow_matching_train(
        self,
        condition: torch.Tensor,
        actions: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """流匹配训练"""
        # 归一化动作
        actions_norm = self._normalize_actions(actions)
        
        # 预测速度
        velocity = self.flow_head(actions_norm, timestep, condition)
        
        # 目标速度
        # 流匹配：目标速度 = x_1 - x_0
        # 这里 x_0 是噪声，x_1 是真实动作
        # 简化：直接使用动作作为目标
        target_velocity = actions_norm
        
        # L1 流匹配损失
        loss = nn.functional.l1_loss(velocity, target_velocity)
        
        return loss
    
    @torch.no_grad()
    def _flow_matching_sample(
        self,
        condition: torch.Tensor,
        num_steps: int = 10
    ) -> torch.Tensor:
        """流匹配采样"""
        actions = self.flow_head.sample(condition, num_steps=num_steps)
        
        # 反归一化
        actions = self._denormalize_actions(actions)
        
        return actions
    
    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """归一化动作"""
        # 使用预计算的归一化统计
        # 从 norm_stats.json 加载
        pass
    
    def _denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """反归一化动作"""
        pass
```

**lingbotvla/models/vla/vision_models/lingbot-depth/** - 深度蒸馏模块:
```python
"""
深度蒸馏模块
使用 MoGe 和 LingBot-Depth 进行深度信息蒸馏
"""

import torch
import torch.nn as nn

class DepthDistillationModule(nn.Module):
    """
    深度蒸馏模块
    
    架构:
    - MoGe: 几何编码器
    - LingBot-Depth: 深度估计
    - Query-based 蒸馏
    """
    
    def __init__(
        self,
        moge_path: str,
        morgbd_path: str,
        num_task_tokens: int = 8,
        dim_out: int = 1024,
        **kwargs
    ):
        super().__init__()
        
        self.num_task_tokens = num_task_tokens
        
        # 加载 MoGe
        self.moge = self._load_moge(moge_path)
        
        # 加载 LingBot-Depth
        self.morgbd = self._load_morgbd(morgbd_path)
        
        # 可学习任务 token
        self.task_tokens = nn.Parameter(
            torch.randn(1, num_task_tokens, dim_out)
        )
        
        # 投影
        self.proj = nn.Linear(1024, dim_out)
        
        # 对比学习
        self.use_contrastive = kwargs.get("use_contrastive", True)
        if self.use_contrastive:
            self.contrastive_head = nn.Linear(dim_out, 256)
    
    def forward(
        self,
        depth_images: torch.Tensor
    ) -> torch.Tensor:
        """
        编码深度图像
        
        Args:
            depth_images: [B, N, C, H, W]
        
        Returns:
            depth_features: [B, num_task_tokens, dim_out]
        """
        B, N, C, H, W = depth_images.shape
        
        # MoGe 编码
        moge_features = self.moge(
            depth_images.reshape(B * N, C, H, W)
        )
        
        # LingBot-Depth 编码
        morgbd_features = self.morgbd(
            depth_images.reshape(B * N, C, H, W)
        )
        
        # 融合
        fused = moge_features + morgbd_features
        
        # 投影
        fused = self.proj(fused)
        
        # 添加任务 token
        task_tokens = self.task_tokens.expand(B * N, -1, -1)
        fused = fused + task_tokens
        
        # 重塑回 [B, N, ...]
        fused = fused.reshape(B, N, self.num_task_tokens, -1)
        
        # 池化
        depth_features = fused.mean(dim=[1, 2])
        
        return depth_features
    
    def contrastive_loss(
        self,
        depth_features: torch.Tensor,
        rgb_features: torch.Tensor
    ) -> torch.Tensor:
        """对比学习损失"""
        if not self.use_contrastive:
            return 0
        
        # 投影
        depth_proj = self.contrastive_head(depth_features)
        rgb_proj = self.contrastive_head(rgb_features)
        
        # 归一化
        depth_proj = nn.functional.normalize(depth_proj, dim=-1)
        rgb_proj = nn.functional.normalize(rgb_proj, dim=-1)
        
        # 对比损失 (InfoNCE)
        logits = torch.matmul(depth_proj, rgb_proj.t())
        labels = torch.arange(len(logits), device=logits.device)
        
        loss = nn.functional.cross_entropy(logits, labels)
        
        return loss
```

## 2. 解决问题的流程和方法

### 2.1 核心问题定义

LingBot-VLA 解决的核心问题是：**如何构建一个实用的、高效的 VLA 基础模型，能够在大规模真实世界数据上预训练，并在多种机器人平台上实现 SOTA 性能？**

### 2.2 技术路线

```
┌─────────────────────────────────────────────────────────────┐
│                  LingBot-VLA 技术路线                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 大规模数据收集                                           │
│     └─► 20,000 小时真实世界数据                               │
│     └─► 9 种双臂机器人配置                                    │
│                                                             │
│  2. 高效训练架构                                             │
│     └─► FSDP2 分布式训练                                     │
│     └─► torch.compile 加速                                   │
│     └─► 1.5~2.8x 训练速度提升                                 │
│                                                             │
│  3. 深度信息蒸馏                                             │
│     └─► MoGe 几何编码                                        │
│     └─► LingBot-Depth 深度估计                               │
│     └─► Query-based 蒸馏                                     │
│                                                             │
│  4. 流匹配动作生成                                           │
│     └─► L1 流匹配损失                                        │
│     └─► 快速推理 (10 步)                                      │
│                                                             │
│  5. 多平台评估                                               │
│     └─► GM-100 基准 (3 种机器人)                               │
│     └─► RoboTwin 2.0 仿真                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 数据准备流程

**步骤 1: 数据收集**
- 20,000 小时真实世界遥操作数据
- 9 种双臂机器人配置
- 多样化任务和操作场景

**步骤 2: 转换为 LeRobot 格式**
```python
# 参考 RoboTwin2.0 数据准备
# experiment/robotwin/README.md

def prepare_robotwin_data(raw_dir, output_dir):
    """准备 RoboTwin 数据"""
    tasks = [
        "open_microwave",
        "click_bell",
        "stack_blocks_three",
        "place_shoe",
        "put_object_cabinet"
    ]
    
    for task in tasks:
        # 加载原始数据
        episodes = load_task_data(raw_dir / task)
        
        # 转换为 LeRobot 格式
        convert_to_lerobot(episodes, output_dir / task)
    
    # 合并所有任务数据
    merge_datasets(output_dir, output_dir / "merged")
    
    return output_dir / "merged"
```

**步骤 3: 计算归一化统计**
```python
# 使用提供的归一化统计
# assets/norm_stats/robotwin_50.json

norm_stats = {
    "action": {
        "mean": [...],
        "std": [...],
        "q01": [...],
        "q99": [...]
    },
    "observation/state": {
        "mean": [...],
        "std": [...],
        "q01": [...],
        "q99": [...]
    }
}
```

### 2.4 训练流程

**训练命令** (不带深度):
```bash
bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/robotwin_load20000h.yaml \
  --model.model_path /path/to/LingBot-VLA \
  --data.train_path /path/to/mixed_robotwin_5tasks \
  --train.output_dir /path/to/lingbot_robotwin5tasks/ \
  --model.tokenizer_path /path/to/Qwen2.5-VL-3B-Instruct \
  --train.micro_batch_size ${your_batch_size} \
  --train.global_batch_size ${your_batch_size * your_gpu_num}
```

**训练命令** (带深度):
```bash
bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/robotwin_load20000h_depth.yaml \
  --model.model_path /path/to/LingBot-VLA-Depth \
  --data.train_path /path/to/mixed_robotwin_5tasks \
  --train.output_dir /path/to/lingbot_depth_robotwin5tasks \
  --model.tokenizer_path /path/to/Qwen2.5-VL-3B-Instruct \
  --model.moge_path /path/to/moge2-vitb-normal.pt \
  --model.morgbd_path /path/to/LingBot-Depth-Pretrained \
  --train.micro_batch_size ${your_batch_size} \
  --train.global_batch_size ${your_batch_size * your_gpu_num}
```

### 2.5 推理部署

**部署命令**:
```bash
export QWEN25_PATH=path_to_Qwen2.5-VL-3B-Instruct

python -m deploy.lingbot_robotwin_policy \
  --model_path path_to_your_model \
  --use_length 50 \
  --port port
```

**Python API**:
```python
from lingbotvla.inference.policy import LingBotPolicy

# 加载策略
policy = LingBotPolicy.from_pretrained(
    model_path="/path/to/model",
    tokenizer_path="/path/to/Qwen2.5-VL-3B-Instruct",
    device="cuda"
)

# 运行推理
actions = policy.predict(
    images=observation["images"],
    text="pick up the cup"
)

# 执行动作
robot.execute(actions[0])
```

## 3. 模型架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                   LingBot-VLA 架构                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   RGB        │    │   Depth      │    │   Text       │  │
│  │   Images     │    │   Images     │    │   Prompt     │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │          │
│         ▼                   ▼                   │          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────┴───────┐  │
│  │   Qwen2.5-VL │    │   MoGe +     │    │   Language   │  │
│  │   Vision     │    │   LingBot-   │    │   Encoder    │  │
│  │   Encoder    │    │   Depth      │    │   (Qwen)     │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │          │
│         │            ┌──────▼───────┐           │          │
│         │            │   Contrastive│           │          │
│         │            │   Loss       │           │          │
│         │            └──────┬───────┘           │          │
│         │                   │                   │          │
│         └────────┬──────────┘                   │          │
│                  ▼                              │          │
│         ┌────────────────┐                      │          │
│         │   Fusion       │◄─────────────────────┘          │
│         │   (Concat+MLP) │                                 │
│         └────────┬───────┘                                 │
│                  │                                         │
│                  ▼                                         │
│         ┌────────────────┐                                 │
│         │   Flow Head    │                                 │
│         │   (L1 Loss)    │                                 │
│         └────────┬───────┘                                 │
│                  │                                         │
│                  ▼                                         │
│         ┌────────────────┐                                 │
│         │   Actions      │                                 │
│         └────────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Qwen2.5-VL 主干

**架构特点**:
- 基于 Qwen2.5-VL-3B-Instruct
- 强大的视觉 - 语言理解能力
- 支持中文和英文

**配置**:
```yaml
model:
  model_path: /path/to/lingbot_vla_checkpoint
  tokenizer_path: /path/to/Qwen2.5-VL-3B-Instruct
  post_training: true
```

### 3.3 深度蒸馏模块

**架构**:
```python
class DepthDistillationConfig:
    """深度蒸馏配置"""
    
    mode: str = 'query'  # Query-based 蒸馏
    num_task_tokens: int = 8
    use_image_tokens: bool = True
    use_task_tokens: bool = False
    use_text_tokens: bool = False
    use_contrastive: bool = True
    contrastive_loss_weight: float = 0.3
    depth_loss_weight: float = 0.002
    
    # VLM 投影
    dim_out: int = 2048
    image_token_size: int = 8
    image_input_size: int = 224
    
    # 深度编码器
    model_type: str = 'MoRGBD'
    moge_path: str = '/path/to/moGe-2-vitb-normal'
    morgbd_path: str = '/path/to/LingBot-Depth'
    num_layers: int = 1
    num_heads: int = 4
    dim_head: int = 32
    ff_mult: int = 1
    num_backbone_tokens: int = 256
    token_size: int = 16
    dim_out: int = 1024
    input_size: int = 224
```

**对比学习损失**:
```python
def contrastive_loss(
    depth_features: torch.Tensor,
    rgb_features: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    对比学习损失 (InfoNCE)
    
    目标：使深度特征和 RGB 特征在嵌入空间中对齐
    """
    # 投影
    depth_proj = F.normalize(depth_features, dim=-1)
    rgb_proj = F.normalize(rgb_features, dim=-1)
    
    # 相似度矩阵
    logits = torch.matmul(depth_proj, rgb_proj.t()) / temperature
    
    # 标签
    labels = torch.arange(len(logits), device=logits.device)
    
    # 交叉熵损失
    loss = F.cross_entropy(logits, labels)
    
    return loss
```

### 3.4 流匹配动作头

**配置**:
```python
class FlowMatchingConfig:
    """流匹配配置"""
    
    input_dim: int = 1024
    action_dim: int = 14  # 实际动作维度
    max_action_dim: int = 75  # 最大动作维度 (支持多种机器人)
    action_horizon: int = 50  # 动作块大小
    hidden_dim: int = 1024
    num_blocks: int = 8
    num_heads: int = 16
    time_embed_dim: int = 256
```

**实现**:
```python
class FlowMatchingHead(nn.Module):
    """流匹配动作头"""
    
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        
        self.config = config
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.time_embed_dim),
            nn.Linear(config.time_embed_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # 动作嵌入
        self.action_embed = nn.Linear(
            config.max_action_dim * config.action_horizon,
            config.hidden_dim
        )
        
        # 条件投影
        self.cond_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # 流匹配块
        self.flow_blocks = nn.ModuleList([
            FlowMatchingBlock(config.hidden_dim, config.num_heads)
            for _ in range(config.num_blocks)
        ])
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.max_action_dim * config.action_horizon)
        )
    
    def forward(
        self,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """预测速度场"""
        B = actions.shape[0]
        
        # 嵌入
        t_emb = self.time_mlp(timestep)
        a_emb = self.action_embed(actions.reshape(B, -1))
        c_emb = self.cond_proj(condition)
        
        # 合并
        x = a_emb + t_emb + c_emb
        
        # 通过流匹配块
        for block in self.flow_blocks:
            x = block(x)
        
        # 预测速度
        velocity = self.output_head(x)
        velocity = velocity.reshape(
            B, self.config.action_horizon, self.config.max_action_dim
        )
        
        return velocity
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_steps: int = 10
    ) -> torch.Tensor:
        """采样动作 (Euler 积分)"""
        B = condition.shape[0]
        device = condition.device
        
        # 从噪声开始
        actions = torch.randn(
            B, self.config.action_horizon, self.config.max_action_dim,
            device=device
        )
        
        # Euler 积分
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.ones(B, device=device) * (1 - i * dt)
            
            # 预测速度
            velocity = self.forward(actions, t, condition)
            
            # Euler 步
            actions = actions - dt * velocity
        
        return actions
```

## 4. 训练过程和原理

### 4.1 训练配置

**完整配置** (`configs/vla/robotwin_load20000h.yaml`):
```yaml
model:
  model_path: "path/to/lingbot_vla_checkpoint"
  tokenizer_path: "path/to/Qwen2.5-VL-3B-Instruct"
  post_training: true
  adanorm_time: true
  old_adanorm: true

data:
  datasets_type: vla
  data_name: robotwin_5_new
  train_path: "path/to/lerobot_merged_data"
  num_workers: 8
  norm_type: bounds_99_woclip
  norm_stats_file: assets/norm_stats/robotwin_50.json

train:
  output_dir: "path/to/output"
  loss_type: L1_fm  # L1 流匹配损失
  data_parallel_mode: fsdp2  # FSDP2 分布式训练
  enable_full_shard: false
  module_fsdp_enable: true
  use_compile: true  # torch.compile 加速
  use_wandb: false
  rmpad: false
  rmpad_with_pos_ids: false
  ulysses_parallel_size: 1
  freeze_vision_encoder: false  # ViT 需要优化
  tokenizer_max_length: 24
  action_dim: 14
  max_action_dim: 75
  max_state_dim: 75
  lr: 1.0e-4
  lr_decay_style: constant
  num_train_epochs: 69  # 20k 步微调
  micro_batch_size: 32
  global_batch_size: 256
  max_steps: 220000
  ckpt_manager: dcp
  save_steps: 220000
  save_epochs: 69
  enable_fp32: true
  enable_resume: true
  
  # 深度蒸馏参数 (仅深度版本)
  align_params:
    mode: 'query'
    num_task_tokens: 8
    use_image_tokens: True
    use_task_tokens: False
    use_text_tokens: False
    use_contrastive: True
    contrastive_loss_weight: 0.3
    depth_loss_weight: 0.002
```

### 4.2 损失函数

**总损失**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{flow}} + \lambda_{\text{contrastive}} \mathcal{L}_{\text{contrastive}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}}$$

**L1 流匹配损失**:
$$\mathcal{L}_{\text{flow}} = \mathbb{E}[\|v_\theta(x_t, t) - (x_1 - x_0)\|_1]$$

**实现**:
```python
def lingbot_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    contrastive_loss: torch.Tensor = 0,
    depth_loss: torch.Tensor = 0,
    contrastive_weight: float = 0.3,
    depth_weight: float = 0.002
) -> torch.Tensor:
    """
    LingBot-VLA 总损失
    
    L = L_flow + λ_contrastive * L_contrastive + λ_depth * L_depth
    """
    # L1 流匹配损失
    flow_loss = F.l1_loss(predicted_velocity, target_velocity)
    
    # 总损失
    total_loss = (
        flow_loss +
        contrastive_weight * contrastive_loss +
        depth_weight * depth_loss
    )
    
    return total_loss
```

### 4.3 分布式训练

**FSDP2 配置**:
```python
# 使用 FSDP2 进行分布式训练
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

# FSDP2 配置
fsdp_config = {
    "sharding_strategy": ShardingStrategy.FULL_SHARD,
    "cpu_offload": False,
    "mixed_precision": {
        "param_dtype": torch.float32,
        "reduce_dtype": torch.float32,
        "buffer_dtype": torch.float32
    }
}

# 包装模型
model = FSDP(model, **fsdp_config)
```

**torch.compile 加速**:
```python
# 使用 torch.compile 加速
model = torch.compile(model)
```

### 4.4 训练效率

**训练速度对比**:
| 配置 | Qwen2.5-VL-π | PaliGemma-3B-pt-224-π |
|------|-------------|----------------------|
| 8 GPUs | 1.5x | 1.6x |
| 16 GPUs | 1.8x | 1.9x |
| 32 GPUs | 2.2x | 2.3x |
| 128 GPUs | 2.5x | 2.6x |
| 256 GPUs | 2.8x | 2.7x |

LingBot-VLA 代码库相比现有 VLA 代码库实现了 1.5~2.8 倍的训练速度提升。

## 5. 数学原理

### 5.1 流匹配理论

**条件流匹配**:
$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, p_1(x_1), p_0(x_0)}\left[\|v_\theta(X_t, t) - (x_1 - x_0)\|_1\right]$$

其中：
- $t \sim \mathcal{U}(0, 1)$
- $x_0 \sim \mathcal{N}(0, I)$ (噪声)
- $x_1 \sim p_{\text{data}}$ (数据)
- $X_t = (1-t)x_0 + tx_1$

**LingBot-VLA 使用 L1 损失而非 L2**:
$$\mathcal{L}_{\text{L1}} = \mathbb{E}[\|v_\theta - (x_1 - x_0)\|_1]$$

**优势**:
- 对异常值更鲁棒
- 产生更稀疏的梯度
- 在实际任务中表现更好

### 5.2 对比学习

**InfoNCE 损失**:
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_d, z_r) / \tau)}{\sum_{j} \exp(\text{sim}(z_d, z_r^{(j)}) / \tau)}$$

其中：
- $z_d$: 深度特征
- $z_r$: RGB 特征
- $\tau$: 温度参数

### 5.3 深度蒸馏

**Query-based 蒸馏**:
$$z_{\text{fused}} = z_{\text{RGB}} + \text{TaskTokens}$$

**深度损失**:
$$\mathcal{L}_{\text{depth}} = \|z_{\text{depth}} - z_{\text{RGB}}\|_2^2$$

### 5.4 归一化策略

**bounds_99_woclip 归一化**:
```python
def normalize_bounds_99(x, q01, q99):
    """
    使用 1% 和 99% 分位数归一化，不裁剪
    
    normalized = (x - q01) / (q99 - q01) * 2 - 1
    """
    normalized = (x - q01) / (q99 - q01 + 1e-8) * 2 - 1
    return normalized
```

## 6. 性能评估

### 6.1 GM-100 基准

**3 种机器人平台**:

| 平台 | WALL-OSS | GR00T N1.6 | π0.5 | Ours w/o depth | Ours w/ depth |
|------|----------|------------|------|----------------|---------------|
| | SR / PS | SR / PS | SR / PS | SR / PS | SR / PS |
| Agibot G1 | 2.99% / 8.75% | 5.23% / 12.63% | 7.77% / 21.98% | 12.82% / 30.04% | 11.98% / 30.47% |
| AgileX | 2.26% / 8.16% | 3.26% / 10.52% | 17.20% / 34.82% | 15.50% / 36.31% | 18.93% / 40.36% |
| Galaxea R1Pro | 6.89% / 14.13% | 14.29% / 24.83% | 14.10% / 26.14% | 18.89% / 34.71% | 20.98% / 35.40% |
| **平均** | **4.05% / 10.35%** | **7.59% / 15.99%** | **13.02% / 27.65%** | **15.74% / 33.69%** | **17.30% / 35.41%** |

### 6.2 RoboTwin 2.0 基准

| 模型 | Clean | Randomized |
|------|-------|------------|
| π0.5 | 82.74% | 76.76% |
| Ours w/o depth | 86.50% | 85.34% |
| Ours w/ depth | 88.56% | 86.68% |

## 7. 总结与启示

### 7.1 LingBot-VLA 的核心创新

1. **大规模数据**:
   - 20,000 小时真实世界数据
   - 9 种双臂机器人配置
   - 多样化任务覆盖

2. **高效训练**:
   - FSDP2 分布式训练
   - torch.compile 加速
   - 1.5~2.8x 速度提升

3. **深度蒸馏**:
   - MoGe + LingBot-Depth
   - Query-based 蒸馏
   - 对比学习对齐

4. **实用设计**:
   - 支持多种机器人 (max_action_dim=75)
   - L1 流匹配损失 (更鲁棒)
   - 完整的部署工具

### 7.2 对 vla-training 的启示

1. **数据规模**:
   - 尽可能收集更多真实世界数据
   - 多样化机器人平台
   - 标准化数据格式

2. **训练效率**:
   - 使用 FSDP2 分布式训练
   - 启用 torch.compile
   - 优化数据加载

3. **深度信息**:
   - 考虑添加深度支持
   - 使用对比学习对齐 RGB 和深度
   - Query-based 蒸馏

4. **归一化**:
   - 使用分位数归一化 (bounds_99)
   - 预计算并保存统计量
   - 支持多种机器人配置

---

*参考文献*:
- LingBot-VLA GitHub: https://github.com/Robbyant/lingbot-vla
- LingBot-VLA Technical Report: https://arxiv.org/abs/2601.18692
- RoboTwin 2.0: https://github.com/robbyant/robotwin
- MoGe: https://github.com/Ruicheng/moge
- Flow Matching Paper: https://arxiv.org/abs/2210.02747
