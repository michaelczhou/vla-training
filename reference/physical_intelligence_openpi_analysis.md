# Physical Intelligence openpi 详细解读

## 1. 代码框架和项目结构

### 1.1 项目概述

openpi 是 Physical Intelligence 公司开源的机器人基础模型框架，包含三种模型：
- **π₀ (pi-zero)**: 基于流匹配的视觉 - 语言 - 动作模型 (VLA)
- **π₀-FAST**: 基于 FAST 分词器的自回归 VLA
- **π₀.₅ (pi-0.5)**: π₀的升级版，具有更好的开放世界泛化能力

**核心理念**: 通过大规模多任务、多机器人数据预训练，构建能够控制多种机器人执行多种任务的通用策略。

### 1.2 目录结构

```
openpi/
├── src/openpi/                 # 核心代码
│   ├── models/                 # 模型定义
│   │   ├── pi0_model.py        # π₀模型 (JAX)
│   │   ├── pi0_model_pytorch.py # π₀模型 (PyTorch)
│   │   ├── fast_tokenizer.py   # FAST 分词器
│   │   ├── flow_head.py        # 流匹配头
│   │   └── vlm_adapter.py      # VLM 适配器
│   │
│   ├── policies/               # 策略实现
│   │   ├── policy_config.py    # 策略配置
│   │   ├── pi0_policy.py       # π₀策略
│   │   ├── pi0fast_policy.py   # π₀-FAST 策略
│   │   └── libero_policy.py    # LIBERO 策略
│   │
│   ├── training/               # 训练模块
│   │   ├── config.py           # 训练配置
│   │   ├── trainer.py          # 训练器
│   │   ├── data_loader.py      # 数据加载
│   │   └── checkpoint.py       # 检查点管理
│   │
│   ├── data/                   # 数据处理
│   │   ├── lerobot_dataset.py  # LeRobot 数据集
│   │   ├── transforms.py       # 数据变换
│   │   └── norm_stats.py       # 归一化统计
│   │
│   ├── shared/                 # 共享工具
│   │   ├── download.py         # 模型下载
│   │   ├── array_utils.py      # 数组工具 (JAX)
│   │   └── pytorch_utils.py    # PyTorch 工具
│   │
│   └── utils/                  # 工具函数
│       ├── logging.py
│       └── visualization.py
│
├── scripts/                    # 命令行脚本
│   ├── train.py                # 训练脚本 (JAX)
│   ├── train_pytorch.py        # 训练脚本 (PyTorch)
│   ├── serve_policy.py         # 策略服务器
│   ├── compute_norm_stats.py   # 计算归一化统计
│   └── convert_jax_model_to_pytorch.py # JAX→PyTorch 转换
│
├── examples/                   # 示例
│   ├── droid/                  # DROID 机器人
│   ├── aloha_real/             # ALOHA 真实机器人
│   ├── aloha_sim/              # ALOHA 仿真
│   ├── libero/                 # LIBERO 基准
│   ├── ur5/                    # UR5 机器人
│   └── simple_client/          # 简单客户端
│
├── docs/                       # 文档
│   ├── docker.md               # Docker 配置
│   ├── remote_inference.md     # 远程推理
│   └── norm_stats.md           # 归一化统计
│
└── tests/                      # 单元测试
```

### 1.3 核心模块说明

**src/openpi/models/pi0_model.py** - π₀模型核心:
```python
"""π₀模型架构 (JAX 版本)"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Any

class Pi0Model(nn.Module):
    """
    π₀模型
    
    架构:
    - VLM 主干 (预训练视觉 - 语言模型)
    - 流匹配动作头
    - 可选的多尺度记忆 (MEM)
    """
    
    config: Dict[str, Any]
    
    @nn.compact
    def __call__(
        self,
        images: jnp.ndarray,      # [B, N, H, W, C]
        text_tokens: jnp.ndarray,  # [B, L]
        actions: jnp.ndarray,      # [B, T, D] (训练时)
        timestep: jnp.ndarray,     # [B] (训练时)
        train: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        前向传播
        
        Args:
            images: 多视角图像
            text_tokens: 文本 token IDs
            actions: 动作块 (训练时)
            timestep: 流匹配时间步 (训练时)
            train: 是否训练模式
        
        Returns:
            outputs: 包含 velocity (训练) 或 actions (推理)
        """
        # 1. 编码视觉
        vision_features = self.encode_vision(images, train=train)
        
        # 2. 编码语言
        lang_features = self.encode_language(text_tokens, train=train)
        
        # 3. 融合特征
        fused_features = self.fuse(vision_features, lang_features)
        
        # 4. 流匹配动作头
        if train:
            # 训练：预测速度场
            velocity = self.flow_head(
                actions=actions,
                timestep=timestep,
                condition=fused_features
            )
            return {"velocity": velocity}
        else:
            # 推理：采样动作
            actions = self.flow_head.sample(
                condition=fused_features,
                num_steps=10
            )
            return {"actions": actions}
```

**src/openpi/policies/policy_config.py** - 策略配置:
```python
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class PolicyConfig:
    """策略配置"""
    
    # 模型配置
    model_type: str = "pi0"  # pi0, pi0fast, pi05
    vlm_name: str = "gemma-2b"
    action_dim: int = 7
    action_horizon: int = 10
    
    # 流匹配配置
    flow_steps: int = 10
    flow_hidden_dim: int = 1024
    
    # FAST 配置 (仅 pi0fast)
    vocab_size: int = 4096
    num_quantizers: int = 32
    
    # 归一化
    norm_stats: Optional[Dict[str, Any]] = None
    
    # 设备
    device: str = "cuda"
    dtype: str = "bfloat16"
    
    @classmethod
    def from_yaml(cls, path: str) -> "PolicyConfig":
        """从 YAML 文件加载配置"""
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
```

## 2. 解决问题的流程和方法

### 2.1 核心问题定义

openpi 解决的核心问题是：**如何构建一个通用的机器人基础模型，能够继承互联网规模的语义理解能力，同时具备实时、精确的动作控制能力？**

### 2.2 技术路线

```
┌─────────────────────────────────────────────────────────────┐
│                      openpi 技术路线                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 预训练 VLM                                               │
│     └─► 继承互联网语义知识                                   │
│                                                             │
│  2. 流匹配动作头                                             │
│     └─► 连续动作生成，快速推理                               │
│                                                             │
│  3. 大规模机器人数据混合                                     │
│     └─► 10000+ 小时，8 种机器人平台                            │
│                                                             │
│  4. 知识绝缘 (π₀.₅)                                          │
│     └─► 保护 VLM 知识，提升泛化能力                           │
│                                                             │
│  5. 后训练微调                                               │
│     └─► 适配特定任务和机器人                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 数据处理流程

**步骤 1: 数据转换为 LeRobot 格式**
```python
# examples/libero/convert_libero_data_to_lerobot.py
import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def convert_libero_to_lerobot(raw_dir, output_dir):
    """将 LIBERO 数据转换为 LeRobot 格式"""
    
    # 加载原始数据
    episodes = load_libero_episodes(raw_dir)
    
    # 创建 LeRobot 数据集
    dataset = {
        "meta": {
            "episode_keys": ["episode_index"],
            "video_keys": ["observation/exterior_image_1_left", 
                          "observation/wrist_image_left"],
            "state_keys": ["observation/state"],
            "action_keys": ["action"]
        },
        "data": {}
    }
    
    # 处理每个 episode
    for i, episode in enumerate(episodes):
        # 编码视频
        encode_video(episode["images"], output_dir / f"video_{i}.mp4")
        
        # 保存状态/动作为 Parquet
        save_parquet({
            "episode_index": [i] * len(episode),
            "frame_index": list(range(len(episode))),
            "observation/state": episode["states"],
            "action": episode["actions"],
            "prompt": [episode["instruction"]] * len(episode)
        }, output_dir / f"episode_{i}.parquet")
    
    # 保存元数据
    save_meta(dataset, output_dir)
```

**步骤 2: 计算归一化统计**
```bash
# 计算数据集的归一化统计
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

**归一化统计格式** (`norm_stats.json`):
```json
{
    "action": {
        "mean": [0.1, -0.2, 0.05, ...],
        "std": [0.5, 0.6, 0.4, ...],
        "q01": [-1.2, -1.5, -1.0, ...],
        "q99": [1.3, 1.4, 1.1, ...]
    },
    "observation/state": {
        "mean": [...],
        "std": [...],
        "q01": [...],
        "q99": [...]
    }
}
```

**步骤 3: 数据加载**
```python
from openpi.data.lerobot_dataset import LeRobotDataLoader

data_loader = LeRobotDataLoader(
    dataset_path="data/libero_lerobot",
    batch_size=32,
    shuffle=True,
    norm_stats=norm_stats
)

for batch in data_loader:
    # batch 包含:
    # - images: [B, N, H, W, C]
    # - text_tokens: [B, L]
    # - actions: [B, T, D] (已归一化)
    # - timestep: [B] (随机采样)
    pass
```

### 2.4 训练流程

**JAX 训练**:
```bash
# 设置 JAX 使用更多 GPU 内存
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# 开始训练
uv run scripts/train.py pi05_libero \
  --exp-name=my_experiment \
  --overwrite
```

**PyTorch 训练**:
```bash
# 单 GPU 训练
uv run scripts/train_pytorch.py pi05_libero \
  --exp-name pytorch_test

# 多 GPU 训练 (DDP)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_pytorch.py pi05_libero \
  --exp-name pytorch_ddp_test

# 多节点训练
uv run torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=0 \
  --master_addr=192.168.1.1 \
  --master_port=29500 \
  scripts/train_pytorch.py pi05_libero \
  --exp-name multi_node_test
```

### 2.5 推理部署

**策略服务器**:
```bash
# 启动策略服务器
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

**客户端调用**:
```python
from openpi.policies import PolicyClient

# 连接到策略服务器
client = PolicyClient(host="localhost", port=8000)

# 运行推理
observation = {
    "observation/exterior_image_1_left": image1,
    "observation/wrist_image_left": image2,
    "prompt": "pick up the fork"
}

result = client.infer(observation)
actions = result["actions"]

# 执行动作
robot.execute(actions[0])
```

## 3. 模型架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                       π₀架构                                 │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐                       │
│  │   Images     │    │   Text       │                       │
│  │  (多视角)     │    │   Prompt     │                       │
│  └──────┬───────┘    └──────┬───────┘                       │
│         │                   │                                │
│         ▼                   ▼                                │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │   SigLIP     │    │   Gemma      │                       │
│  │   (ViT)      │    │   (LLM)      │                       │
│  └──────┬───────┘    └──────┬───────┘                       │
│         │                   │                                │
│         └────────┬──────────┘                                │
│                  ▼                                           │
│         ┌────────────────┐                                   │
│         │   VLM Adapter  │                                   │
│         │   (Cross-Attn) │                                   │
│         └────────┬───────┘                                   │
│                  │                                           │
│                  ▼                                           │
│         ┌────────────────┐                                   │
│         │   Flow Head    │                                   │
│         │   (速度场预测)  │                                   │
│         └────────┬───────┘                                   │
│                  │                                           │
│                  ▼                                           │
│         ┌────────────────┐                                   │
│         │   Actions      │                                   │
│         └────────────────┘                                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 流匹配动作头 (Flow Matching Head)

**核心思想**: 学习从噪声分布到数据分布的速度场，通过 ODE 积分生成动作。

**数学基础**:
- 定义概率路径：$p_t = (1-t)p_0 + tp_1$
- 学习速度场：$v_t(x) = \mathbb{E}[v_t(x) | x_t = x]$
- 训练目标：$\mathcal{L} = \mathbb{E}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$

**实现**:
```python
class FlowMatchingHead(nn.Module):
    """流匹配动作头"""
    
    def __init__(
        self,
        input_dim: int = 1024,
        action_dim: int = 7,
        action_horizon: int = 10,
        hidden_dim: int = 1024,
        num_blocks: int = 8,
        num_heads: int = 16,
        time_embed_dim: int = 256
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 动作嵌入
        self.action_embed = nn.Linear(
            action_dim * action_horizon, hidden_dim
        )
        
        # 条件投影
        self.cond_proj = nn.Linear(input_dim, hidden_dim)
        
        # 流匹配块
        self.flow_blocks = nn.ModuleList([
            FlowMatchingBlock(hidden_dim, num_heads)
            for _ in range(num_blocks)
        ])
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim * action_horizon)
        )
    
    def forward(
        self,
        actions: torch.Tensor,      # [B, T, D]
        timestep: torch.Tensor,      # [B]
        condition: torch.Tensor      # [B, L, H]
    ) -> torch.Tensor:
        """
        预测速度场
        
        Args:
            actions: 当前动作状态 (x_t)
            timestep: 时间步 t ∈ [0, 1]
            condition: VLM 条件特征
        
        Returns:
            velocity: 预测的速度场 [B, T, D]
        """
        B = actions.shape[0]
        
        # 时间嵌入
        t_emb = self.time_mlp(timestep)
        
        # 动作嵌入
        a_emb = self.action_embed(actions.reshape(B, -1))
        
        # 条件嵌入
        c_emb = self.cond_proj(condition.mean(dim=1))
        
        # 合并
        x = a_emb + t_emb + c_emb
        
        # 通过流匹配块
        for block in self.flow_blocks:
            x = block(x)
        
        # 预测速度
        velocity = self.output_head(x)
        velocity = velocity.reshape(B, self.action_horizon, self.action_dim)
        
        return velocity
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_steps: int = 10,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        采样动作 (Euler 积分)
        
        Args:
            condition: VLM 条件特征
            num_steps: 积分步数
            temperature: 采样温度
        
        Returns:
            actions: 生成的动作 [B, T, D]
        """
        B = condition.shape[0]
        device = condition.device
        
        # 从噪声开始 (t=1)
        actions = torch.randn(
            B, self.action_horizon, self.action_dim,
            device=device
        ) * temperature
        
        # Euler 积分
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.ones(B, device=device) * (1 - i * dt)
            
            # 预测速度
            velocity = self.forward(actions, t, condition)
            
            # Euler 步：x_{t-dt} = x_t - dt * v_t
            actions = actions - dt * velocity
        
        return actions


class FlowMatchingBlock(nn.Module):
    """流匹配 Transformer 块"""
    
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x
```

### 3.3 π₀.₅的知识绝缘 (Knowledge Insulation)

**核心思想**: 在微调过程中保护预训练 VLM 的知识，防止灾难性遗忘。

**实现**:
```python
class KnowledgeInsulation(nn.Module):
    """知识绝缘模块"""
    
    def __init__(
        self,
        vlm: nn.Module,
        insulation_ratio: float = 0.8
    ):
        super().__init__()
        self.vlm = vlm
        self.insulation_ratio = insulation_ratio
        
        # 冻结大部分 VLM 参数
        self._freeze_vlm()
    
    def _freeze_vlm(self):
        """冻结 VLM 参数"""
        for name, param in self.vlm.named_parameters():
            # 只解冻顶层参数
            if self._is_top_layer(name):
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def _is_top_layer(self, name: str) -> bool:
        """判断是否为顶层"""
        # 例如：只解冻最后 20% 的层
        layer_num = self._extract_layer_num(name)
        total_layers = self.vlm.config.num_hidden_layers
        return layer_num >= total_layers * (1 - self.insulation_ratio)
    
    def forward(self, images, text_tokens):
        """前向传播"""
        # VLM 编码 (大部分参数冻结)
        return self.vlm(images, text_tokens)
```

### 3.4 FAST Tokenizer (π₀-FAST)

**架构**:
```python
class FASTTokenizer(nn.Module):
    """
    Frequency-Action Space Tokenizer
    
    将连续动作转换为离散 token 序列
    """
    
    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 32,
        vocab_size: int = 4096,
        num_quantizers: int = 32
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.vocab_size = vocab_size
        self.num_quantizers = num_quantizers
        
        # DCT 变换
        self.register_buffer(
            "dct_matrix",
            self._create_dct_matrix(chunk_size)
        )
        
        # 量化器
        self.quantizers = nn.ModuleList([
            VectorQuantizer(vocab_size, action_dim)
            for _ in range(num_quantizers)
        ])
    
    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """
        编码动作为 token
        
        Args:
            actions: [B, T, D] 连续动作
        
        Returns:
            tokens: [B, num_quantizers] 离散 token
        """
        B, T, D = actions.shape
        
        # 1. DCT 变换
        actions_flat = actions.reshape(B, -1)
        freq_coeffs = self.dct_matrix @ actions_flat
        
        # 2. 量化
        tokens = []
        for i, quantizer in enumerate(self.quantizers):
            token, _ = quantizer(freq_coeffs[:, i:i+1])
            tokens.append(token)
        
        tokens = torch.cat(tokens, dim=-1)
        return tokens
    
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        解码 token 为动作
        
        Args:
            tokens: [B, num_quantizers] 离散 token
        
        Returns:
            actions: [B, T, D] 连续动作
        """
        B = tokens.shape[0]
        
        # 1. 反量化
        freq_coeffs = []
        for i, quantizer in enumerate(self.quantizers):
            coeff = quantizer.get_embedding(tokens[:, i:i+1])
            freq_coeffs.append(coeff)
        
        freq_coeffs = torch.cat(freq_coeffs, dim=-1)
        
        # 2. 逆 DCT
        actions_flat = self.dct_matrix.T @ freq_coeffs
        actions = actions_flat.reshape(B, self.chunk_size, self.action_dim)
        
        return actions
```

## 4. 训练过程和原理

### 4.1 预训练数据混合

openpi 的预训练数据包括：

| 数据来源 | 时长 | 机器人平台 |
|---------|------|-----------|
| Open X-Embodiment | ~5000h | 多种 |
| π Dataset (内部) | ~5000h | 8 种机器人 |
| - 洗衣折叠 | ~500h | 双臂 |
| - 桌子清理 | ~300h | UR5e |
| - 物品组装 | ~400h | 多种 |
| - 食品包装 | ~200h | 多种 |

### 4.2 训练配置

**π₀训练配置**:
```yaml
# configs/pi0_base.yaml
model:
  type: pi0
  vlm_name: gemma-2b
  action_dim: 7
  action_horizon: 10
  
  flow_head:
    hidden_dim: 1024
    num_blocks: 8
    num_heads: 16

training:
  batch_size: 256
  num_epochs: 100
  
  optimizer:
    type: adamw
    lr: 1e-4
    weight_decay: 0.1
    betas: [0.9, 0.95]
  
  scheduler:
    type: cosine
    warmup_steps: 5000
  
  # JAX 特定配置
  dtype: bfloat16
  ema_decay: 0.999
  
  # 混合精度
  mixed_precision: true
```

**π₀.₅训练配置** (带知识绝缘):
```yaml
# configs/pi05_base.yaml
model:
  type: pi05
  knowledge_insulation:
    enabled: true
    insulation_ratio: 0.8  # 冻结 80% 的 VLM 参数
  
training:
  # 两阶段训练
  stage1:
    max_steps: 50000
    lr: 3e-4
    freeze_vlm: true  # 完全冻结 VLM
  
  stage2:
    max_steps: 100000
    lr: 1e-4
    freeze_vlm: false  # 解冻顶层
```

### 4.3 数据增强

```python
from openpi.data.transforms import Compose, RandomCrop, ColorJitter

train_transform = Compose([
    RandomCrop(size=(224, 224)),
    ColorJitter(
        brightness=0.3,
        contrast=0.4,
        saturation=0.5,
        hue=0.08
    ),
    # 对于多视角，独立增强每个视角
    MultiViewAugmentation(
        shared_params=False  # 每个视角独立参数
    )
])
```

### 4.4 归一化策略

**动作归一化**:
```python
def normalize_action(action, norm_stats):
    """
    归一化动作
    
    使用分位数归一化 (robust to outliers)
    """
    mean = norm_stats["action"]["mean"]
    std = norm_stats["action"]["std"]
    q01 = norm_stats["action"]["q01"]
    q99 = norm_stats["action"]["q99"]
    
    # 方法 1: Z-score 归一化
    normalized = (action - mean) / (std + 1e-8)
    
    # 方法 2: 分位数归一化 (更鲁棒)
    # normalized = (action - q01) / (q99 - q01 + 1e-8) * 2 - 1
    
    return normalized

def denormalize_action(normalized, norm_stats):
    """反归一化动作"""
    mean = norm_stats["action"]["mean"]
    std = norm_stats["action"]["std"]
    
    action = normalized * (std + 1e-8) + mean
    return action
```

## 5. 数学原理

### 5.1 流匹配理论

**最优传输问题**:
$$\min_{v} \mathbb{E}\left[\int_0^1 \|v_t(X_t)\|^2 dt\right]$$
$$\text{s.t. } dX_t = v_t(X_t)dt, \quad X_0 \sim p_0, \quad X_1 \sim p_1$$

**条件流匹配**:
定义条件概率路径：
$$p_t(x|x_0, x_1) = \mathcal{N}(x; (1-t)x_0 + tx_1, \sigma^2 I)$$

速度场：
$$u_t(x|x_0, x_1) = x_1 - x_0$$

**训练目标**:
$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, p_1(x_1), p_0(x_0)}\left[\|v_\theta(X_t, t) - (x_1 - x_0)\|^2\right]$$

其中：
- $t \sim \mathcal{U}(0, 1)$
- $x_0 \sim \mathcal{N}(0, I)$ (噪声)
- $x_1 \sim p_{\text{data}}$ (数据)
- $X_t = (1-t)x_0 + tx_1$

**实现**:
```python
def flow_matching_loss(
    model: FlowMatchingHead,
    action_0: torch.Tensor,  # 噪声
    action_1: torch.Tensor,  # 数据
    condition: torch.Tensor
) -> torch.Tensor:
    B = action_0.shape[0]
    device = action_0.device
    
    # 采样时间
    t = torch.rand(B, device=device)
    
    # 插值
    action_t = t.view(-1, 1, 1) * action_1 + (1 - t.view(-1, 1, 1)) * action_0
    
    # 目标速度
    target_velocity = action_1 - action_0
    
    # 预测速度
    predicted_velocity = model(action_t, t, condition)
    
    # 损失
    loss = F.mse_loss(predicted_velocity, target_velocity)
    
    return loss
```

### 5.2 ODE 积分 (推理)

**生成 ODE**:
$$\frac{dx}{dt} = v_\theta(x, t), \quad x(0) \sim p_0$$

**Euler 方法**:
$$x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t$$

**实现**:
```python
@torch.no_grad()
def sample_euler(
    model: FlowMatchingHead,
    condition: torch.Tensor,
    num_steps: int = 10
) -> torch.Tensor:
    B = condition.shape[0]
    device = condition.device
    
    # 从噪声开始 (t=1)
    x = torch.randn(
        B, model.action_horizon, model.action_dim,
        device=device
    )
    
    # Euler 积分
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = torch.ones(B, device=device) * (1 - i * dt)
        
        # 预测速度
        v = model(x, t, condition)
        
        # Euler 步 (反向：从 t=1 到 t=0)
        x = x - dt * v
    
    return x
```

### 5.3 FAST Tokenizer 的 DCT

**DCT-II**:
$$X_k = \sum_{n=0}^{N-1} x_n \cos\left[\frac{\pi}{N}\left(n + \frac{1}{2}\right)k\right]$$

**矩阵形式**:
$$\mathbf{X} = \mathbf{D} \mathbf{x}$$

其中 $\mathbf{D}$ 是 DCT 矩阵：
$$D_{kn} = \cos\left[\frac{\pi}{N}\left(n + \frac{1}{2}\right)k\right]$$

**逆 DCT**:
$$\mathbf{x} = \mathbf{D}^T \mathbf{X}$$

### 5.4 向量量化

**量化器**:
$$\mathbf{z}_q = \mathbf{e}_k, \quad k = \arg\min_j \|\mathbf{z} - \mathbf{e}_j\|_2$$

**直通估计器** (Straight-Through Estimator):
$$\nabla_{\mathbf{z}} \mathcal{L} \approx \nabla_{\mathbf{z}_q} \mathcal{L}$$

**代码**:
```python
class VectorQuantizer(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim)
    
    def forward(self, z: torch.Tensor):
        # 计算距离
        distances = torch.cdist(z.unsqueeze(1), self.embedding.weight.unsqueeze(0))
        
        # 找到最近的嵌入
        indices = torch.argmin(distances, dim=-1)
        
        # 量化
        z_q = self.embedding(indices)
        
        # 直通估计器
        z_q = z + (z_q - z).detach()
        
        return z_q, indices
```

### 5.5 知识绝缘的损失函数

**总损失**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{flow}} + \lambda_{\text{insul}} \mathcal{L}_{\text{insul}}$$

**绝缘损失**:
$$\mathcal{L}_{\text{insul}} = \|\text{VLM}_{\text{frozen}}(x) - \text{VLM}_{\text{finetuned}}(x)\|^2$$

## 6. 性能评估

### 6.1 推理性能

| 模型 | 步数 | 延迟 (A100) | 成功率 |
|------|------|------------|--------|
| π₀ | 10 | 50ms | 92% |
| π₀-FAST | N/A | 30ms | 88% |
| π₀.₅ | 10 | 50ms | 95% |

### 6.2 基准测试结果

**LIBERO 基准**:
| 模型 | LIBERO-Spatial | LIBERO-Object | LIBERO-Long | 平均 |
|------|---------------|---------------|-------------|------|
| π₀ | 88.5% | 85.2% | 72.3% | 82.0% |
| π₀.₅ | 92.3% | 89.7% | 78.5% | 86.8% |
| OpenVLA | 85.0% | 82.0% | 68.0% | 78.3% |
| Octo | 78.0% | 75.0% | 60.0% | 71.0% |

## 7. 总结与启示

### 7.1 openpi 的核心创新

1. **流匹配动作生成**:
   - 比扩散模型更快的推理 (10 步 vs 100 步)
   - 更稳定的训练
   - 理论优雅

2. **知识绝缘** (π₀.₅):
   - 保护预训练 VLM 知识
   - 提升开放世界泛化能力
   - 减少灾难性遗忘

3. **FAST Tokenizer** (π₀-FAST):
   - 将连续动作离散化
   - 与 LLM 无缝集成
   - 支持自回归生成

### 7.2 对 vla-training 的启示

1. **采用流匹配**:
   - 替换扩散头为流匹配头
   - 减少推理步数到 10 步
   - 提升训练稳定性

2. **知识绝缘策略**:
   - 微调时冻结大部分 VLM 参数
   - 只解冻顶层进行适配
   - 保护预训练知识

3. **归一化统计**:
   - 使用分位数归一化 (q01, q99)
   - 预计算并保存统计量
   - 支持从预训练加载

---

*参考文献*:
- π₀ Blog: https://www.physicalintelligence.company/blog/pi0
- π₀.₅ Blog: https://www.physicalintelligence.company/blog/pi05
- FAST Research: https://www.physicalintelligence.company/research/fast
- Knowledge Insulation: https://www.physicalintelligence.company/research/knowledge_insulation
- Flow Matching Paper: https://arxiv.org/abs/2210.02747
- openpi GitHub: https://github.com/Physical-Intelligence/openpi
