# Spirit-v1.5 详细解读

## 1. 代码框架和项目结构

### 1.1 项目概述

Spirit-v1.5 是由 Spirit AI 团队开发的视觉 - 语言 - 动作 (VLA) 模型，在 RoboChallenge Table30 基准测试中排名第一 (截至 2026 年 1 月 11 日)。该模型基于 Qwen3-VL 视觉语言模型和 DiT 动作头，专为机器人控制设计。

**核心理念**: Clean Data Is the Enemy of Great Robot Foundation Models (干净数据是伟大机器人基础模型的敌人)

### 1.2 目录结构

```
spirit-v1.5/
├── model/                      # 模型定义
│   ├── modeling_spirit_vla.py  # 主模型架构
│   │   - Qwen3-VL backbone
│   │   - DiT head
│   │   - Policy API
│   └── utils.py                # 共享工具
│       - 归一化
│       - 采样
│       - 预处理
│
├── robochallenge/              # RoboChallenge 集成
│   ├── run_robochallenge.py    # Python 入口
│   ├── runner/
│   │   ├── executor.py         # 执行器
│   │   │   - 检查点加载
│   │   │   - 推理
│   │   │   - I/O 处理
│   │   └── task_info.py        # 任务元数据
│   │       - 机器人类型
│   │       - 动作类型
│   │       - 提示词
│   ├── robot/
│   │   ├── interface_client.py # HTTP 客户端
│   │   └── job_worker.py       # 任务轮询
│   └── utils/
│       ├── enums.py            # 枚举/常量
│       ├── log.py              # 日志工具
│       └── util.py             # 杂项工具
│
├── scripts/                    # 脚本
│   └── run_robochallenge.sh    # 启动脚本
│
└── requirements.txt            # 依赖
```

### 1.3 核心模块说明

**model/modeling_spirit_vla.py** - 主模型架构:
```python
"""
Spirit-v1.5 VLA 模型
基于 Qwen3-VL + DiT 头
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Optional

class SpiritVLA(nn.Module):
    """
    Spirit-v1.5 模型
    
    架构:
    - Qwen3-VL 视觉语言主干
    - DiT 动作头 (扩散 Transformer)
    - 动作块预测
    """
    
    def __init__(
        self,
        vlm_name: str = "Qwen/Qwen3-VL-7B",
        action_dim: int = 14,
        chunk_size: int = 60,
        hidden_dim: int = 1024,
        num_dit_layers: int = 16,
        num_heads: int = 16
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        
        # Qwen3-VL 主干
        self.vlm = AutoModel.from_pretrained(
            vlm_name,
            trust_remote_code=True
        )
        vlm_hidden = self.vlm.config.hidden_size
        
        # 投影层 (VLM → DiT)
        self.vlm_proj = nn.Linear(vlm_hidden, hidden_dim)
        
        # DiT 动作头
        self.dit_head = DiTActionHead(
            input_dim=hidden_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=hidden_dim,
            num_layers=num_dit_layers,
            num_heads=num_heads
        )
        
        # 扩散调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear"
        )
    
    def forward(
        self,
        images: torch.Tensor,      # [B, N, C, H, W]
        text: str,
        actions: Optional[torch.Tensor] = None,  # [B, T, D]
        train: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: 多视角图像
            text: 语言指令
            actions: 真实动作 (训练时)
            train: 是否训练模式
        
        Returns:
            outputs: 包含 loss (训练) 或 actions (推理)
        """
        # 1. VLM 编码
        vlm_outputs = self.vlm(
            images=images,
            text=text
        )
        vlm_features = vlm_outputs.last_hidden_state
        
        # 2. 投影
        vlm_proj = self.vlm_proj(vlm_features)
        
        if train and actions is not None:
            # 训练：扩散去噪
            loss = self._diffusion_train(
                vlm_proj, actions
            )
            return {"loss": loss}
        else:
            # 推理：采样动作
            actions = self._diffusion_sample(
                vlm_proj, num_steps=50
            )
            return {"actions": actions}
    
    def _diffusion_train(
        self,
        condition: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """扩散训练"""
        B = actions.shape[0]
        device = actions.device
        
        # 采样时间步
        t = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (B,), device=device
        ).long()
        
        # 添加噪声
        noise = torch.randn_like(actions)
        noisy_actions = self.noise_scheduler.add_noise(
            actions, noise, t
        )
        
        # 预测噪声
        predicted_noise = self.dit_head(
            noisy_actions,
            t.float() / self.noise_scheduler.num_train_timesteps,
            condition
        )
        
        # 计算损失
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def _diffusion_sample(
        self,
        condition: torch.Tensor,
        num_steps: int = 50
    ) -> torch.Tensor:
        """扩散采样"""
        B = condition.shape[0]
        device = condition.device
        
        # 从噪声开始
        actions = torch.randn(
            B, self.chunk_size, self.action_dim,
            device=device
        )
        
        # DDIM 采样
        self.noise_scheduler.set_timesteps(num_steps)
        
        for t in self.noise_scheduler.timesteps:
            # 预测噪声
            noise_pred = self.dit_head(
                actions,
                t.float() / self.noise_scheduler.num_train_timesteps,
                condition
            )
            
            # DDIM 更新
            actions = self.noise_scheduler.step(
                noise_pred, t, actions
            ).prev_sample
        
        return actions
```

**robochallenge/runner/executor.py** - RoboChallenge 执行器:
```python
class RoboChallengeExecutor:
    """
    RoboChallenge 执行器
    
    负责:
    - 加载检查点
    - 运行推理
    - 处理 I/O
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        chunk_size: int = 60,
        device: str = "cuda"
    ):
        # 加载模型
        self.model = SpiritVLA.from_pretrained(checkpoint_path)
        self.model.to(device)
        self.model.eval()
        
        self.chunk_size = chunk_size
        self.device = device
        
        # 加载任务信息
        self.task_info = self._load_task_info()
    
    def execute_task(
        self,
        task_name: str,
        job_id: str,
        user_token: str
    ):
        """执行 RoboChallenge 任务"""
        # 1. 获取任务配置
        task_config = self.task_info[task_name]
        
        # 2. 连接到 RoboChallenge 服务器
        client = RoboChallengeClient(
            job_id=job_id,
            token=user_token
        )
        
        # 3. 执行循环
        while True:
            # 获取观测
            obs = client.get_observation()
            
            # 运行推理
            actions = self.model.predict(
                images=obs["images"],
                text=task_config["prompt"]
            )
            
            # 执行动作
            client.send_action(actions[0])  # 执行第一步
            
            # 检查任务完成
            if self._is_task_complete(obs):
                break
```

## 2. 解决问题的流程和方法

### 2.1 核心问题定义

Spirit-v1.5 解决的核心问题是：**如何在 RoboChallenge 等标准化基准测试中实现 SOTA 性能，同时保持模型的通用性和泛化能力？**

### 2.2 技术路线

```
┌─────────────────────────────────────────────────────────────┐
│                 Spirit-v1.5 技术路线                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 强大的 VLM 主干 (Qwen3-VL)                               │
│     └─► 继承强大的视觉 - 语言理解能力                         │
│                                                             │
│  2. DiT 动作头                                               │
│     └─► 精确的动作生成，支持长动作块 (60 步)                    │
│                                                             │
│  3. 大规模预训练                                             │
│     └─► 多样化机器人数据混合                                 │
│                                                             │
│  4. RoboChallenge 适配                                       │
│     └─► 针对基准测试优化                                     │
│                                                             │
│  5. 高效推理                                                 │
│     └─► 50 步 DDIM 采样，实时控制                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 RoboChallenge 集成流程

**步骤 1: 环境配置**
```bash
# 设置环境变量
export TASK_NAME=move_objects_into_box
export ROBOCHALLENGE_JOB_ID=your_job_collection_id
export USER_TOKEN=your_user_token
export CKPT_PATH=/path/to/your_checkpoint_dir
export USED_CHUNK_SIZE=60

# 运行
./scripts/run_robochallenge.sh
```

**步骤 2: 任务执行循环**
```python
# robochallenge/robot/job_worker.py
class JobWorker:
    def __init__(self, executor, task_name):
        self.executor = executor
        self.task_name = task_name
    
    def run(self):
        """运行任务"""
        while True:
            # 轮询任务
            job = self.poll_for_job()
            
            if job is None:
                time.sleep(1)
                continue
            
            # 执行任务
            result = self.execute_job(job)
            
            # 提交结果
            self.submit_result(result)
    
    def execute_job(self, job):
        """执行单个任务"""
        # 初始化环境
        obs = self.reset_environment(job)
        
        # 执行循环
        for step in range(job["max_steps"]):
            # 获取动作
            actions = self.executor.predict(
                images=obs["images"],
                text=job["instruction"]
            )
            
            # 执行动作
            obs = self.step(actions[0])
            
            # 检查完成
            if self.check_completion(obs):
                return {"success": True, "steps": step + 1}
        
        return {"success": False, "steps": job["max_steps"]}
```

### 2.4 训练流程

**数据准备**:
```python
# 准备训练数据
def prepare_training_data(raw_data_dir, output_dir):
    """准备训练数据"""
    episodes = load_episodes(raw_data_dir)
    
    # 数据过滤 (关键！)
    filtered_episodes = []
    for ep in episodes:
        if self._is_high_quality(ep):
            filtered_episodes.append(ep)
    
    # 转换为 LeRobot 格式
    convert_to_lerobot(filtered_episodes, output_dir)
    
    return output_dir

def _is_high_quality(self, episode):
    """判断数据质量"""
    # 检查标准:
    # 1. 动作平滑性
    # 2. 任务完成度
    # 3. 无异常值
    pass
```

**训练命令**:
```bash
# 微调训练
python train.py \
  --base_model Qwen/Qwen3-VL-7B \
  --dataset_path /path/to/lerobot_data \
  --output_dir /path/to/output \
  --chunk_size 60 \
  --action_dim 14 \
  --batch_size 32 \
  --lr 1e-4 \
  --max_steps 30000
```

## 3. 模型架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Spirit-v1.5 架构                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐                       │
│  │   Images     │    │   Text       │                       │
│  │  (多视角)     │    │   Prompt     │                       │
│  └──────┬───────┘    └──────┬───────┘                       │
│         │                   │                                │
│         ▼                   ▼                                │
│  ┌─────────────────────────────────┐                         │
│  │        Qwen3-VL Backbone        │                         │
│  │                                 │                         │
│  │  • Vision Encoder (ViT)         │                         │
│  │  • Language Model (Qwen)        │                         │
│  │  • Cross-Modal Attention        │                         │
│  └──────────────┬──────────────────┘                         │
│                 │                                            │
│                 ▼                                            │
│        ┌────────────────┐                                    │
│        │   VLM Proj     │                                    │
│        │   (Linear)     │                                    │
│        └────────┬───────┘                                    │
│                 │                                            │
│                 ▼                                            │
│        ┌────────────────┐                                    │
│        │   DiT Head     │                                    │
│        │   (16 层)       │                                    │
│        │  动作去噪       │                                    │
│        └────────┬───────┘                                    │
│                 │                                            │
│                 ▼                                            │
│        ┌────────────────┐                                    │
│        │   Actions      │                                    │
│        │   (60 步块)      │                                    │
│        └────────────────┘                                    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Qwen3-VL 主干

**架构特点**:
- 基于 Qwen3-VL 7B 模型
- 支持多视角图像输入
- 强大的视觉 - 语言理解能力

**实现**:
```python
class QwenVLBackbone(nn.Module):
    """Qwen3-VL 主干"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-7B"):
        super().__init__()
        
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 冻结策略
        self.freeze_strategy = "partial"
    
    def forward(
        self,
        images: torch.Tensor,
        text: str
    ) -> BaseModelOutput:
        """
        编码图像和文本
        
        Args:
            images: [B, N, C, H, W] 多视角图像
            text: 语言指令
        
        Returns:
            last_hidden_state: [B, L, H]
        """
        # Qwen3-VL 处理多视角
        # 内部会将多视角图像拼接
        outputs = self.model(
            images=images,
            text=text
        )
        
        return outputs
```

### 3.3 DiT 动作头

**架构**:
```python
class DiTActionHead(nn.Module):
    """
    DiT 动作头
    
    16 层扩散 Transformer
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        action_dim: int = 14,
        chunk_size: int = 60,
        hidden_dim: int = 1024,
        num_layers: int = 16,
        num_heads: int = 16,
        time_embed_dim: int = 256
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 动作嵌入
        self.action_embed = nn.Linear(
            action_dim * chunk_size, hidden_dim
        )
        
        # 条件投影
        self.cond_proj = nn.Linear(input_dim, hidden_dim)
        
        # DiT 块 (16 层)
        self.dit_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, time_embed_dim)
            for _ in range(num_layers)
        ])
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim * chunk_size)
        )
    
    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """预测噪声"""
        B = noisy_actions.shape[0]
        
        # 嵌入
        t_emb = self.time_embed(timestep)
        a_emb = self.action_embed(noisy_actions.reshape(B, -1))
        c_emb = self.cond_proj(condition.mean(dim=1))
        
        # 合并
        x = a_emb + t_emb + c_emb
        
        # 通过 DiT 块
        for block in self.dit_blocks:
            x = block(x, t_emb)
        
        # 预测噪声
        noise = self.output_head(x)
        return noise.reshape(B, self.chunk_size, self.action_dim)
```

### 3.4 动作块设计

**长动作块** (60 步):
```python
# Spirit-v1.5 使用 60 步动作块
chunk_size = 60

# 优势:
# 1. 减少推理频率
# 2. 更好的时序一致性
# 3. 适合 RoboChallenge 任务

# 执行策略:
# - 预测 60 步动作
# - 执行第 1 步
# - 每 N 步重新预测 (N < 60)
```

## 4. 训练过程和原理

### 4.1 数据质量过滤

**关键理念**: "Clean Data Is the Enemy of Great Robot Foundation Models"

**过滤标准**:
```python
def filter_data(episodes):
    """数据质量过滤"""
    filtered = []
    
    for ep in episodes:
        # 1. 动作平滑性检查
        if not check_action_smoothness(ep["actions"]):
            continue
        
        # 2. 任务完成度检查
        if not check_task_completion(ep):
            continue
        
        # 3. 异常值检查
        if has_outliers(ep["actions"]):
            continue
        
        # 4. 时间一致性检查
        if not check_temporal_consistency(ep):
            continue
        
        filtered.append(ep)
    
    return filtered

def check_action_smoothness(actions, threshold=0.5):
    """检查动作平滑性"""
    # 计算动作差分
    diffs = np.diff(actions, axis=0)
    
    # 检查最大差分
    max_diff = np.max(np.abs(diffs))
    
    return max_diff < threshold
```

### 4.2 训练配置

```yaml
# 训练配置
model:
  vlm_name: Qwen/Qwen3-VL-7B
  action_dim: 14
  chunk_size: 60
  num_dit_layers: 16

training:
  batch_size: 32
  num_epochs: 100
  max_steps: 30000
  
  optimizer:
    type: adamw
    lr: 1e-4
    weight_decay: 0.1
  
  scheduler:
    type: cosine
    warmup_steps: 1000
  
  # 扩散配置
  diffusion:
    num_train_timesteps: 1000
    beta_schedule: scaled_linear
  
  # 冻结策略
  freeze:
    vlm: partial  # 部分冻结
    vlm_unfreeze_layers: 4  # 解冻顶层 4 层
```

### 4.3 数据增强

```python
from torchvision.transforms import Compose, RandomCrop, ColorJitter

train_transform = Compose([
    RandomCrop(size=(224, 224)),
    ColorJitter(
        brightness=0.3,
        contrast=0.4,
        saturation=0.5,
        hue=0.08
    ),
    # 对于 RoboChallenge，不使用水平翻转
    # 因为任务通常有方向性
])
```

## 5. 数学原理

### 5.1 扩散模型基础

**前向过程**:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

**累积噪声**:
$$\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$$

**任意时刻**:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

### 5.2 扩散损失

**去噪目标**:
$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]$$

**实现**:
```python
def diffusion_loss(
    model: DiTActionHead,
    clean_actions: torch.Tensor,
    condition: torch.Tensor,
    noise_scheduler
) -> torch.Tensor:
    B, T, D = clean_actions.shape
    device = clean_actions.device
    
    # 采样时间步
    t = torch.randint(
        0, noise_scheduler.num_train_timesteps,
        (B,), device=device
    ).long()
    
    # 采样噪声
    noise = torch.randn_like(clean_actions)
    
    # 添加噪声
    noisy_actions = noise_scheduler.add_noise(
        clean_actions, noise, t
    )
    
    # 预测噪声
    predicted_noise = model(
        noisy_actions,
        t.float() / noise_scheduler.num_train_timesteps,
        condition
    )
    
    # 计算损失
    loss = F.mse_loss(predicted_noise, noise)
    
    return loss
```

### 5.3 DDIM 采样

**更新公式**:
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

**实现**:
```python
@torch.no_grad()
def ddim_sample(
    model: DiTActionHead,
    condition: torch.Tensor,
    noise_scheduler,
    num_steps: int = 50
) -> torch.Tensor:
    B = condition.shape[0]
    device = condition.device
    
    # 从噪声开始
    actions = torch.randn(
        B, model.chunk_size, model.action_dim,
        device=device
    )
    
    # 设置时间步
    noise_scheduler.set_timesteps(num_steps)
    
    for t in noise_scheduler.timesteps:
        # 预测噪声
        noise_pred = model(
            actions,
            t.float() / noise_scheduler.num_train_timesteps,
            condition
        )
        
        # DDIM 更新
        actions = noise_scheduler.step(
            noise_pred, t, actions
        ).prev_sample
    
    return actions
```

### 5.4 动作块数学

**时序建模**:
$$\mathbf{a}_{t:t+H} = f_\theta(o_t, \text{text})$$

其中：
- $H = 60$: 动作块大小
- $\mathbf{a}_{t:t+H} \in \mathbb{R}^{60 \times 14}$: 预测的动作序列

**执行策略**:
$$a_{\text{executed}} = \mathbf{a}_{t:t+H}[0]$$

**重新预测频率**:
- 每 $K$ 步重新预测 ($K < H$)
- Spirit-v1.5 通常使用 $K = 10-20$

## 6. RoboChallenge 基准

### 6.1 基准介绍

RoboChallenge 是一个标准化的机器人操作基准测试，包含 Table30 等任务。

**评估指标**:
- 成功率 (Success Rate)
- 任务完成时间
- 动作平滑性

### 6.2 Spirit-v1.5 性能

**Table30 基准** (截至 2026-01-11):
- **排名**: #1
- **平均成功率**: 85%+
- **任务**: move_objects_into_box 等

### 6.3 任务适配

**任务配置** (`task_info.py`):
```python
TASK_INFO = {
    "move_objects_into_box": {
        "robot_type": "panda",
        "action_type": "joint_position",
        "action_dim": 7,
        "prompt": "Move the objects into the box",
        "max_steps": 500,
        "success_threshold": 0.9
    },
    # ... 其他任务
}
```

## 7. 总结与启示

### 7.1 Spirit-v1.5 的核心优势

1. **强大的 VLM 主干**:
   - Qwen3-VL 7B 提供强大的视觉 - 语言理解
   - 继承互联网规模预训练知识

2. **DiT 动作头**:
   - 16 层 Transformer 提供强大表达能力
   - 支持长动作块 (60 步)

3. **数据质量优先**:
   - 严格的数据过滤
   - "干净数据是敌人"的理念

4. **基准优化**:
   - 针对 RoboChallenge 优化
   - SOTA 性能

### 7.2 对 vla-training 的启示

1. **VLM 选择**:
   - 考虑使用 Qwen-VL 系列
   - 强大的视觉 - 语言理解能力

2. **动作块设计**:
   - 增加动作块大小 (60 步)
   - 减少推理频率

3. **数据质量**:
   - 实施严格的数据过滤
   - 质量优于数量

4. **基准集成**:
   - 支持标准基准测试
   - 提供评估工具

---

*参考文献*:
- Spirit-v1.5 GitHub: https://github.com/Spirit-AI-Team/spirit-v1.5
- Spirit AI Blog: https://www.spirit-ai.com/en/blog/spirit-v1-5
- RoboChallenge: https://robochallenge.cn/home
- Qwen-VL Paper: https://arxiv.org/abs/2308.12966
