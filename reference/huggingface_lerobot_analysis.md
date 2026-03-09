# Hugging Face LeRobot 详细解读

## 1. 代码框架和项目结构

### 1.1 项目概述

LeRobot 是 Hugging Face 开发的开源机器人学习框架，旨在通过 PyTorch 实现最先进的机器学习和机器人技术的民主化。它提供了模型、数据集和工具的完整生态系统，支持从低成本机械臂到人形机器人的各种平台。

**核心理念**: 降低机器人学习的门槛，使每个人都能贡献和受益于共享数据集和预训练模型。

### 1.2 目录结构

```
lerobot/
├── lerobot/                    # 核心代码目录
│   ├── robots/                 # 机器人硬件接口
│   │   ├── robot.py            # 基础机器人接口
│   │   ├── so100/              # SO-100 机械臂
│   │   ├── koch/               # Koch 机械臂
│   │   ├── reachy2/            # Reachy2 机器人
│   │   ├── unitree/            # Unitree G1
│   │   └── robot_device.py     # 设备抽象层
│   │
│   ├── datasets/               # 数据集模块
│   │   ├── lerobot_dataset.py  # LeRobotDataset 核心类
│   │   ├── utils/              # 数据工具
│   │   │   ├── video_utils.py  # 视频处理
│   │   │   └── parquet_utils.py # Parquet 处理
│   │   └── transforms.py       # 数据变换
│   │
│   ├── policies/               # 策略实现
│   │   ├── policy_protocol.py  # 策略协议
│   │   ├── act/                # ACT 策略
│   │   │   ├── modeling_act.py # ACT 模型
│   │   │   └── policy_act.py   # ACT 策略接口
│   │   ├── diffusion/          # Diffusion 策略
│   │   │   ├── modeling_diffusion.py
│   │   │   └── policy_diffusion.py
│   │   ├── vqbet/              # VQ-BeT 策略
│   │   ├── tdmpc/              # TDMPC 策略
│   │   ├── groot/              # GR00T 策略
│   │   ├── smolvla/            # SmolVLA 策略
│   │   └── pi0fast/            # Pi0-FAST 策略
│   │
│   ├── scripts/                # 命令行工具
│   │   ├── train.py            # 训练脚本
│   │   ├── eval.py             # 评估脚本
│   │   └── record.py           # 数据记录
│   │
│   └── utils/                  # 工具函数
│       ├── import_utils.py
│       ├── logging_utils.py
│       └── io_utils.py
│
├── examples/                   # 示例代码
│   ├── hardware/               # 硬件示例
│   ├── datasets/               # 数据集示例
│   └── policies/               # 策略示例
│
├── tests/                      # 单元测试
└── docs/                       # 文档
```

### 1.3 核心模块说明

**lerobot/datasets/lerobot_dataset.py** - 数据集核心类:
```python
class LeRobotDataset(Dataset):
    """
    LeRobot 数据集类
    
    特点:
    - 标准化数据格式 (Parquet + MP4)
    - 支持 Hugging Face Hub 集成
    - 高效流式加载
    - 视频自动解码
    """
    
    def __init__(
        self,
        repo_id: str,
        version: Optional[str] = None,
        root: str = "./data",
        split: str = "train",
        transform: Optional[Callable] = None
    ):
        self.repo_id = repo_id
        self.root = Path(root)
        self.split = split
        
        # 下载或加载数据集
        self.repo_path = self._download_or_load(repo_id, version)
        
        # 加载元数据
        self.meta = self._load_meta()
        
        # 加载 Parquet 数据
        self.parquet_data = self._load_parquet()
        
        # 视频解码器
        self.video_decoder = VideoDecoder(self.repo_path)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        episode_index = self._get_episode_index(idx)
        
        # 加载图像 (从 MP4 视频)
        images = self.video_decoder.get_frame(
            episode_index,
            frame_index=self._get_frame_index(idx)
        )
        
        # 加载状态和动作 (从 Parquet)
        state = self.parquet_data["observation.state"][idx]
        action = self.parquet_data["action"][idx]
        
        # 加载语言指令
        language = self.parquet_data["language"][idx]
        
        sample = {
            "observation.images": images,
            "observation.state": torch.tensor(state),
            "action": torch.tensor(action),
            "language": language
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self) -> int:
        return len(self.parquet_data)
```

**lerobot/policies/policy_protocol.py** - 策略协议:
```python
from typing import Protocol, Dict, Any

class PolicyProtocol(Protocol):
    """策略接口协议"""
    
    def __call__(
        self, 
        observation: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        从观测预测动作
        
        Args:
            observation: 观测字典
                - "observation.images": 图像
                - "observation.state": 本体感知状态
                - "language": 语言指令 (可选)
        
        Returns:
            action_dict: 动作字典
                - "action": 预测的动作
        """
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """获取策略配置"""
        ...
    
    def save_pretrained(self, save_dir: str):
        """保存预训练模型"""
        ...
    
    @classmethod
    def from_pretrained(cls, model_id: str) -> "PolicyProtocol":
        """从预训练模型加载"""
        ...
```

## 2. 解决问题的流程和方法

### 2.1 核心问题定义

LeRobot 解决的核心问题是：**如何标准化机器人学习的数据格式和模型接口，使研究和部署更加高效和可复现？**

### 2.2 技术路线

```
┌─────────────────────────────────────────────────────────────┐
│                    LeRobot 生态系统                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   硬件层    │    │   数据层    │    │   模型层    │     │
│  │             │    │             │    │             │     │
│  │ • SO-100    │───►│ • 标准化    │───►│ • ACT       │     │
│  │ • Koch      │    │ • LeRobot   │    │ • Diffusion │     │
│  │ • Reachy2   │    │   Dataset   │    │ • VLA       │     │
│  │ • Unitree   │    │ • HF Hub    │    │ • TDMPC     │     │
│  │   G1        │    │             │    │             │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│                   ┌─────────────────┐                       │
│                   │   统一接口层    │                       │
│                   │                 │                       │
│                   │ • Robot 类      │                       │
│                   │ • Policy 协议   │                       │
│                   │ • 评估框架      │                       │
│                   └─────────────────┘                       │
│                            │                                │
│                            ▼                                │
│                   ┌─────────────────┐                       │
│                   │   部署应用      │                       │
│                   └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 数据标准化流程

**LeRobotDataset V3 格式**:

```python
# 数据集目录结构
dataset_name/
├── meta.json              # 元数据
├── train/                 # 训练 split
│   ├── episode_00000.parquet    # 状态/动作数据
│   ├── episode_00001.parquet
│   ├── ...
│   └── video.mp4               # 同步视频
├── test/                  # 测试 split
│   └── ...
└── info.json              # 数据集信息
```

**元数据格式** (`meta.json`):
```json
{
    "episode_keys": ["episode_index"],
    "video_keys": ["observation.images.top", "observation.images.wrist"],
    "state_keys": ["observation.state"],
    "action_keys": ["action"],
    "fps": 30,
    "image_shape": {"top": [480, 640, 3], "wrist": [480, 640, 3]},
    "state_dim": 14,
    "action_dim": 14,
    "total_episodes": 100,
    "total_frames": 30000
}
```

**数据转换工具**:
```python
from lerobot.datasets.utils import dataset_to_lerobot

def convert_raw_to_lerobot(
    raw_data_dir: str,
    output_dir: str,
    fps: int = 30
):
    """将原始数据转换为 LeRobot 格式"""
    
    # 1. 收集所有 episode
    episodes = collect_episodes(raw_data_dir)
    
    # 2. 创建视频文件
    video_path = output_dir / "video.mp4"
    encode_videos(episodes["images"], video_path, fps)
    
    # 3. 创建 Parquet 文件
    for i, episode in enumerate(episodes):
        parquet_data = {
            "episode_index": [i] * len(episode),
            "frame_index": list(range(len(episode))),
            "observation.state": episode["states"],
            "action": episode["actions"],
            "language": [episode["instruction"]] * len(episode)
        }
        
        # 保存为 Parquet
        df = pd.DataFrame(parquet_data)
        df.to_parquet(output_dir / f"episode_{i:05d}.parquet")
    
    # 4. 保存元数据
    save_meta(output_dir, episodes)
```

### 2.4 训练流程

**使用命令行训练**:
```bash
# 训练 ACT 策略
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --output_dir=output/act_aloha \
  --batch_size=32 \
  --lr=1e-4 \
  --num_epochs=100

# 训练 Diffusion 策略
lerobot-train \
  --policy=diffusion \
  --dataset.repo_id=lerobot/pusht \
  --output_dir=output/diffusion_pusht

# 训练 VLA 策略 (Pi0-FAST)
lerobot-train \
  --policy=pi0fast \
  --dataset.repo_id=lerobot/droid \
  --output_dir=output/pi0fast_droid \
  --pretrained_model_name_or_path=physical-intelligence/pi0-fast-base
```

**Python API 训练**:
```python
from lerobot import train

config = {
    "policy": "act",
    "dataset": {
        "repo_id": "lerobot/aloha_mobile_cabinet"
    },
    "training": {
        "batch_size": 32,
        "num_epochs": 100,
        "lr": 1e-4,
        "optimizer": "adamw",
        "scheduler": "cosine"
    },
    "output_dir": "output/act_aloha"
}

train(config)
```

### 2.5 评估流程

```python
from lerobot import evaluate

# 在真实硬件上评估
evaluate(
    policy_path="output/act_aloha/checkpoint_best",
    env_type="real",
    robot="so100",
    num_episodes=10,
    max_episode_steps=500
)

# 在仿真环境中评估
evaluate(
    policy_path="output/act_aloha/checkpoint_best",
    env_type="sim",
    env_name="aloha_sim",
    task="aloha_sim_insertion",
    num_episodes=50
)

# 在 LIBERO 基准上评估
evaluate(
    policy_path="output/pi0fast_liberto/checkpoint_best",
    env_type="sim",
    env_name="libero",
    task="libero_object",
    num_episodes=10
)
```

## 3. 模型架构设计

### 3.1 支持的策略类型

LeRobot 实现了多种最先进的策略：

| 策略 | 类型 | 架构 | 适用场景 |
|------|------|------|---------|
| **ACT** | 模仿学习 | Transformer + VAE | 高精度操作 |
| **Diffusion** | 模仿学习 | U-Net + Transformer | 多模态动作 |
| **VQ-BeT** | 模仿学习 | VQ-VAE + Transformer | 离散动作 |
| **TDMPC** | 强化学习 | Model-based RL | 复杂决策 |
| **Pi0-FAST** | VLA | VLM + FAST Tokenizer | 通用任务 |
| **GR00T** | VLA | VLM + DiT | 人形机器人 |
| **SmolVLA** | VLA | 小型 VLM | 边缘部署 |

### 3.2 ACT (Action Chunking with Temporal Transforms)

**架构概述**:
```
┌─────────────────────────────────────────────────────────────┐
│                      ACT 架构                                │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Images     │    │   State      │    │   Language   │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │          │
│         ▼                   ▼                   │          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────┴───────┐  │
│  │   ResNet-18  │    │   State MLP  │    │   Text       │  │
│  │   (编码器)    │    │   (编码器)    │    │   Embedding  │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │          │
│         └────────┬──────────┘                   │          │
│                  ▼                              │          │
│         ┌────────────────┐                      │          │
│         │   Transformer  │◄─────────────────────┘          │
│         │   Encoder      │                                 │
│         │   (4 层)        │                                 │
│         └────────┬───────┘                                 │
│                  │                                         │
│         ┌────────▼────────┐                                │
│         │   VAE Latent    │                                │
│         │   (采样)         │                                │
│         └────────┬────────┘                                │
│                  │                                         │
│         ┌────────▼────────┐                                │
│         │   Transformer   │                                │
│         │   Decoder       │                                │
│         │   (4 层)         │                                │
│         └────────┬────────┘                                │
│                  │                                         │
│                  ▼                                         │
│         ┌────────────────┐                                 │
│         │   Actions      │                                 │
│         │   (chunk)      │                                 │
│         └────────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

**ACT 模型实现**:
```python
class ACTPolicy(nn.Module):
    """ACT 策略实现"""
    
    def __init__(
        self,
        input_shape: Dict[str, Tuple],
        action_dim: int,
        chunk_size: int = 100,
        hidden_dim: int = 512,
        num_queries: int = 100,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        use_vae: bool = True,
        kl_weight: float = 10.0
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_queries = num_queries
        self.use_vae = use_vae
        self.kl_weight = kl_weight
        
        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 60 * 80, hidden_dim)
        )
        
        # 状态编码器
        self.state_encoder = nn.Linear(input_shape["state"][0], hidden_dim)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # VAE 潜在变量
        if use_vae:
            self.latent_dim = 32
            self.kl_proj = nn.Linear(hidden_dim, 2 * self.latent_dim)
            self.latent_proj = nn.Linear(self.latent_dim, hidden_dim)
        
        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # 动作查询嵌入
        self.action_queries = nn.Embedding(num_queries, hidden_dim)
        
        # 动作头
        self.action_head = nn.Linear(hidden_dim, action_dim)
    
    def encode(self, images, state):
        """编码观测"""
        # 编码图像
        img_feat = self.image_encoder(images)
        
        # 编码状态
        state_feat = self.state_encoder(state)
        
        # 拼接
        feat = torch.cat([img_feat, state_feat], dim=1).unsqueeze(1)
        
        # Transformer 编码
        encoded = self.transformer_encoder(feat)
        
        return encoded
    
    def sample_latent(self, encoded):
        """从 VAE 采样潜在变量"""
        if not self.use_vae:
            return encoded
        
        # 预测均值和方差
        stats = self.kl_proj(encoded.mean(dim=1))
        mu, logvar = stats.chunk(2, dim=-1)
        
        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        
        # 投影回隐藏维度
        z_feat = self.latent_proj(z)
        
        return z_feat, mu, logvar
    
    def decode(self, latent, num_actions=None):
        """解码动作"""
        if num_actions is None:
            num_actions = self.num_queries
        
        # 获取动作查询
        queries = self.action_queries.weight[:num_actions].unsqueeze(0).expand(
            latent.shape[0], -1, -1
        )
        
        # Transformer 解码
        decoded = self.transformer_decoder(
            queries,
            latent.expand(-1, num_actions, -1)
        )
        
        # 预测动作
        actions = self.action_head(decoded)
        
        return actions
    
    def forward(self, images, state, actions=None):
        """
        前向传播
        
        Args:
            images: [B, C, H, W]
            state: [B, state_dim]
            actions: [B, chunk_size, action_dim] (训练时)
        """
        # 编码
        encoded = self.encode(images, state)
        
        # 采样潜在变量
        if self.use_vae:
            latent, mu, logvar = self.sample_latent(encoded)
        else:
            latent = encoded
            mu, logvar = None, None
        
        # 解码动作
        if actions is not None:
            # 训练模式：使用真实动作
            num_actions = actions.shape[1]
        else:
            # 推理模式：使用完整 chunk
            num_actions = self.num_queries
        
        predicted_actions = self.decode(latent, num_actions)
        
        # 计算 KL 散度
        if self.use_vae and mu is not None:
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            kl_loss = 0
        
        return predicted_actions, kl_loss
```

### 3.3 Diffusion Policy

**架构概述**:
```python
class DiffusionPolicy(nn.Module):
    """Diffusion 策略实现"""
    
    def __init__(
        self,
        input_shape: Dict[str, Tuple],
        action_dim: int,
        chunk_size: int = 8,
        horizon: int = 10,
        diff_step: int = 100,
        hidden_dim: int = 512,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.horizon = horizon
        self.diff_step = diff_step
        
        # 噪声调度
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=diff_step,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        
        # 观测编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_shape["state"][0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # U-Net 风格去噪网络
        self.denoise_net = nn.ModuleList([
            DiffusionBlock(hidden_dim, num_heads)
            for _ in range(8)
        ])
        
        # 动作预测头
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim * chunk_size)
        )
    
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """预测噪声"""
        # 编码观测
        obs_feat = self.obs_encoder(obs["state"])
        
        # 时间嵌入
        t_emb = self.time_embed(timestep)
        
        # 合并特征
        x = torch.cat([
            noisy_actions.reshape(noisy_actions.shape[0], -1),
            obs_feat,
            t_emb
        ], dim=-1)
        
        # 去噪网络
        for block in self.denoise_net:
            x = block(x, t_emb)
        
        # 预测噪声
        noise = self.action_head(x)
        return noise.reshape(noisy_actions.shape)
    
    @torch.no_grad()
    def sample(
        self,
        obs: Dict[str, torch.Tensor],
        num_steps: int = 10
    ) -> torch.Tensor:
        """采样动作"""
        B = obs["state"].shape[0]
        device = obs["state"].device
        
        # 从噪声开始
        actions = torch.randn(
            B, self.chunk_size, self.action_dim,
            device=device
        )
        
        # DDIM 采样
        self.noise_scheduler.set_timesteps(num_steps)
        
        for t in self.noise_scheduler.timesteps:
            # 预测噪声
            noise_pred = self.forward(obs, actions, t)
            
            # DDIM 更新
            actions = self.noise_scheduler.step(
                noise_pred, t, actions
            ).prev_sample
        
        return actions
```

### 3.4 VLA 策略 (Pi0-FAST)

**FAST Tokenizer**:
```python
class FASTTokenizer:
    """
    Frequency-Action Space Tokenizer
    
    将连续动作转换为离散 token:
    1. 动作分块
    2. DCT 变换到频率域
    3. 量化
    4. BPE 编码
    """
    
    def __init__(
        self,
        action_dim: int,
        chunk_size: int = 32,
        num_bins: int = 256
    ):
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_bins = num_bins
        
        # DCT 基矩阵
        self.dct_matrix = self._create_dct_matrix(chunk_size)
    
    def _create_dct_matrix(self, N: int) -> np.ndarray:
        """创建 DCT 变换矩阵"""
        dct = np.zeros((N, N))
        for k in range(N):
            for n in range(N):
                if k == 0:
                    dct[k, n] = 1 / np.sqrt(N)
                else:
                    dct[k, n] = np.sqrt(2/N) * np.cos(
                        np.pi * k * (2*n + 1) / (2*N)
                    )
        return dct
    
    def encode(self, actions: np.ndarray) -> List[int]:
        """将动作编码为 token"""
        # 1. DCT 变换
        freq_coeffs = self.dct_matrix @ actions
        
        # 2. 量化
        tokens = []
        for coeff in freq_coeffs.flatten():
            bin_id = int((coeff + 1) / 2 * self.num_bins)
            bin_id = np.clip(bin_id, 0, self.num_bins - 1)
            tokens.append(bin_id)
        
        return tokens
    
    def decode(self, tokens: List[int]) -> np.ndarray:
        """将 token 解码为动作"""
        # 1. 反量化
        coeffs = []
        for token in tokens:
            coeff = (token / self.num_bins) * 2 - 1
            coeffs.append(coeff)
        
        # 2. 逆 DCT
        actions = self.dct_matrix.T @ np.array(coeffs).reshape(
            self.chunk_size, self.action_dim
        )
        
        return actions
```

## 4. 训练过程和原理

### 4.1 数据增强

LeRobot 提供丰富的数据增强策略：

```python
from lerobot.datasets.transforms import Compose, RandomCrop, ColorJitter

train_transform = Compose([
    RandomCrop(size=(224, 224)),
    ColorJitter(
        brightness=0.3,
        contrast=0.4,
        saturation=0.5,
        hue=0.08
    ),
    # 注意：对于非对称机器人，不使用水平翻转
])
```

### 4.2 训练配置

**ACT 训练配置**:
```yaml
# configs/policy/act.yaml
policy:
  name: act
  chunk_size: 100
  hidden_dim: 512
  num_encoder_layers: 4
  num_decoder_layers: 4
  num_heads: 8
  use_vae: true
  kl_weight: 10.0

training:
  batch_size: 32
  num_epochs: 100
  lr: 1e-4
  weight_decay: 0.0
  grad_clip: 10.0
  
  optimizer:
    type: adamw
    betas: [0.9, 0.999]
  
  scheduler:
    type: cosine
    warmup_epochs: 10
  
  checkpoint:
    save_every: 10
    keep_last: 5
```

**Diffusion 训练配置**:
```yaml
# configs/policy/diffusion.yaml
policy:
  name: diffusion
  chunk_size: 8
  horizon: 10
  diff_step: 100
  hidden_dim: 512
  
  noise_scheduler:
    type: ddim
    beta_schedule: scaled_linear
    prediction_type: epsilon

training:
  batch_size: 64
  num_epochs: 200
  lr: 1e-4
  
  # 扩散特定配置
  ema:
    enabled: true
    decay: 0.999
```

### 4.3 损失函数

**ACT 损失**:
```python
def act_loss(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    kl_loss: torch.Tensor,
    kl_weight: float = 10.0
) -> torch.Tensor:
    """
    ACT 损失函数
    
    L = L_action + kl_weight * L_KL
    """
    # 动作损失 (L1)
    action_loss = F.l1_loss(predicted_actions, target_actions)
    
    # 总损失
    total_loss = action_loss + kl_weight * kl_loss
    
    return total_loss
```

**Diffusion 损失**:
```python
def diffusion_loss(
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor
) -> torch.Tensor:
    """
    扩散损失函数
    
    L = ||ε - ε_θ(x_t, t)||²
    """
    return F.mse_loss(predicted_noise, target_noise)
```

## 5. 数学原理

### 5.1 ACT 的 VAE 公式

**变分下界 (ELBO)**:
$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) || p(z))$$

**重参数化技巧**:
$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**KL 散度**:
$$D_{KL}(q(z|x) || p(z)) = \frac{1}{2} \sum_{j=1}^{J} \left( \mu_j^2 + \sigma_j^2 - \log(\sigma_j^2) - 1 \right)$$

### 5.2 扩散模型原理

**前向过程**:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

**反向过程**:
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**训练目标**:
$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

**DDIM 采样**:
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

### 5.3 FAST Tokenizer 的 DCT 变换

**DCT-II 变换**:
$$X_k = \sum_{n=0}^{N-1} x_n \cos\left[ \frac{\pi}{N} \left( n + \frac{1}{2} \right) k \right]$$

**逆 DCT**:
$$x_n = \frac{1}{2} X_0 + \sum_{k=1}^{N-1} X_k \cos\left[ \frac{\pi}{N} \left( n + \frac{1}{2} \right) k \right]$$

**量化**:
$$q = \text{clip}\left( \left\lfloor \frac{X + 1}{2} \times N_{\text{bins}} \right\rfloor, 0, N_{\text{bins}}-1 \right)$$

### 5.4 动作分块 (Action Chunking)

**时序一致性**:
$$\mathbf{a}_{t:t+H} = f_\theta(o_t, o_{t-1}, ..., o_{t-K})$$

**执行策略**:
- 预测整个动作块 $\mathbf{a}_{t:t+H}$
- 只执行第一步 $a_t$
- 下一时刻重新预测

**优势**:
- 减少推理频率
- 提高时序平滑性
- 隐式建模未来状态

## 6. LeRobotDataset 详解

### 6.1 数据存储格式

**Parquet 文件** (状态/动作):
```python
import pyarrow as pa
import pyarrow.parquet as pq

# 创建 Parquet 表
table = pa.table({
    "episode_index": pa.array([0, 0, 1, 1, ...]),
    "frame_index": pa.array([0, 1, 0, 1, ...]),
    "timestamp": pa.array([0.0, 0.033, 0.0, 0.033, ...]),
    "observation.state": pa.array([[...], [...], ...]),
    "action": pa.array([[...], [...], ...]),
    "language": pa.array(["pick up the cup", ...])
})

# 保存
pq.write_table(table, "episode_00000.parquet")
```

**视频文件** (MP4):
```python
import av

# 编码视频
container = av.open("video.mp4", "w")
stream = container.add_stream("h264", rate=30)
stream.width = 640
stream.height = 480

for frame in frames:
    packet = stream.encode(frame)
    container.mux(packet)

# 关闭
container.close()
```

### 6.2 数据集工具

**删除 episode**:
```python
from lerobot.datasets.utils import delete_episodes

delete_episodes(
    repo_id="lerobot/my_dataset",
    episode_indices=[0, 5, 10]  # 删除指定 episode
)
```

**分割数据集**:
```python
from lerobot.datasets.utils import split_dataset

split_dataset(
    repo_id="lerobot/my_dataset",
    train_ratio=0.8,
    seed=42
)
```

**合并数据集**:
```python
from lerobot.datasets.utils import merge_datasets

merge_datasets(
    repo_ids=["lerobot/dataset1", "lerobot/dataset2"],
    output_repo_id="lerobot/merged_dataset"
)
```

## 7. 总结与启示

### 7.1 LeRobot 的核心优势

1. **标准化数据格式**:
   - Parquet + MP4 组合
   - 高效的存储和流式加载
   - Hugging Face Hub 集成

2. **统一接口**:
   - Robot 类抽象硬件差异
   - Policy 协议标准化模型接口
   - 评估框架支持多种基准

3. **丰富的策略库**:
   - 支持 IL、RL、VLA 多种范式
   - 预训练模型开箱即用
   - 易于扩展自定义策略

### 7.2 对 vla-training 的启示

1. **数据格式**:
   - 采用 LeRobot V3 格式
   - 使用 Parquet 存储状态/动作
   - 使用 MP4 存储视频

2. **模型接口**:
   - 定义统一的 Policy 协议
   - 支持多种策略类型
   - 提供预训练模型加载

3. **评估框架**:
   - 支持开环和闭环评估
   - 集成标准基准 (LIBERO 等)
   - 提供可视化分析工具

---

*参考文献*:
- LeRobot GitHub: https://github.com/huggingface/lerobot
- LeRobot Documentation: https://huggingface.co/docs/lerobot
- LeRobot ICLR Paper: https://arxiv.org/abs/2602.22818
- ACT Paper: https://arxiv.org/abs/2304.13705
- Diffusion Policy Paper: https://arxiv.org/abs/2303.04137
