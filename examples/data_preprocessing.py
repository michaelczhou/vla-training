"""
Data Preprocessing Example
==========================
VLA 数据预处理流程示例

展示如何处理机器人操作数据，包括：
- 图像预处理
- 动作归一化
- 数据增强
- 数据加载器
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as T
import sys
sys.path.insert(0, '/root/.openclaw/workspace/vla-training')


class VLADataset(Dataset):
    """
    VLA 数据集类
    
    处理机器人操作数据，格式：
    {
        "image": (H, W, 3) - 摄像头图像
        "state": (S,) - 机器人状态
        "action": (T, A) - 未来 T 步的动作
        "language": str - 语言指令
    }
    """
    
    def __init__(
        self,
        data_path,
        image_size=(224, 224),
        action_dim=7,
        chunk_size=10,
        augment=True,
        normalize=True
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据路径
            image_size: 图像尺寸 (H, W)
            action_dim: 动作维度
            chunk_size: 动作块大小
            augment: 是否数据增强
            normalize: 是否归一化
        """
        self.data_path = data_path
        self.image_size = image_size
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.augment = augment
        self.normalize = normalize
        
        # 图像变换
        self.image_transform = self._build_image_transform()
        
        # 动作归一化参数
        self.action_mean = None
        self.action_std = None
        
        # 加载数据索引
        self.samples = self._load_data_index()
        
    def _build_image_transform(self):
        """构建图像变换流程"""
        transforms = []
        
        # 基础变换
        transforms.append(T.Resize(self.image_size))
        transforms.append(T.ToTensor())
        
        # 数据增强
        if self.augment:
            transforms.extend([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomHorizontalFlip(p=0.5),
            ])
        
        # 归一化 (ImageNet 统计)
        if self.normalize:
            transforms.append(T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        
        return T.Compose(transforms)
    
    def _load_data_index(self):
        """加载数据索引"""
        # 这里应该从实际数据文件加载
        # 示例返回模拟数据
        return list(range(1000))
    
    def __len__(self):
        return len(self.samples)
    
    def _load_image(self, idx):
        """加载图像"""
        # 示例：生成随机图像
        # 实际应用中从文件加载
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        image = Image.fromarray(image)
        return image
    
    def _load_state(self, idx):
        """加载机器人状态"""
        # 示例：生成随机状态
        # 实际应用中从数据加载
        return np.random.randn(14)  # 14维状态
    
    def _load_action(self, idx):
        """加载动作序列"""
        # 示例：生成随机动作
        # 实际应用中从数据加载
        return np.random.randn(self.chunk_size, self.action_dim)
    
    def _load_language(self, idx):
        """加载语言指令"""
        # 示例指令列表
        instructions = [
            "pick up the red block",
            "place the blue cup on the table",
            "open the drawer",
            "push the button",
            "grasp the bottle",
        ]
        return instructions[idx % len(instructions)]
    
    def _normalize_action(self, action):
        """归一化动作"""
        if self.action_mean is not None and self.action_std is not None:
            action = (action - self.action_mean) / self.action_std
        return action
    
    def _denormalize_action(self, action):
        """反归一化动作"""
        if self.action_mean is not None and self.action_std is not None:
            action = action * self.action_std + self.action_mean
        return action
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 加载数据
        image = self._load_image(idx)
        state = self._load_state(idx)
        action = self._load_action(idx)
        language = self._load_language(idx)
        
        # 图像预处理
        image_tensor = self.image_transform(image)
        
        # 状态预处理
        state_tensor = torch.from_numpy(state).float()
        
        # 动作预处理
        action_tensor = torch.from_numpy(action).float()
        action_tensor = self._normalize_action(action_tensor)
        
        return {
            "image": image_tensor,
            "state": state_tensor,
            "action": action_tensor,
            "language": language,
        }


class DataCollator:
    """
    数据收集器
    
    将多个样本组合成一个 batch
    """
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        """组合 batch"""
        images = torch.stack([item["image"] for item in batch])
        states = torch.stack([item["state"] for item in batch])
        actions = torch.stack([item["action"] for item in batch])
        languages = [item["language"] for item in batch]
        
        # 如果有 tokenizer，对语言指令进行编码
        if self.tokenizer is not None:
            language_tokens = self.tokenizer(
                languages,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
        else:
            language_tokens = None
        
        return {
            "images": images,
            "states": states,
            "actions": actions,
            "languages": languages,
            "language_tokens": language_tokens,
        }


def create_dataloader(
    data_path,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    **dataset_kwargs
):
    """
    创建数据加载器
    
    Args:
        data_path: 数据路径
        batch_size: batch 大小
        num_workers: 数据加载线程数
        shuffle: 是否打乱数据
        **dataset_kwargs: 数据集其他参数
    
    Returns:
        DataLoader 实例
    """
    dataset = VLADataset(data_path, **dataset_kwargs)
    collator = DataCollator()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    
    return dataloader


def demo_data_preprocessing():
    """演示数据预处理"""
    print("=" * 60)
    print("VLA 数据预处理示例")
    print("=" * 60)
    
    # 创建数据集
    print("\n1. 创建数据集...")
    dataset = VLADataset(
        data_path="/path/to/data",
        image_size=(224, 224),
        action_dim=7,
        chunk_size=10,
        augment=True,
        normalize=True
    )
    print(f"   数据集大小: {len(dataset)}")
    
    # 获取单个样本
    print("\n2. 获取单个样本...")
    sample = dataset[0]
    print(f"   图像形状: {sample['image'].shape}")
    print(f"   状态形状: {sample['state'].shape}")
    print(f"   动作形状: {sample['action'].shape}")
    print(f"   语言指令: {sample['language']}")
    
    # 创建数据加载器
    print("\n3. 创建数据加载器...")
    dataloader = create_dataloader(
        data_path="/path/to/data",
        batch_size=4,
        num_workers=0,  # 示例用 0
        shuffle=True,
        image_size=(224, 224),
        action_dim=7,
        chunk_size=10,
    )
    print(f"   Batch 数量: {len(dataloader)}")
    
    # 获取一个 batch
    print("\n4. 获取一个 batch...")
    batch = next(iter(dataloader))
    print(f"   图像 batch: {batch['images'].shape}")
    print(f"   状态 batch: {batch['states'].shape}")
    print(f"   动作 batch: {batch['actions'].shape}")
    print(f"   语言指令: {batch['languages']}")
    
    # 数据增强效果
    print("\n5. 数据增强效果...")
    print("   - 颜色抖动 (ColorJitter)")
    print("   - 随机水平翻转 (RandomHorizontalFlip)")
    print("   - 图像归一化 (ImageNet 统计)")
    print("   - 动作归一化 (Z-score)")
    
    print("\n" + "=" * 60)
    print("✅ 数据预处理示例完成！")
    print("=" * 60)


def demo_action_normalization():
    """演示动作归一化"""
    print("\n" + "=" * 60)
    print("动作归一化示例")
    print("=" * 60)
    
    # 生成模拟动作数据
    np.random.seed(42)
    actions = np.random.randn(100, 10, 7) * 0.5 + np.array([
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5]  # 每个关节的均值
    ])
    
    print(f"\n原始动作统计:")
    print(f"   均值: {actions.mean(axis=(0, 1))}")
    print(f"   标准差: {actions.std(axis=(0, 1))}")
    print(f"   范围: [{actions.min():.3f}, {actions.max():.3f}]")
    
    # 归一化
    action_mean = actions.mean(axis=(0, 1), keepdims=True)
    action_std = actions.std(axis=(0, 1), keepdims=True)
    actions_normalized = (actions - action_mean) / action_std
    
    print(f"\n归一化后统计:")
    print(f"   均值: {actions_normalized.mean(axis=(0, 1))}")
    print(f"   标准差: {actions_normalized.std(axis=(0, 1))}")
    print(f"   范围: [{actions_normalized.min():.3f}, {actions_normalized.max():.3f}]")
    
    # 反归一化
    actions_denormalized = actions_normalized * action_std + action_mean
    print(f"\n反归一化后与原数据差异: {np.abs(actions - actions_denormalized).max():.6f}")


def demo_image_preprocessing():
    """演示图像预处理"""
    print("\n" + "=" * 60)
    print("图像预处理示例")
    print("=" * 60)
    
    # 生成模拟图像
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    image_pil = Image.fromarray(image)
    
    print(f"\n原始图像:")
    print(f"   尺寸: {image.shape}")
    print(f"   像素范围: [0, 255]")
    
    # 预处理流程
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image_tensor = transform(image_pil)
    
    print(f"\n预处理后:")
    print(f"   张量形状: {image_tensor.shape}")
    print(f"   像素范围: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    print(f"   数据类型: {image_tensor.dtype}")


if __name__ == "__main__":
    demo_data_preprocessing()
    demo_action_normalization()
    demo_image_preprocessing()
    
    print("\n" + "=" * 60)
    print("提示: 实际使用时需要:")
    print("  1. 替换数据加载逻辑 (从真实数据文件读取)")
    print("  2. 配置 tokenizer (用于语言指令编码)")
    print("  3. 根据数据集计算动作归一化参数")
    print("=" * 60)
