# 具身智能经典论文深度研究

**创建时间**: 2026-03-03  
**研究周期**: 2026-03-03  
**论文数量**: 10 篇

---

## 📚 项目概述

本项目深度研究具身智能 (Embodied AI) 领域 10 篇经典论文，涵盖从基础奠基工作到最新 VLA 范式的完整技术演进脉络。

### 研究目标

1. **理解核心技术**: 深入理解每篇论文的创新点、方法细节
2. **掌握技术演进**: 梳理技术发展脉络和关键转折点
3. **实践应用**: 提供代码示例和部署指南
4. **选型参考**: 为不同场景提供方法选型建议

---

## 📁 目录结构

```
embodied-ai-classics/
├── README.md                          # 本文件
├── papers/                            # 10 篇论文详细分析
│   ├── 01-RT1-Robotics-Transformer.md
│   ├── 02-GATO-Generalist-Agent.md
│   ├── 03-PaLM-E-Embodied-Multimodal.md
│   ├── 04-RT2-Vision-Language-Action.md
│   ├── 05-OpenVLA-OpenSource-VLA.md
│   ├── 06-Octo-OpenSource-Generalist.md
│   ├── 07-ACT-Action-Chunking.md
│   ├── 08-Diffusion-Policy.md
│   ├── 09-PerAct-Perceiver-Actor-3D.md
│   └── 10-VoxPoser-Composable-3D.md
├── summaries/                         # 汇总对比报告
│   ├── 01-method-comparison.md        # 方法对比表
│   ├── 02-timeline.md                 # 时间线图
│   └── 03-recommendation-guide.md     # 推荐路线
└── code/                              # 代码示例库
    └── README.md                      # 代码说明
```

---

## 📖 10 篇经典论文

### 奠基性工作 (2022)

| # | 论文 | 机构 | 核心贡献 | 影响力 |
|---|------|------|----------|--------|
| 1 | **RT-1** | Google | 机器人 Transformer + 大规模数据 | ⭐⭐⭐⭐⭐ |
| 2 | **GATO** | DeepMind | 通用智能体架构 | ⭐⭐⭐⭐ |
| 3 | **PaLM-E** | Google | 具身多模态 LLM | ⭐⭐⭐⭐⭐ |

### VLA 核心架构 (2023-2024)

| # | 论文 | 机构 | 核心贡献 | 影响力 |
|---|------|------|----------|--------|
| 4 | **RT-2** | Google | VLA 知识迁移 | ⭐⭐⭐⭐⭐ |
| 5 | **OpenVLA** | 社区 | 开源 VLA | ⭐⭐⭐⭐ |
| 6 | **Octo** | Berkeley | 开源通用策略 | ⭐⭐⭐⭐ |

### 动作生成方法 (2023)

| # | 论文 | 机构 | 核心贡献 | 影响力 |
|---|------|------|----------|--------|
| 7 | **ACT** | Stanford | 动作分块 + 时序集成 | ⭐⭐⭐⭐⭐ |
| 8 | **Diffusion Policy** | Columbia | 扩散动作生成 | ⭐⭐⭐⭐⭐ |

### 3D 感知与操作 (2023)

| # | 论文 | 机构 | 核心贡献 | 影响力 |
|---|------|------|----------|--------|
| 9 | **PerAct** | - | 3D Perceiver | ⭐⭐⭐ |
| 10 | **VoxPoser** | Stanford | 3D 价值图组合 | ⭐⭐⭐⭐ |

---

## 🔑 核心发现

### 1. VLA 成为主流范式

从 RT-1 (2022) 到 RT-2/OpenVLA (2023-2024)，VLA (Vision-Language-Action) 已成为具身智能的主流架构：

```
视觉编码 (ViT) + 语言模型 (LLM) → 动作输出
```

### 2. 动作分块是关键改进

ACT 和 Octo 都使用动作分块 (Action Chunking)，显著提升时序一致性和成功率：

- 无分块：~60% 成功率
- 有分块：~90% 成功率

### 3. 扩散模型展现优势

Diffusion Policy 在多模态动作建模上优于传统 VAE/GAN 方法：

- VAE: ~80% 成功率
- Diffusion: ~92% 成功率

### 4. 开源生态加速发展

2023 年后，OpenVLA、Octo 等开源项目大幅降低研究门槛：

- 训练时间：从数周到数天
- 计算需求：从 TPU Pod 到单节点
- 代码可用性：从闭源到完全开源

### 5. 零样本组合是新方向

VoxPoser 展示无需训练、通过 LLM 组合约束完成新任务的可能性：

- 新任务组合：75% 成功率
- 无需额外训练

---

## 📊 技术对比速览

### 架构选择

| 需求 | 推荐方法 |
|------|----------|
| 语义理解 | RT-2 / OpenVLA |
| 精细操作 | ACT / Diffusion |
| 3D 抓取 | PerAct |
| 组合任务 | VoxPoser |
| 快速原型 | ACT |
| 通用策略 | Octo |

### 资源需求

| 方法 | GPU 需求 | 训练时间 | 开源 |
|------|----------|----------|------|
| RT-2 | TPU Pod | 数周 | ❌ |
| OpenVLA | 8x A100 | 7 天 | ✅ |
| ACT | 1x GPU | 数小时 | ✅ |
| Octo | 8x V100 | 3 天 | ✅ |

---

## 🚀 快速开始

### 入门路线 (推荐新手)

```
1. 阅读 ACT 论文 (最简单有效)
   ↓
2. 复现 ACT 代码 (官方实现)
   ↓
3. 在仿真环境测试
   ↓
4. 部署到真实机器人
   ↓
5. 学习更高级方法 (VLA 等)
```

### 进阶路线 (有研究者)

```
1. 深入理解 VLA 架构 (RT-2, OpenVLA)
   ↓
2. 微调预训练模型
   ↓
3. 方法改进创新
   ↓
4. 大规模训练
   ↓
5. 发表论文/开源项目
```

---

## 📈 技术演进趋势

```
2022                    2023                    2024
 │                       │                       │
 ├─ 专用 Transformer     ├─ VLA 范式             ├─ 开源生态
 │  (RT-1)              │  (RT-2, PaLM-E)       │  (OpenVLA, Octo)
 │                       │                       │
 ├─ 通用智能体           ├─ 动作生成改进         ├─ 效率优化
 │  (GATO)              │  (ACT, Diffusion)     │  (量化，蒸馏)
 │                       │                       │
 │                       ├─ 3D 感知              ├─ 零样本组合
 │                       │  (PerAct)            │  (VoxPoser 改进)
 │                       │                       │
 │                       └─ 价值图方法           │
 │                          (VoxPoser)           │
 │
 ▼
VLA + 动作分块 + 扩散 → 当前 SOTA 组合
```

---

## 💡 关键启示

### 对研究者

1. **站在巨人肩膀上**: 使用 OpenVLA/Octo 等开源模型，避免重复造轮子
2. **重视数据质量**: 100 条高质量演示 > 1000 条低质量数据
3. **动作分块必备**: 几乎成为精细操作的标准配置
4. **关注效率**: 推理速度决定能否落地

### 对工程师

1. **从简单开始**: ACT 是最容易上手的 SOTA 方法
2. **考虑部署**: 早期考虑量化、加速
3. **评估真实场景**: 仿真成功≠真实成功
4. **安全第一**: 特别是人机协作场景

### 对决策者

1. **技术成熟度**: VLA 仍在发展中，动作分块已成熟
2. **投入产出比**: 小团队建议用开源模型 + 微调
3. **长期规划**: 关注零样本、组合能力发展
4. **人才储备**: 培养 VLA + 机器人复合人才

---

## 🔗 资源链接

### 论文链接

| 论文 | arXiv | 项目页 |
|------|-------|--------|
| RT-1 | [2212.06817](https://arxiv.org/abs/2212.06817) | [链接](https://robotics-transformer1.github.io) |
| GATO | [2205.06173](https://arxiv.org/abs/2205.06173) | [链接](https://deepmind.google/discover/blog/gato/) |
| PaLM-E | [2303.03378](https://arxiv.org/abs/2303.03378) | [链接](https://palm-e.github.io) |
| RT-2 | [2307.15818](https://arxiv.org/abs/2307.15818) | [链接](https://robotics-transformer2.github.io) |
| OpenVLA | [2406.09246](https://arxiv.org/abs/2406.09246) | [链接](https://github.com/openvla/openvla) |
| Octo | [2308.04131](https://arxiv.org/abs/2308.04131) | [链接](https://octo-models.github.io) |
| ACT | [2304.13705](https://arxiv.org/abs/2304.13705) | [链接](https://action-chunking.github.io) |
| Diffusion Policy | [2303.04137](https://arxiv.org/abs/2303.04137) | [链接](https://diffusion-policy.github.io) |
| PerAct | [2306.17817](https://arxiv.org/abs/2306.17817) | [链接](https://github.com/peract/peract) |
| VoxPoser | [2307.05973](https://arxiv.org/abs/2307.05973) | [链接](https://voxposer.github.io) |

### 代码仓库

- **OpenVLA**: https://github.com/openvla/openvla
- **Octo**: https://github.com/octo-models/octo
- **ACT**: https://github.com/tonyzhaozh/act
- **Diffusion Policy**: https://github.com/columbia-ai-robotics/diffusion_policy

### 数据集

- **Open X-Embodiment**: https://open-x-embodiment.github.io
- **Bridge Data**: https://bridge-data.github.io

---

## 📝 更新日志

- **2026-03-03**: 初始版本，完成 10 篇论文分析 + 3 份汇总报告 + 代码示例

---

## 🙏 致谢

感谢所有开源论文和代码的作者，你们的工作推动了具身智能领域的发展！

---

## 📄 许可证

本研究报告采用 CC BY-NC-SA 4.0 许可证。

---

*最后更新：2026-03-03*
