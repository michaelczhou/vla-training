# 具身智能方法选型推荐指南

**创建时间**: 2026-03-03  
**目标读者**: 研究者、工程师、技术决策者

---

## 快速选型表

| 应用场景 | 推荐方法 | 理由 | 难度 |
|----------|----------|------|------|
| **快速原型** | ACT | 简单高效，代码开源 | ⭐⭐ |
| **精细操作** | Diffusion Policy | 多模态建模，平滑轨迹 | ⭐⭐⭐ |
| **语义任务** | RT-2 / OpenVLA | 知识迁移，语义理解 | ⭐⭐⭐⭐ |
| **3D 抓取** | PerAct | 3D 感知，6DoF 输出 | ⭐⭐⭐ |
| **组合任务** | VoxPoser | 零样本组合，长视野 | ⭐⭐⭐⭐ |
| **通用策略** | Octo | 跨机器人，开源 | ⭐⭐⭐ |
| **资源有限** | ACT / Octo | 单 GPU 可训练 | ⭐⭐ |
| **追求 SOTA** | RT-2 / Diffusion | 最佳性能 | ⭐⭐⭐⭐⭐ |

---

## 按场景推荐

### 1. 工业分拣场景

**需求**: 高速、高成功率、固定场景

**推荐方案**:
```
首选：ACT
备选：Diffusion Policy

理由:
- 动作分块保证时序平滑
- 高成功率 (90%+)
- 推理速度快 (50 Hz)
- 易于部署
```

**配置建议**:
```yaml
model: ACT
chunk_size: 100
camera: 固定 RGB-D
training_data: 50-100 演示/任务
gpu: 1x RTX 3090
```

---

### 2. 家庭服务场景

**需求**: 语义理解、泛化能力、安全

**推荐方案**:
```
首选：OpenVLA
备选：RT-2 (如有资源)

理由:
- 语义理解能力强
- 可处理开放指令
- 开源可定制
- 安全可控
```

**配置建议**:
```yaml
model: OpenVLA-7B
quantization: INT8
camera: 移动 RGB-D
training: 微调 (LoRA)
gpu: 8x A100 (训练) / 1x RTX 4090 (推理)
```

---

### 3. 实验室研究场景

**需求**: 灵活性、可解释性、快速迭代

**推荐方案**:
```
首选：Octo
备选：ACT

理由:
- 模块化设计
- 易于修改
- 开源完整
- 社区支持好
```

**配置建议**:
```yaml
model: Octo-Base
modular: True
robot: 任意支持平台
training: 从头训练或微调
gpu: 8x V100
```

---

### 4. 3D 操作场景

**需求**: 精确 3D 定位、6DoF 抓取、避障

**推荐方案**:
```
首选：PerAct
备选：ACT + 3D 感知

理由:
- 原生 3D 处理
- 6DoF 输出
- 避障能力强
```

**配置建议**:
```yaml
model: PerAct
voxel_size: 0.01
camera: RGB-D 或多视角
training_data: 1 万 + 抓取
gpu: 1x RTX 3090
```

---

### 5. 长视野任务场景

**需求**: 多步骤推理、组合技能、零样本

**推荐方案**:
```
首选：VoxPoser
备选：RT-2 + 规划器

理由:
- 零样本组合
- 长视野规划
- 可解释性强
```

**配置建议**:
```yaml
system: VoxPoser
llm: GPT-4 / Llama-70B
vlm: CLIP
robot: 任意
gpu: 推理为主
```

---

### 6. 资源受限场景

**需求**: 低成本、低功耗、边缘部署

**推荐方案**:
```
首选：ACT (量化版)
备选：小型 Octo

理由:
- 模型小 (~50M)
- 可量化到 INT8
- 边缘设备可运行
```

**配置建议**:
```yaml
model: ACT-Small
quantization: INT8
inference: TensorRT
hardware: Jetson Orin / 树莓派 5
```

---

## 按资源推荐

### 计算资源充足 (TPU Pod / 多节点 GPU)

**推荐**: RT-2 级别 VLA
```
- 训练：TPU v4 Pod 或 64x A100
- 数据：100 万 + 轨迹
- 预期：SOTA 性能
- 时间：2-4 周
```

### 中等资源 (8-16 GPU)

**推荐**: OpenVLA / Octo
```
- 训练：8x A100
- 数据：10-100 万轨迹
- 预期：接近 SOTA
- 时间：3-7 天
```

### 有限资源 (1-4 GPU)

**推荐**: ACT / Diffusion Policy
```
- 训练：1-4x GPU
- 数据：50-200 演示/任务
- 预期：任务特定 SOTA
- 时间：数小时 -1 天
```

### 极少资源 (CPU / 边缘)

**推荐**: 量化 ACT / 蒸馏模型
```
- 训练：云端训练后部署
- 推理：边缘设备
- 预期：可用性能
- 时间：部署优化 1-2 周
```

---

## 按团队规模推荐

### 小团队 (1-3 人)

**推荐**: 使用开源模型 + 微调
```
- 模型：OpenVLA / Octo / ACT
- 策略：站在巨人肩膀上
- 重点：应用创新而非基础模型
- 时间：2-4 周出原型
```

### 中团队 (4-10 人)

**推荐**: 改进现有方法 + 新场景
```
- 模型：基于 ACT/Diffusion 改进
- 策略：方法创新 + 场景创新
- 重点：特定场景 SOTA
- 时间：2-3 月出成果
```

### 大团队 (10+ 人)

**推荐**: 自研 VLA / 新范式
```
- 模型：自研或深度改进
- 策略：引领方向
- 重点：基础模型创新
- 时间：6-12 月
```

---

## 技术栈推荐

### 深度学习框架

| 框架 | 适用场景 | 推荐度 |
|------|----------|--------|
| PyTorch | 研究/原型 | ⭐⭐⭐⭐⭐ |
| JAX/Flax | 大规模训练 | ⭐⭐⭐⭐ |
| TensorFlow | 生产部署 | ⭐⭐⭐ |

### 机器人接口

| 接口 | 适用平台 | 推荐度 |
|------|----------|--------|
| ROS 2 | 研究/工业 | ⭐⭐⭐⭐⭐ |
| 直接控制 | 简单场景 | ⭐⭐⭐⭐ |
| Isaac Gym | 仿真训练 | ⭐⭐⭐⭐ |

### 部署工具

| 工具 | 适用场景 | 推荐度 |
|------|----------|--------|
| TensorRT | NVIDIA GPU | ⭐⭐⭐⭐⭐ |
| ONNX Runtime | 跨平台 | ⭐⭐⭐⭐ |
| TorchScript | PyTorch 生态 | ⭐⭐⭐⭐ |

---

## 学习路线推荐

### 入门 (0-3 个月)

```
1. 学习基础
   - Transformer 架构
   - 行为克隆
   - PyTorch 基础

2. 复现简单方法
   - ACT (代码简单)
   - 在仿真环境测试

3. 理解核心概念
   - 动作分块
   - 时序集成
   - 多模态融合
```

### 进阶 (3-6 个月)

```
1. 深入 VLA
   - 学习 OpenVLA / Octo
   - 理解 VLA 架构
   - 微调预训练模型

2. 掌握扩散模型
   - Diffusion Policy
   - 扩散理论基础
   - 加速采样技术

3. 实践项目
   - 真实机器人部署
   - 完整 pipeline
```

### 高级 (6-12 个月)

```
1. 方法创新
   - 改进现有方法
   - 提出新架构
   - 发表论文

2. 大规模训练
   - 分布式训练
   - 数据收集 pipeline
   - 模型优化

3. 系统整合
   - 完整机器人系统
   - 人机交互
   - 产品化
```

---

## 常见陷阱与建议

### ❌ 陷阱 1: 盲目追求大模型

**问题**: 直接上 7B+ VLA，忽略实际需求

**建议**: 
- 从 ACT 等小模型开始
- 验证任务需求
- 逐步升级

### ❌ 陷阱 2: 忽视数据质量

**问题**: 大量低质数据 vs 少量高质数据

**建议**:
- 数据质量 > 数量
- 仔细清洗数据
- 多样性重要

### ❌ 陷阱 3: 忽视评估

**问题**: 只在训练集评估

**建议**:
- 建立测试集
- 零样本评估
- 真实世界测试

### ❌ 陷阱 4: 忽视部署

**问题**: 只关注训练，不考虑推理

**建议**:
- 早期考虑部署
- 优化推理速度
- 量化/蒸馏

---

## 资源链接

### 代码仓库

| 方法 | 仓库 | Star |
|------|------|------|
| ACT | github.com/tonyzhaozh/act | 2000+ |
| Diffusion Policy | github.com/columbia-ai-robotics/diffusion_policy | 1500+ |
| Octo | github.com/octo-models/octo | 1000+ |
| OpenVLA | github.com/openvla/openvla | 500+ |
| PerAct | github.com/peract/peract | 300+ |
| VoxPoser | github.com/huangwl18/VoxPoser | 800+ |

### 数据集

| 数据集 | 规模 | 链接 |
|--------|------|------|
| Open X-Embodiment | 100 万 + | open-x-embodiment.github.io |
| RT-1 Dataset | 17 万 | robotics-transformer1.github.io |
| Bridge Data | 10 万 | bridge-data.github.io |

### 教程

- [HuggingFace Robot Learning](https://huggingface.co/robot-learning)
- [Deep RL Course](https://huggingface.co/deep-rl-course)
- [Stanford CS329S](https://stanford-cs329s.github.io/)

---

*最后更新：2026-03-03*
