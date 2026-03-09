# π0 引用论文深度研究报告

## 项目概述

本研究以 Physical Intelligence 团队的 **π0 (pi0)** 论文为起点，系统性地追踪和分析引用该论文的后续研究工作。

## π0 基础信息

**论文标题**: A Vision-Language-Action Flow Model for General Robot Control

**作者团队**: Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, Ury Zhilinsky

**机构**: Physical Intelligence (π)

**发表时间**: 
- v1: 2024 年 10 月 31 日
- v2: 2026 年 1 月 8 日

**核心贡献**:
- 提出首个 Vision-Language-Action (VLA) Flow Model
- 使用 flow matching 替代传统扩散模型进行动作生成
- 实现跨机器人形态的零样本泛化能力
- 在真实机器人平台上验证了大规模预训练 VLA 的有效性

## 研究方法论

### 引用追踪来源
1. arxiv.org - 直接引用追踪
2. Google Scholar - "Cited by" 功能
3. Semantic Scholar - 引用网络分析
4. PapersWithCode - 代码关联

### 筛选标准（权重排序）
| 标准 | 权重 | 说明 |
|------|------|------|
| 被引量 | 40% | 引用次数、学术影响力 |
| 相关性 | 30% | 与 VLA/机器人学习的相关度 |
| 创新性 | 20% | 方法新颖性、技术突破 |
| 实用性 | 10% | 代码开源、可复现性 |

## 输出文件结构

```
pi0-citations/
├── README.md                    # 本文件
├── papers/                      # 单篇论文深度分析
│   ├── 01-VLS.md
│   ├── 02-StreamVLA.md
│   ├── 03-RDT2.md
│   ├── 04-TwinBrainVLA.md
│   ├── 05-Being-H0.5.md
│   ├── 06-mimic-video.md
│   ├── 07-Motus.md
│   ├── 08-Compose-Policies.md
│   ├── 09-CLAW.md
│   └── 10-EO-1.md
├── reports/                     # 汇总报告
│   ├── citation-timeline.md     # 技术演进时间线
│   ├── comparison-matrix.md     # 对比分析矩阵
│   └── citation-graph.md        # 引用关系图谱
└── bibliography.bib             # BibTeX 引用
```

## 已识别的关键引用论文

| # | 论文标题 | 机构 | 时间 | 核心方向 |
|---|----------|------|------|----------|
| 1 | VLS: Steering Pretrained Robot Policies via Vision-Language Models | UW/Ranjay Krishna | 2026.02 | VLA  Steering/修正 |
| 2 | StreamVLA: Breaking the Reason-Act Cycle via Completion-State Gating | - | 2026.02 | 双系统架构 |
| 3 | RDT2: Exploring the Scaling Limit of UMI Data | - | 2026.02 | 数据扩展/跨形态 |
| 4 | TwinBrainVLA: Asymmetric Mixture-of-Transformers | - | 2026.01 | 架构创新 |
| 5 | Being-H0.5: Scaling Human-Centric Robot Learning | - | 2026.01 | 人机协作 |
| 6 | mimic-video: Video-Action Models Beyond VLAs | - | 2025.12 | 视频 - 动作模型 |
| 7 | Motus: A Unified Latent Action World Model | - | 2025.12 | 潜在动作世界模型 |
| 8 | Compose Your Policies! Distribution-level Composition | - | 2025.10 | 策略组合 |
| 9 | CLAW: Weight-Aware Robotic Grasping | - | 2025.09 | 重量感知抓取 |
| 10 | EO-1: Open Unified Embodied Foundation Model | - | 2025.08 | 开源基础模型 |

## 技术演进趋势

### 2024 Q4 - π0 奠基
- Flow Matching 引入 VLA
- 大规模预训练验证

### 2025 Q3-Q4 - 架构扩展
- 双系统架构 (StreamVLA)
- 潜在动作模型 (Motus)
- 视频 - 动作模型 (mimic-video)

### 2026 Q1-Q2 - 应用深化
- Steering/修正方法 (VLS)
- 跨形态泛化 (RDT2)
- 人机协作 (Being-H0.5)

## 持续更新

本报告将持续更新，追踪 π0 后续引用论文的最新进展。

---
*最后更新: 2026-03-03*
*研究状态: 进行中*
