# VoxPoser: Composable 3D Value Maps for Robotic Manipulation

**Stanford, 2023** | arXiv:2307.05973

---

## 0. 核心观点与结论

### 研究问题
- 如何组合多个约束进行复杂操作？
- 如何利用语言模型进行 3D 推理？
- 如何实现零样本组合技能？

### 核心贡献
1. **3D 价值图**：将任务分解为 3D 空间约束
2. **语言模型推理**：用 LLM 提取约束并组合
3. **可组合性**：多个约束相乘得到最终价值图
4. **零样本泛化**：新任务无需训练

### 主要结论
- 价值图表示支持约束组合
- LLM 有效提取空间约束
- 零样本组合新任务
- 在长视野任务上表现优异

### 领域启示
- 3D 价值图是有效的任务表示
- LLM 可作为"任务解析器"
- 组合性是实现泛化的关键

---

## 1. 创新点详解

### 核心创新

#### 1.1 3D 价值图
```python
class VoxPoser:
    def __init__(self):
        self.llm = load_llm()  # 语言模型
        self.vlm = load_vlm()  # 视觉语言模型
    
    def compute_value_map(self, scene, task):
        # 1. 用 LLM 提取约束
        constraints = self.llm.extract_constraints(task)
        # 例如："避开红色物体"，"靠近蓝色杯子"
        
        # 2. 对每个约束计算 3D 价值图
        value_maps = []
        for constraint in constraints:
            vm = self.vlm.compute_value_map(scene, constraint)
            value_maps.append(vm)
        
        # 3. 组合价值图 (相乘)
        final_vm = torch.stack(value_maps).prod(dim=0)
        
        return final_vm
    
    def sample_action(self, value_map):
        # 从高价值区域采样动作
        prob = softmax(value_map)
        action = sample(prob)
        return action
```

#### 1.2 约束提取
```
任务："把苹果放到碗里，避开红色积木"

LLM 输出：
- 约束 1: "靠近苹果" (抓取)
- 约束 2: "靠近碗" (放置)
- 约束 3: "远离红色积木" (避障)
```

#### 1.3 价值图组合
$$V_{\text{final}}(\mathbf{x}) = \prod_{i} V_i(\mathbf{x})^{w_i}$$

### 方法流程

```
┌─────────────────────────────────────────────────────────────┐
│                    VoxPoser 流程                             │
├─────────────────────────────────────────────────────────────┤
│  1. 任务解析：                                               │
│     输入："把苹果放到碗里"                                   │
│     LLM → 子任务序列：[抓取苹果，移动到碗，放置]              │
├─────────────────────────────────────────────────────────────┤
│  2. 约束提取 (每子任务)：                                     │
│     "抓取苹果" → [靠近苹果，避开障碍物]                      │
│     "移动到碗" → [靠近碗，避开障碍物]                        │
│     "放置" → [在碗上方，松开夹爪]                            │
├─────────────────────────────────────────────────────────────┤
│  3. 价值图计算 (每约束)：                                     │
│     VLM 输入：场景图像 + 约束文本                             │
│     输出：3D 价值图 (每个体素的价值)                          │
├─────────────────────────────────────────────────────────────┤
│  4. 价值图组合：                                             │
│     V_final = V_1 × V_2 × ... × V_n                         │
├─────────────────────────────────────────────────────────────┤
│  5. 动作采样：                                               │
│     从 V_final 采样高价值动作                                │
│     执行动作，更新场景                                        │
└─────────────────────────────────────────────────────────────┘
```

### 数学表达

#### 价值图表示
$$V: \mathbb{R}^3 \rightarrow [0, 1]$$

每个 3D 位置的价值表示执行动作的适宜性。

#### 约束组合
$$V_{\text{compose}}(\mathbf{x}) = \left(\prod_{i=1}^n V_i(\mathbf{x})\right)^{1/n}$$

#### 动作采样
$$\mathbf{a} \sim \frac{\exp(V(\mathbf{x}) / \tau)}{\sum_{\mathbf{x}'} \exp(V(\mathbf{x}') / \tau)}$$

### 实验结论

#### 性能对比
| 方法 | 简单任务 | 组合任务 | 长视野任务 |
|------|----------|----------|------------|
| BC | 85% | 45% | 20% |
| VLA | 90% | 60% | 40% |
| VoxPoser | 88% | 82% | 75% |

#### 零样本组合
- 新约束组合：75% 成功率
- 新物体：70% 成功率
- 新场景：65% 成功率

---

## 2. 公式与流程对照表

| 步骤 | 数学公式 | 代码实现 | 说明 |
|------|----------|----------|------|
| 1. 任务解析 | $\text{subtasks} = \text{LLM}(task)$ | `subtasks = llm.parse(task)` | 分解任务 |
| 2. 约束提取 | $\text{constraints} = \text{LLM}(subtask)$ | `constraints = llm.extract(subtask)` | 提取约束 |
| 3. 价值图计算 | $V_i = \text{VLM}(scene, c_i)$ | `vm = vlm.compute(scene, constraint)` | VLM 推理 |
| 4. 组合 | $V_{\text{final}} = \prod_i V_i$ | `final_vm = torch.stack(vms).prod(dim=0)` | 相乘组合 |
| 5. 动作采样 | $\mathbf{a} \sim \text{softmax}(V/\tau)$ | `action = sample(softmax(vm / temp))` | 采样动作 |

---

## 3. 迁移部署指南

### 环境依赖
```bash
python >= 3.9
torch >= 2.0
transformers >= 4.30  # LLM
openai >= 0.27  # 或本地 LLM
```

### 系统配置
```yaml
llm:
  model: gpt-4  # 或本地 Llama
  temperature: 0.0  # 确定性输出

vlm:
  model: clip-vit
  resolution: 224

voxel:
  size: 0.01
  bounds: [[-0.5, 0.5], [-0.5, 0.5], [0, 1.0]]
```

### 推理示例
```python
from voxposer import VoxPoser

# 初始化
vp = VoxPoser(llm='gpt-4', vlm='clip')

# 设置场景
scene = {
    'rgb': camera_image,
    'depth': depth_image,
    'objects': detected_objects
}

# 执行任务
task = "put the apple in the bowl, avoiding the red blocks"
trajectory = vp.execute(scene, task)

# 输出：动作序列
for action in trajectory:
    robot.execute(action)
```

### LLM 提示词示例
```python
constraint_prompt = """
Extract spatial constraints from the task.
Task: {task}

Output format (JSON):
{{
  "constraints": [
    {{"type": "proximity", "object": "apple", "relation": "close"}},
    {{"type": "avoidance", "object": "red block", "relation": "far"}}
  ]
}}
"""
```

### 常见问题

#### Q1: LLM 输出不稳定
**解决：** 降低 temperature，使用 few-shot 示例

#### Q2: 价值图计算慢
**解决：** 缓存 VLM 输出，降低分辨率

---

## 参考资源

- **论文**: https://arxiv.org/abs/2307.05973
- **项目页**: https://voxposer.github.io/
- **代码**: https://github.com/huangwl18/VoxPoser

---

*最后更新：2026-03-03*
