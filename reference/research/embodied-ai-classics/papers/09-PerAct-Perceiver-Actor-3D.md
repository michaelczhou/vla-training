# PerAct: Perceiver-Actor for 3D Affordance Detection

**2023** | arXiv:2306.17817

---

## 0. 核心观点与结论

### 研究问题
- 如何高效处理 3D 体素化机器人输入？
- 如何从 3D 感知直接输出动作？
- 如何降低 3D  Transformer 计算复杂度？

### 核心贡献
1. **Perceiver-Actor 架构**：高效 3D 视觉 - 动作 Transformer
2. **体素化输入**：直接处理 3D 点云/深度
3. **交叉注意力**：稀疏查询降低计算量
4. **6DoF 抓取**：端到端 3D 抓取位姿预测

### 主要结论
- Perceiver 架构显著降低计算量
- 3D 体素化提供丰富几何信息
- 在 3D 抓取任务上达到 SOTA
- 可泛化到新物体

### 领域启示
- 3D 感知对精细操作重要
- 高效 Transformer 架构是关键
- 为 3D 操作任务提供新方案

---

## 1. 创新点详解

### 核心创新

#### 1.1 Perceiver 架构
```python
class PerceiverActor(nn.Module):
    def __init__(self):
        # 3D 体素编码
        self.voxel_encoder = VoxNet()
        
        # Perceiver 编码器 (交叉注意力)
        self.perceiver = PerceiverEncoder(
            num_latents=512,  # 稀疏潜变量
            d_model=512
        )
        
        # Actor 解码器
        self.actor = TransformerDecoder()
        
        # 动作 head
        self.grasp_head = nn.Linear(512, 7)  # (x,y,z, qx,qy,qz,qw)
```

#### 1.2 体素化表示
```python
def voxelize(point_cloud, resolution=0.01):
    # 将点云转换为 3D 体素网格
    voxels = np.zeros((100, 100, 100))
    for point in point_cloud:
        ix, iy, iz = (point / resolution).astype(int)
        voxels[ix, iy, iz] = 1
    return voxels
```

#### 1.3 交叉注意力机制
```
密集输入 (100³ 体素) → 交叉注意力 → 稀疏潜变量 (512) → 自注意力 → 输出

计算量：O(N·M) + O(M²) vs O(N²)
其中 N=1000000 (体素), M=512 (潜变量)
```

### 方法流程

```
┌─────────────────────────────────────────────────────────────┐
│                     PerAct 架构流程                          │
├─────────────────────────────────────────────────────────────┤
│  1. 3D 输入处理：                                            │
│     - RGB-D 图像 → 点云                                     │
│     - 点云 → 体素网格 (100³)                                │
│     - 体素 → 特征 (VoxNet)                                  │
├─────────────────────────────────────────────────────────────┤
│  2. Perceiver 编码：                                         │
│     - 体素特征 (密集) → 交叉注意力 → 潜变量 (稀疏)            │
│     - 潜变量 → 自注意力 → 编码潜变量                         │
├─────────────────────────────────────────────────────────────┤
│  3. Actor 解码：                                             │
│     - 动作查询 → 交叉注意力 → 编码潜变量                     │
│     - 输出：抓取位姿 (位置 + 四元数)                          │
├─────────────────────────────────────────────────────────────┤
│  4. 训练：                                                   │
│     - 监督学习 (演示位姿)                                    │
│     - 损失：位置 L2 + 旋转 L2                               │
└─────────────────────────────────────────────────────────────┘
```

### 数学表达

#### 体素化
$$V(x, y, z) = \begin{cases} 1 & \text{if } \exists p \in \text{point cloud}, ||p - (x,y,z)|| < r \\ 0 & \text{otherwise} \end{cases}$$

#### 交叉注意力
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

其中 $Q$ 来自潜变量，$K, V$ 来自体素特征。

#### 抓取预测
$$\hat{\mathbf{g}} = (\hat{\mathbf{p}}, \hat{\mathbf{q}}) = f_{\text{actor}}(\mathbf{z})$$

### 实验结论

#### 性能对比
| 方法 | 成功率 | 推理时间 |
|------|--------|----------|
| GPD | 75% | 0.5s |
| PointNet++ | 82% | 0.1s |
| PerAct | 92% | 0.08s |

#### 泛化能力
- 新物体：85% 成功率
- 新场景：78% 成功率
- 遮挡：70% 成功率

---

## 2. 公式与流程对照表

| 步骤 | 数学公式 | 代码实现 | 说明 |
|------|----------|----------|------|
| 1. 体素化 | $V = \text{Voxelize}(P)$ | `voxels = voxelize(point_cloud)` | 点云→体素 |
| 2. 体素编码 | $\mathbf{F} = \text{VoxNet}(V)$ | `features = voxnet(voxels)` | 3D CNN |
| 3. 交叉注意力 | $\mathbf{z} = \text{CrossAttn}(\mathbf{L}, \mathbf{F})$ | `latents = cross_attn(latents, features)` | 稀疏化 |
| 4. 自注意力 | $\mathbf{z}' = \text{SelfAttn}(\mathbf{z})$ | `encoded = self_attn(latents)` | 编码 |
| 5. 动作解码 | $\hat{\mathbf{g}} = \text{Decoder}(\mathbf{z}')$ | `grasp = decoder(action_query, encoded)` | 抓取预测 |
| 6. 损失 | $\mathcal{L} = ||\mathbf{g} - \hat{\mathbf{g}}||^2$ | `loss = mse_loss(grasp_pred, grasp_gt)` | 监督学习 |

---

## 3. 迁移部署指南

### 环境依赖
```bash
python >= 3.8
torch >= 1.10
open3d >= 0.15  # 点云处理
```

### 数据准备
```python
# RGB-D 到点云
def rgbd_to_pointcloud(rgb, depth, camera_intrinsics):
    points = []
    colors = []
    for u, v in np.ndindex(depth.shape):
        if depth[u, v] > 0:
            z = depth[u, v]
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
            colors.append(rgb[u, v])
    return np.array(points), np.array(colors)

# 体素化
voxels = voxelize(points, resolution=0.01)
```

### 模型配置
```yaml
model:
  voxel_size: 0.01
  voxel_grid: [100, 100, 100]
  num_latents: 512
  d_model: 512
  n_heads: 8
  
training:
  batch_size: 32
  learning_rate: 1e-4
  loss_weights:
    position: 1.0
    rotation: 1.0
```

### 推理部署
```python
class PerActDeploy:
    def __init__(self, checkpoint):
        self.model = PerAct.load(checkpoint)
    
    def predict_grasp(self, rgb, depth):
        # RGB-D → 点云
        points, colors = rgbd_to_pointcloud(rgb, depth, K)
        
        # 点云 → 体素
        voxels = voxelize(points)
        
        # 预测抓取
        grasp = self.model(voxels)
        
        return grasp  # (x, y, z, qx, qy, qz, qw)
    
    def execute_grasp(self, grasp):
        # 将抓取位姿转换为机器人指令
        pose = grasp_to_robot_pose(grasp)
        robot.move_to(pose)
        robot.close_gripper()
```

### 常见问题

#### Q1: 显存不足
**解决：** 减少体素分辨率，使用更小的潜变量数

#### Q2: 推理慢
**解决：** 体素化优化 (CUDA)，减少 Transformer 层数

---

## 参考资源

- **论文**: https://arxiv.org/abs/2306.17817
- **代码**: https://github.com/peract/peract

---

*最后更新：2026-03-03*
