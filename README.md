# Simple Approach: Conditional Policy

## 概述

这是一个**简单、直接、稳定**的方法来训练自动驾驶车辆的行为控制器。

与之前的 DIAYN 方法相比：
- ❌ **DIAYN**: 无监督、复杂、不稳定、训练困难
- ✅ **Conditional Policy**: 监督、简单、稳定、训练快速

## 核心思想

直接训练一个**以行为标签为条件**的策略网络：

```
Input: State + Behavior_Label (straight/left/right)
  ↓
Policy Network
  ↓
Output: Action (ax, ay)
```

## 文件结构

```
simple_approach/
├── models/
│   └── conditional_policy.py    # Policy 和 Value 网络定义
├── training/
│   └── trainer.py               # 训练器（Behavior Cloning）
├── results/                     # 训练结果和模型保存
├── train_simple.py              # 主训练脚本
├── test_simple.py               # 测试脚本
└── README.md                    # 本文件
```

## 快速开始

### 1. 训练模型

```bash
cd /Users/clorisspike/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/manuscript6-RL/codes/

python simple_approach/train_simple.py
```

**训练流程**：
1. 加载高质量合成数据（44,700 samples，平衡分布）
2. Behavior Cloning（监督学习）
   - Epochs: 50
   - Batch size: 128
   - Learning rate: 3e-4
   - Early stopping with patience=10
3. 保存最佳模型

**预期结果**：
- 训练时间: ~5-10 分钟（CPU）
- 最终验证损失: < 0.5
- 动作准确率 (threshold=0.1): > 80%

### 2. 测试模型

```bash
python simple_approach/test_simple.py
```

**测试内容**：
1. Behavior Conditioning Test - 验证不同 behavior 产生不同 action
2. Temporal Consistency Test - 验证动作的平滑性
3. Action Distribution Visualization - 可视化不同 behavior 的动作分布
4. Specific Scenarios Test - 测试特定场景

### 3. 查看结果

```bash
ls simple_approach/results/

# 输出：
# - best_policy_bc.pth              # 最佳模型（BC）
# - final_policy.pth                # 最终模型
# - training_curves.png             # 训练曲线
# - behavior_action_distribution.png # 动作分布可视化
```

## 模型架构

### Conditional Policy

```python
ConditionalPolicy(
    state_dim=34,
    action_dim=2,
    num_behaviors=3,
    hidden_dim=128,
    embedding_dim=32
)

# 架构:
# 1. Behavior Embedding: [3] → [32]
# 2. Concatenate: [State(34) + Embedding(32)] → [66]
# 3. MLP: [66] → [128] → [128] → [64]
# 4. Heads:
#    - Mean: [64] → [2]
#    - Log_std: [64] → [2]
# 5. Output: Normal(mean, std)
```

**参数数量**: ~50K (非常轻量)

### 训练方法

**Behavior Cloning (BC)**:
- 监督学习，最大化似然
- Loss: Negative Log-Likelihood
  ```
  L = -Σ log p(action | state, behavior)
  ```
- 优点: 简单、快速、稳定
- 缺点: 受限于训练数据分布

## 性能指标

### 目标指标

| Metric | Target | Current |
|--------|--------|---------|
| Train Loss | < 0.5 | TBD |
| Val Loss | < 0.5 | TBD |
| Accuracy (threshold=0.1) | > 80% | TBD |
| Left turn success | > 90% | TBD |
| Straight success | > 95% | TBD |
| Right turn success | > 90% | TBD |

### 与 DIAYN 对比

| Aspect | DIAYN | Conditional Policy |
|--------|-------|-------------------|
| 训练时间 | ~3-4 hours | ~5-10 minutes |
| 模型复杂度 | 3 networks (300K params) | 1 network (50K params) |
| 训练稳定性 | 不稳定，易发散 | 稳定 |
| 最终成功率 | Straight: 40%, Left: 0%, Right: 20% | TBD (预计 >90%) |
| 可解释性 | 差（无监督学习） | 好（直接映射） |

## 后续计划：CPT 集成

### Phase 1: 基础能力验证（当前）
- ✅ 实现 Conditional Policy
- ✅ 训练并验证基础能力
- ⏳ 确保 left/straight/right 准确率 >90%

### Phase 2: CPT 模块设计
1. **CPT 参数标定**
   - 收集真实驾驶数据
   - 拟合 α, β, λ, γ 参数
   - 分析不同司机的风险偏好

2. **High-level Policy 设计**
   ```python
   class CPTHighLevelPolicy:
       def forward(self, state, cpt_params):
           # 1. 评估不同 behavior 的 CPT 价值
           values = []
           for behavior in ["follow", "overtake", "lane_change"]:
               value = compute_cpt_value(state, behavior, cpt_params)
               values.append(value)

           # 2. 选择最高价值的 behavior
           behavior = argmax(values)
           return behavior
   ```

3. **端到端训练**
   - 冻结 Low-level policy（已训练好的 Conditional Policy）
   - 训练 High-level policy（CPT-driven）
   - 在仿真环境中优化

### Phase 3: 个性化与部署
- 个性化 CPT 参数（不同司机有不同的风险态度）
- 在仿真环境中测试
- 对比不同 CPT 参数的表现

## 故障排查

### 问题1: 训练损失不下降

**可能原因**:
- 学习率过大/过小
- 数据未归一化
- 模型容量不足

**解决方案**:
- 调整学习率: 尝试 1e-3, 1e-4, 1e-5
- 检查数据范围: `states.mean()`, `states.std()`
- 增加 `hidden_dim`: 128 → 256

### 问题2: 不同 behavior 输出相似

**可能原因**:
- Embedding 维度太小
- 训练 epoch 不够
- 数据标签有误

**解决方案**:
- 增加 `embedding_dim`: 32 → 64
- 训练更多 epochs: 50 → 100
- 检查数据: `print(behavior_ids[:100])`

### 问题3: 动作过于剧烈（不平滑）

**可能原因**:
- 标准差过大
- 缺少动作平滑约束

**解决方案**:
- 限制 `log_std` 范围: `torch.clamp(log_std, -20, 1)`
- 添加动作平滑损失:
  ```python
  action_diff = actions[1:] - actions[:-1]
  smooth_loss = (action_diff ** 2).mean()
  ```

## 参考

- **Behavior Cloning**: Pomerleau, D. (1989). ALVINN: An autonomous land vehicle in a neural network.
- **Conditional VAE**: Sohn, K., Lee, H., & Yan, X. (2015). Learning structured output representation using deep conditional generative models.
- **CPT in Driving**: Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty.

## 联系

项目路径: `/Users/clorisspike/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/manuscript6-RL/codes/`

查看完整项目日志: `PROJECT_LOG.md`
