# Conditional Policy 方法理论

## 核心思想

**Conditional Reinforcement Learning with Explicit Behavior Conditioning**

将驾驶行为（直行/左转/右转）作为**显式条件**输入到策略网络，通过强化学习训练条件化策略。

## 理论基础

### 1. 条件化策略 (Conditional Policy)

**定义**:
```
π(a | s, b)
```
其中:
- `s`: state (车辆状态)
- `b`: behavior (行为标签: 0=straight, 1=left, 2=right)
- `a`: action (控制动作: ax, ay)

**与标准RL的区别**:
- 标准RL: `π(a | s)` - 策略只依赖于状态
- 条件化RL: `π(a | s, b)` - 策略同时依赖于状态和行为意图

### 2. 网络架构

```
Input: State (s) + Behavior (b)
         ↓
    Behavior Embedding: b → e_b ∈ ℝ^32
         ↓
    Concatenate: [s, e_b] ∈ ℝ^(34+32) = ℝ^66
         ↓
    Neural Network: f_θ
         ↓
    Output: μ(s,b), σ(s,b)  (Gaussian policy parameters)
         ↓
    Action: a ~ N(μ(s,b), σ(s,b))
```

**关键设计**:
- **Behavior Embedding**: 将离散的行为ID映射到连续向量空间
- **可学习**: Embedding是网络的一部分，通过RL训练自动学习最优表示
- **条件化**: 不同的behavior ID通过不同的embedding影响最终动作

### 3. 训练方法: PPO (Proximal Policy Optimization)

**目标函数**:
```
L^CLIP(θ) = E_t [ min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t) ]
```

其中:
- `r_t(θ) = π_θ(a_t | s_t, b_t) / π_θ_old(a_t | s_t, b_t)`: 概率比
- `A_t`: advantage function
- `ε`: clip参数 (0.2)

**为什么用PPO?**
1. **稳定性**: 相比REINFORCE等on-policy方法更稳定
2. **样本效率**: 可以重复使用轨迹数据
3. **简单**: 不需要像TRPO那样计算二阶导数

### 4. Reward设计

**基于行为目标的奖励函数**:

```python
if behavior == "straight":
    r = -|a_y| * 0.5 - |a_x| * 0.1
    # 奖励横向加速度小（保持直行）

elif behavior == "left":
    r = a_y * 1.0  (if a_y > 0)
    r = a_y * 2.0  (if a_y < 0, penalty)
    # 奖励正的横向加速度（向左）

elif behavior == "right":
    r = -a_y * 1.0  (if a_y < 0)
    r = -a_y * 2.0  (if a_y > 0, penalty)
    # 奖励负的横向加速度（向右）
```

**设计原则**:
- 不同behavior有不同的奖励函数
- 通过reward引导policy学习behavior-specific动作
- 惩罚与behavior不一致的动作（权重2.0）

## 理论优势

### 相比DIAYN

| 维度 | DIAYN | Conditional Policy |
|------|-------|-------------------|
| **Skill来源** | Unsupervised (需要学习) | Supervised (直接给定) |
| **训练复杂度** | 高（3个网络协同训练） | 低（单一policy网络） |
| **可解释性** | 低（skill是隐变量） | 高（behavior是明确标签） |
| **适用场景** | 无标签探索 | 有标签任务 |

### 相比Behavior Cloning

| 维度 | Behavior Cloning | Conditional Policy (RL) |
|------|-----------------|------------------------|
| **训练方式** | 监督学习（模仿） | 强化学习（优化） |
| **泛化能力** | 受限于数据分布 | 可通过exploration提升 |
| **奖励信号** | 无（只有action label） | 有（可设计复杂reward） |
| **优化目标** | 最小化模仿误差 | 最大化长期回报 |

## 理论假设

1. **Behavior可区分性**: 不同的driving behavior（straight/left/right）在动作空间中有可区分的模式
2. **状态充分性**: 34维状态包含足够信息来执行条件化动作
3. **Reward可设计性**: 可以设计明确的reward函数来区分不同behavior
4. **Policy可学习性**: 神经网络有足够容量学习条件化映射

## 数学形式化

**问题定义**:
```
MDP with Behavior Conditioning: (S, B, A, P, R, γ)
```
- `S`: 状态空间 (ℝ^34)
- `B`: 行为空间 ({0, 1, 2})
- `A`: 动作空间 (ℝ^2)
- `P`: 状态转移概率 P(s' | s, a)
- `R`: 奖励函数 R(s, b, a)
- `γ`: 折扣因子

**目标**:
```
max E_b~U(B) E_τ~π(·|·,b) [ Σ_t γ^t R(s_t, b, a_t) ]
```

即：对于均匀采样的behavior，最大化期望累积奖励。

## 预期效果

如果训练成功，应该观察到：

1. **Behavior条件化**: π(a | s, b=0) ≠ π(a | s, b=1) ≠ π(a | s, b=2)
2. **动作模式**:
   - b=0 (straight) → |a_y| ≈ 0
   - b=1 (left) → a_y > 0
   - b=2 (right) → a_y < 0
3. **状态依赖**: 相同behavior在不同状态下有不同动作（适应性）

## 与CPT集成的接口

未来可以将behavior选择建模为CPT决策：

```
High-level: CPT决策 → 选择behavior (b)
Low-level: Conditional Policy → 执行behavior (a)
```

这样形成层次化RL (Hierarchical RL):
- **上层**: 基于CPT的behavior selector
- **下层**: 基于条件化policy的action executor
