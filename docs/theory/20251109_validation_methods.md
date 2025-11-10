# Conditional Policy 验证方法

## 验证策略

我使用了**多层次验证**来评估模型是否学到了条件化行为。

## 1. 训练过程监控

### 监控指标

```
Iter   1/100: Avg Reward =    0.00, Policy Loss = 0.0000, Value Loss = 0.7714
Iter  50/100: Avg Reward =    0.22, Policy Loss = -0.0000, Value Loss = 0.2197
Iter 100/100: Avg Reward =   -0.51, Policy Loss = 0.0000, Value Loss = 0.1432
```

**验证点**:
- ✅ **Value Loss下降**: 从0.77 → 0.14 (说明value网络在学习)
- ⚠️ **Policy Loss ≈ 0**: PPO更新很小（可能是clip太保守）
- ⚠️ **Avg Reward波动**: 没有明显上升趋势

**结论**: 训练在进行，但policy更新不够强。

---

## 2. 行为条件化测试

### 测试方法

对于**相同的状态**，分别输入3种不同的behavior ID，观察输出动作的差异。

```python
test_state = torch.randn(34)  # 固定状态

for behavior_id in [0, 1, 2]:  # Straight, Left, Right
    action = policy.get_action(test_state, behavior_id, deterministic=True)
    print(f"{behavior_name}: ax={action[0]:.3f}, ay={action[1]:.3f}")
```

### 预期结果

如果模型学会了条件化行为：
- **Straight (b=0)**: ay ≈ 0 (横向加速度小)
- **Left (b=1)**: ay > 0 (正的横向加速度，向左)
- **Right (b=2)**: ay < 0 (负的横向加速度，向右)

### 实际结果

```
测试状态 1:
  Straight: ax= +0.173, ay= -1.071
  Left    : ax= +0.125, ay= -1.072
  Right   : ax= +0.176, ay= -1.581
  差异: S-L: ax=0.048 ay=0.001, S-R: ax=0.002 ay=0.510
```

**观察**:
- ❌ Left 和 Straight 的 ay 几乎相同 (差异0.001)
- ⚠️ Right 的 ay 确实更负 (差异0.510)
- ❌ 所有behavior的ay都是负值（没有看到Left的正ay）

**结论**: 模型有轻微的条件化趋势，但**不够强**。

---

## 3. 统计分析验证

### 方法

对10个随机状态重复测试，计算统计量：

```python
n_test_states = 10
differences = []

for test_idx in range(n_test_states):
    test_state = torch.randn(34)

    action_straight = policy.get_action(test_state, 0, deterministic=True)
    action_left = policy.get_action(test_state, 1, deterministic=True)
    action_right = policy.get_action(test_state, 2, deterministic=True)

    # 计算差异
    diff_SL = |action_straight - action_left|
    diff_SR = |action_straight - action_right|
    diff_LR = |action_left - action_right|

    differences.append([diff_SL, diff_SR, diff_LR])
```

### 结果

| 差异类型 | 平均 ax 差异 | 平均 ay 差异 |
|---------|-------------|-------------|
| S-L | 0.091 | 0.262 |
| S-R | 0.064 | 0.570 |
| L-R | 0.094 | 0.461 |

**解读**:
- ✅ S-R 的 ay 差异最大 (0.570) - 说明Straight和Right有区分
- ⚠️ S-L 的 ay 差异较小 (0.262) - Straight和Left区分度低
- ❌ 平均差异 < 1.0 - 相比动作的std (7.55, 4.46)，差异很小

**结论**: 模型学到了**微弱的条件化**，但远未达到理想状态。

---

## 4. 轨迹级验证 (未完成)

### 理想方法

应该在环境中运行完整轨迹，观察：

```python
env.reset(behavior_id=0)  # Straight
trajectory_straight = collect_trajectory(policy, env, max_steps=50)

env.reset(behavior_id=1)  # Left
trajectory_left = collect_trajectory(policy, env, max_steps=50)

# 比较轨迹
analyze_trajectory_difference(trajectory_straight, trajectory_left)
```

**应该观察到**:
- Straight轨迹: 横向位移 ≈ 0
- Left轨迹: 横向位移 > 0（逐渐向左）
- Right轨迹: 横向位移 < 0（逐渐向右）

**现状**: 未实现此验证（因为初步测试已经发现问题）

---

## 5. 与数据集对比验证 (未完成)

### 理想方法

将模型预测与真实数据对比：

```python
# 从平衡数据集中提取
straight_data = [sample for sample in data if sample[4] == "straight"]
left_data = [sample for sample in data if sample[4] == "left"]
right_data = [sample for sample in data if sample[4] == "right"]

# 计算真实数据的动作分布
real_ay_straight = [sample[1][1] for sample in straight_data]
real_ay_left = [sample[1][1] for sample in left_data]
real_ay_right = [sample[1][1] for sample in right_data]

# 计算模型预测的动作分布
pred_ay_straight = []
for sample in straight_data[:1000]:
    state = sample[0]
    action = policy.get_action(state, behavior_id=0)
    pred_ay_straight.append(action[1])

# 对比分布
compare_distributions(real_ay_straight, pred_ay_straight)
```

**预期**: 模型预测的ay分布应该接近真实数据的ay分布

**现状**: 未实现（因为需要先解决条件化问题）

---

## 验证总结

### 当前验证覆盖

| 验证类型 | 是否完成 | 结果 |
|---------|---------|------|
| 训练损失监控 | ✅ | Value loss下降，但policy loss≈0 |
| 单点行为测试 | ✅ | 有微弱条件化，但不够强 |
| 统计分析 | ✅ | 差异 < 1.0，相比数据std很小 |
| 轨迹级验证 | ❌ | 未实现 |
| 数据分布对比 | ❌ | 未实现 |

### 问题诊断

基于验证结果，识别出的问题：

1. **Policy更新太弱** (Policy Loss ≈ 0)
   - PPO的clip可能太保守
   - 学习率可能太小

2. **Reward信号不够强**
   - 当前reward范围 [-3, +0.2]
   - behavior间的reward差异可能不明显

3. **训练不充分**
   - 100 iterations可能太少
   - 每iteration只收集10个episodes (约500步)
   - 总训练步数 ≈ 50K，相比450K数据太少

### 改进建议

1. **增强验证**:
   - 实现轨迹级验证
   - 可视化不同behavior的action分布
   - 在测试环境中定量评估成功率

2. **改进训练**:
   - 调大reward权重（2x-5x）
   - 增加训练iterations (500-1000)
   - 调整PPO超参数（clip, lr）

3. **混合方法**:
   - 先用BC预训练（建立基础）
   - 再用PPO fine-tune（优化）
