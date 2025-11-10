# Simple Approach - 进度总结

## 当前状态 (2025-01-09)

### ✅ 已完成

1. **生成平衡的合成数据**
   - 文件: `cached_generated_balanced.pkl`
   - 样本数: 450,000
   - 分布: Straight 33.3%, Left 33.3%, Right 33.3%
   - 动作范围: ax std=7.55, ay std=4.46

2. **创建 Conditional Policy 模型**
   - 文件: `simple_approach/models/conditional_policy.py`
   - 输入: State (34维) + Behavior ID (0/1/2)
   - 输出: Action distribution (ax, ay)
   - 参数量: 34,212

3. **创建简化环境**
   - 文件: `simple_approach/simple_environment.py`
   - 不依赖 DIAYN 组件（无需 encoder/discriminator）
   - Reward基于behavior目标

4. **实现 PPO 训练**
   - 文件: `simple_approach/train_rl.py`
   - 使用强化学习而非监督学习
   - 训练了100个iterations
   - 模型保存: `simple_approach/results/policy_iter_100.pth`

### ⚠️ 当前问题

**模型没有学会有效的条件化行为**：
- Left/Right/Straight 的动作差异很小
- ay (横向加速度) 没有表现出预期的模式:
  - Left 应该 ay > 0（向左）
  - Right 应该 ay < 0（向右）
  - Straight 应该 ay ≈ 0
- Policy loss ≈ 0，说明PPO更新很小

### 🔍 可能的原因

1. **Reward信号太弱**
   - 当前reward基于单步的 ay 值
   - 可能需要更强的奖励差异

2. **训练不充分**
   - 100 iterations 可能不够
   - 每个iteration只收集10个episodes

3. **PPO超参数**
   - clip_epsilon = 0.2 可能太保守
   - 学习率可能需要调整

4. **环境设计**
   - 简化环境的状态转移可能过于简单
   - Reward函数可能需要重新设计

##下一步建议

### 选项1: 调整训练参数
- 增加训练iterations (500-1000)
- 增加每轮收集的episodes
- 调整reward权重和PPO超参数

### 选项2: 改进Reward设计
- 使用轨迹级别的reward（而不是单步）
- 添加更明确的行为区分奖励
- 考虑sparse reward（只在成功执行行为时给高奖励）

### 选项3: 混合方法
- Stage 1: Behavior Cloning 预训练（使用平衡数据）
- Stage 2: PPO Fine-tuning（强化学习优化）

## 文件结构

```
simple_approach/
├── models/
│   └── conditional_policy.py     # Policy 和 Value network
├── training/
│   └── trainer.py                 # BC 训练器（已包含归一化）
├── results/
│   ├── policy_iter_10.pth         # RL checkpoints
│   ├── policy_iter_100.pth        # 最终RL模型
│   └── normalization_params.pth   # 归一化参数（BC训练时生成）
├── training_logs/
│   ├── train_rl_20250109.log      # RL训练日志
│   └── training_new.log           # BC训练日志
├── simple_environment.py          # 简化的驾驶环境
├── train_rl.py                    # PPO训练脚本
└── train_simple.py                # BC训练脚本

根目录:
├── cached_generated_balanced.pkl  # 平衡的合成数据(450K样本)
└── generate_balanced_data.py      # 数据生成脚本
```

## 关键技术决策

1. **放弃DIAYN** - 太复杂，unsupervised不适合有标签的任务
2. **Conditional Policy** - 直接以behavior_id为条件
3. **使用合成数据** - 平衡且清晰的行为标签
4. **RL训练** - 通过环境交互学习，而非纯监督
