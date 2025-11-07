"""
Conditional Policy Network
简单、直接的条件化策略网络，以 behavior 为条件输出动作分布

优点：
1. 直接利用标签数据（left/straight/right）
2. 训练稳定，收敛快
3. 可解释性强
4. 易于扩展到 CPT 集成
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class ConditionalPolicy(nn.Module):
    """
    条件化 Policy 网络

    输入：
        - state: [batch, state_dim] 车辆状态
        - behavior_id: [batch] 行为标签 (0=straight, 1=left, 2=right)

    输出：
        - action_dist: Normal distribution over actions
    """

    def __init__(self, state_dim, action_dim, num_behaviors=3, hidden_dim=128,
                 embedding_dim=32, use_learned_embedding=True):
        """
        Args:
            state_dim: 状态维度 (34)
            action_dim: 动作维度 (2: ax, ay)
            num_behaviors: 行为数量 (3: straight, left, right)
            hidden_dim: 隐藏层维度
            embedding_dim: behavior embedding 维度
            use_learned_embedding: 是否使用可学习的 embedding（否则用 one-hot）
        """
        super(ConditionalPolicy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_behaviors = num_behaviors
        self.use_learned_embedding = use_learned_embedding

        # Behavior embedding
        if use_learned_embedding:
            self.behavior_embedding = nn.Embedding(num_behaviors, embedding_dim)
        else:
            # 使用 one-hot，不需要学习
            embedding_dim = num_behaviors
            self.behavior_embedding = None

        # Policy network
        self.network = nn.Sequential(
            nn.Linear(state_dim + embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Action mean head
        self.mean_head = nn.Linear(hidden_dim // 2, action_dim)

        # Action log_std head
        self.log_std_head = nn.Linear(hidden_dim // 2, action_dim)

    def get_behavior_features(self, behavior_id):
        """
        将 behavior_id 转换为 feature vector

        Args:
            behavior_id: [batch] 或 [batch, 1] 的整数 tensor，或 int

        Returns:
            behavior_features: [batch, embedding_dim]
        """
        # Handle int/numpy inputs
        if isinstance(behavior_id, int):
            behavior_id = torch.tensor([behavior_id])
        elif not isinstance(behavior_id, torch.Tensor):
            behavior_id = torch.tensor(behavior_id)

        if behavior_id.dim() == 2:
            behavior_id = behavior_id.squeeze(-1)

        if self.use_learned_embedding:
            # 使用可学习的 embedding
            return self.behavior_embedding(behavior_id)
        else:
            # 使用 one-hot encoding
            batch_size = behavior_id.shape[0]
            behavior_features = torch.zeros(batch_size, self.num_behaviors,
                                           device=behavior_id.device)
            behavior_features.scatter_(1, behavior_id.unsqueeze(-1), 1.0)
            return behavior_features

    def forward(self, state, behavior_id):
        """
        前向传播

        Args:
            state: [batch, state_dim]
            behavior_id: [batch] 或 [batch, 1]

        Returns:
            action_dist: Normal distribution
        """
        # 获取 behavior embedding
        behavior_features = self.get_behavior_features(behavior_id)

        # 拼接 state 和 behavior features
        x = torch.cat([state, behavior_features], dim=-1)

        # 通过网络
        x = self.network(x)

        # 计算 mean 和 std
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)

        # 限制 log_std 范围，防止数值不稳定
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        # 返回正态分布
        return Normal(mean, std)

    def get_action(self, state, behavior_id, deterministic=False):
        """
        采样动作

        Args:
            state: [batch, state_dim] 或 [state_dim]
            behavior_id: [batch] 或 标量
            deterministic: 是否使用确定性策略（返回 mean）

        Returns:
            action: [batch, action_dim] 或 [action_dim]
        """
        # 处理单个样本的情况
        single_sample = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            if isinstance(behavior_id, int):
                behavior_id = torch.tensor([behavior_id], device=state.device)
            else:
                behavior_id = behavior_id.unsqueeze(0)
            single_sample = True

        # 获取动作分布
        action_dist = self.forward(state, behavior_id)

        # 采样或取均值
        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.sample()

        # 返回单个样本
        if single_sample:
            action = action.squeeze(0)

        return action


class ValueNetwork(nn.Module):
    """
    Value network for policy gradient training
    估计 V(s, behavior)
    """

    def __init__(self, state_dim, num_behaviors=3, hidden_dim=128, embedding_dim=32):
        super(ValueNetwork, self).__init__()

        # Behavior embedding
        self.behavior_embedding = nn.Embedding(num_behaviors, embedding_dim)

        # Value network
        self.network = nn.Sequential(
            nn.Linear(state_dim + embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state, behavior_id):
        """
        Args:
            state: [batch, state_dim]
            behavior_id: [batch]

        Returns:
            value: [batch, 1]
        """
        if behavior_id.dim() == 2:
            behavior_id = behavior_id.squeeze(-1)

        behavior_features = self.behavior_embedding(behavior_id)
        x = torch.cat([state, behavior_features], dim=-1)
        value = self.network(x)
        return value


# 测试代码
if __name__ == "__main__":
    print("Testing ConditionalPolicy...")

    # 参数
    state_dim = 34
    action_dim = 2
    batch_size = 16

    # 创建模型
    policy = ConditionalPolicy(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)

    # 创建测试数据
    state = torch.randn(batch_size, state_dim)
    behavior_id = torch.randint(0, 3, (batch_size,))

    print(f"State shape: {state.shape}")
    print(f"Behavior IDs: {behavior_id}")

    # 测试前向传播
    action_dist = policy(state, behavior_id)
    print(f"\n✓ Forward pass successful")
    print(f"  Action mean shape: {action_dist.mean.shape}")
    print(f"  Action std shape: {action_dist.stddev.shape}")

    # 测试动作采样
    action = policy.get_action(state, behavior_id)
    print(f"\n✓ Action sampling successful")
    print(f"  Action shape: {action.shape}")
    print(f"  Action range: [{action.min().item():.2f}, {action.max().item():.2f}]")

    # 测试单个样本
    single_state = torch.randn(state_dim)
    single_behavior = 1  # left
    single_action = policy.get_action(single_state, single_behavior, deterministic=True)
    print(f"\n✓ Single sample test successful")
    print(f"  Single action shape: {single_action.shape}")
    print(f"  Single action: {single_action}")

    # 测试 Value network
    value = value_net(state, behavior_id)
    print(f"\n✓ Value network test successful")
    print(f"  Value shape: {value.shape}")
    print(f"  Value range: [{value.min().item():.2f}, {value.max().item():.2f}]")

    # 测试不同 behavior 的输出差异
    print(f"\n✓ Testing behavior conditioning...")
    test_state = torch.randn(state_dim)
    for behavior_name, behavior_id in [("Straight", 0), ("Left", 1), ("Right", 2)]:
        action = policy.get_action(test_state, behavior_id, deterministic=True)
        print(f"  {behavior_name:8s}: ax={action[0].item():+.3f}, ay={action[1].item():+.3f}")

    print("\n✓ All tests passed!")
