"""
Conditional Policy with RL (PPO) Training
使用强化学习（PPO）训练条件化策略

关键区别于监督学习:
1. 不是直接拟合 (state, behavior) -> action
2. 而是通过与环境交互，最大化 reward
3. Policy 根据 behavior_id 执行不同的动作策略
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_approach.models.conditional_policy import ConditionalPolicy, ValueNetwork
from simple_approach.simple_environment import SimpleEnvironment


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PPOTrainer:
    """
    PPO Trainer for Conditional Policy
    """
    def __init__(self, policy, value_net, env, device='cpu'):
        self.policy = policy
        self.value_net = value_net
        self.env = env
        self.device = device

        self.policy.to(device)
        self.value_net.to(device)

    def collect_trajectories(self, n_episodes=10, behavior_distribution=None):
        """
        收集轨迹数据

        Args:
            n_episodes: 收集的episode数量
            behavior_distribution: 行为分布 {0: prob_straight, 1: prob_left, 2: prob_right}
                                  如果为None，则均匀采样

        Returns:
            trajectories: list of (state, action, reward, next_state, done, behavior_id)
        """
        if behavior_distribution is None:
            behavior_distribution = {0: 1/3, 1: 1/3, 2: 1/3}

        trajectories = []

        for episode in range(n_episodes):
            # 采样一个 behavior_id
            behavior_id = np.random.choice(
                list(behavior_distribution.keys()),
                p=list(behavior_distribution.values())
            )

            state = self.env.reset(behavior_id=behavior_id)
            done = False
            episode_reward = 0
            max_steps_per_episode = 50

            while not done:
                # 将state转换为tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # Policy选择动作（根据behavior_id）
                with torch.no_grad():
                    action = self.policy.get_action(
                        state_tensor, behavior_id, deterministic=False
                    ).cpu().numpy()

                # Ensure action is 1D
                if action.ndim > 1:
                    action = action.squeeze()

                # 与环境交互
                next_state, reward, done, _ = self.env.step(action)

                # 保存轨迹
                trajectories.append((
                    state, action, reward, next_state, done, behavior_id
                ))

                state = next_state
                episode_reward += reward

                # 防止单个episode过长
                if len(trajectories) - len([t for t in trajectories if t[4] != behavior_id]) > max_steps_per_episode:
                    break

        return trajectories

    def compute_advantages(self, trajectories, gamma=0.99, lam=0.95):
        """
        计算 GAE (Generalized Advantage Estimation)
        """
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        behavior_ids = []

        for state, action, reward, next_state, done, behavior_id in trajectories:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            behavior_ids.append(behavior_id)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        behavior_ids = torch.LongTensor(behavior_ids).to(self.device)

        # 计算 values
        with torch.no_grad():
            values = self.value_net(states, behavior_ids).squeeze()
            next_values = self.value_net(next_states, behavior_ids).squeeze()

        # 计算 TD error 和 advantages
        deltas = rewards + gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards)

        advantage = 0
        for t in reversed(range(len(rewards))):
            advantage = deltas[t] + gamma * lam * advantage * (1 - dones[t])
            advantages[t] = advantage

        # 归一化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = advantages + values

        return states, actions, behavior_ids, advantages, returns, values

    def ppo_update(self, states, actions, behavior_ids, advantages, returns, old_values,
                   policy_lr=3e-4, value_lr=1e-3, clip_epsilon=0.2, n_epochs=10, batch_size=64):
        """
        PPO 更新
        """
        policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)

        n_samples = len(states)

        for epoch in range(n_epochs):
            # 打乱数据
            indices = torch.randperm(n_samples)

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_behavior_ids = behavior_ids[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Policy loss (PPO clip)
                action_dist = self.policy(batch_states, batch_behavior_ids)
                log_probs = action_dist.log_prob(batch_actions).sum(dim=-1)

                # 旧的log_probs（需要在训练前计算）
                with torch.no_grad():
                    old_action_dist = self.policy(batch_states, batch_behavior_ids)
                    old_log_probs = old_action_dist.log_prob(batch_actions).sum(dim=-1)

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values = self.value_net(batch_states, batch_behavior_ids).squeeze()
                value_loss = nn.MSELoss()(values, batch_returns)

                # 更新
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

        return policy_loss.item(), value_loss.item()

    def train(self, n_iterations=100, episodes_per_iter=10, save_dir='simple_approach/results'):
        """
        主训练循环
        """
        print("="*70)
        print("PPO TRAINING - Conditional Policy")
        print("="*70)

        os.makedirs(save_dir, exist_ok=True)

        for iteration in range(n_iterations):
            # 收集轨迹
            trajectories = self.collect_trajectories(n_episodes=episodes_per_iter)

            # 计算advantages
            states, actions, behavior_ids, advantages, returns, old_values = \
                self.compute_advantages(trajectories)

            # PPO更新
            policy_loss, value_loss = self.ppo_update(
                states, actions, behavior_ids, advantages, returns, old_values
            )

            # 打印进度
            avg_reward = np.mean([t[2] for t in trajectories])
            print(f"Iter {iteration+1:3d}/{n_iterations}: "
                  f"Avg Reward = {avg_reward:7.2f}, "
                  f"Policy Loss = {policy_loss:.4f}, "
                  f"Value Loss = {value_loss:.4f}")

            # 定期保存
            if (iteration + 1) % 10 == 0:
                torch.save(self.policy.state_dict(),
                          f'{save_dir}/policy_iter_{iteration+1}.pth')
                torch.save(self.value_net.state_dict(),
                          f'{save_dir}/value_iter_{iteration+1}.pth')

        print(f"\n✓ Training complete!")
        print(f"Models saved to {save_dir}/")


def main():
    print("="*80)
    print(" CONDITIONAL POLICY TRAINING WITH RL (PPO)")
    print("="*80)

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 加载平衡数据
    print("\n[1/3] Loading balanced synthetic data...")
    with open('cached_generated_balanced.pkl', 'rb') as f:
        data = pickle.load(f)

    print(f"  Total samples: {len(data):,}")

    # 创建环境
    print("\n[2/3] Creating environment...")
    env = SimpleEnvironment(data=data, max_steps=50)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    # 创建模型
    print("\n[3/3] Creating models...")
    policy = ConditionalPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        num_behaviors=3,
        hidden_dim=128,
        embedding_dim=32
    )

    value_net = ValueNetwork(
        state_dim=state_dim,
        num_behaviors=3,
        hidden_dim=128,
        embedding_dim=32
    )

    print(f"  Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Value params: {sum(p.numel() for p in value_net.parameters()):,}")

    # 训练
    trainer = PPOTrainer(policy, value_net, env, device=device)
    trainer.train(n_iterations=100, episodes_per_iter=10)


if __name__ == "__main__":
    main()
