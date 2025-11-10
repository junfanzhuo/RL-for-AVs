"""
Conditional Policy Trainer
使用 Behavior Cloning + PPO 训练条件化策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class ConditionalPolicyTrainer:
    """
    两阶段训练：
    Stage 1: Behavior Cloning（监督学习）- 快速收敛到合理策略
    Stage 2: PPO Fine-tuning（可选）- 在环境中优化
    """

    def __init__(self, policy, value_net=None, device='cpu'):
        self.policy = policy
        self.value_net = value_net
        self.device = device

        # Move to device
        self.policy.to(device)
        if self.value_net is not None:
            self.value_net.to(device)

    def load_normalization_params(self, param_path='simple_approach/results/normalization_params.pth'):
        """加载归一化参数"""
        params = torch.load(param_path, map_location=self.device)
        self.state_mean = params['state_mean']
        self.state_std = params['state_std']
        self.action_mean = params['action_mean']
        self.action_std = params['action_std']
        print(f"✓ Loaded normalization params from {param_path}")

    def behavior_cloning(self, states, actions, behavior_ids,
                        epochs=50, batch_size=64, lr=3e-4,
                        weight_decay=1e-5, validation_split=0.1):
        """
        Stage 1: Behavior Cloning（行为克隆）

        Args:
            states: [N, state_dim] numpy array or tensor
            actions: [N, action_dim] numpy array or tensor
            behavior_ids: [N] numpy array or tensor (0/1/2)
            epochs: 训练轮数
            batch_size: 批大小
            lr: 学习率
            weight_decay: 权重衰减
            validation_split: 验证集比例

        Returns:
            train_losses, val_losses: 训练和验证损失历史
        """
        print("="*70)
        print("STAGE 1: BEHAVIOR CLONING")
        print("="*70)

        # 转换为 tensor
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states)
        if not isinstance(actions, torch.Tensor):
            actions = torch.FloatTensor(actions)
        if not isinstance(behavior_ids, torch.Tensor):
            behavior_ids = torch.LongTensor(behavior_ids)

        # 数据统计
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(states):,}")
        print(f"  State dim: {states.shape[1]}")
        print(f"  Action dim: {actions.shape[1]}")

        # 行为分布
        unique, counts = torch.unique(behavior_ids, return_counts=True)
        behavior_names = ["Straight", "Left", "Right"]
        print(f"\nBehavior Distribution:")
        for bid, count in zip(unique, counts):
            print(f"  {behavior_names[bid]:10s}: {count:6,} ({100*count/len(behavior_ids):.1f}%)")

        # 归一化数据
        print(f"\nNormalizing data...")
        state_mean = states.mean(dim=0)
        state_std = states.std(dim=0) + 1e-8
        states = (states - state_mean) / state_std

        action_mean = actions.mean(dim=0)
        action_std = actions.std(dim=0) + 1e-8
        actions = (actions - action_mean) / action_std

        print(f"  State: mean={state_mean[:3].numpy()}, std={state_std[:3].numpy()}")
        print(f"  Action: mean={action_mean.numpy()}, std={action_std.numpy()}")

        # 保存归一化参数
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std

        # 划分训练/验证集
        n_samples = len(states)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_dataset = TensorDataset(
            states[train_indices],
            actions[train_indices],
            behavior_ids[train_indices]
        )
        val_dataset = TensorDataset(
            states[val_indices],
            actions[val_indices],
            behavior_ids[val_indices]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"\nDataset Split:")
        print(f"  Training:   {n_train:6,} samples")
        print(f"  Validation: {n_val:6,} samples")

        # 优化器
        optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=weight_decay)

        # 学习率调度
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # 训练循环
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        print(f"\nStarting Behavior Cloning Training...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print("="*70)

        for epoch in range(epochs):
            # 训练阶段
            self.policy.train()
            train_loss = 0.0

            for batch_states, batch_actions, batch_behaviors in train_loader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_behaviors = batch_behaviors.to(self.device)

                # 前向传播
                action_dist = self.policy(batch_states, batch_behaviors)

                # 负对数似然损失（等价于最大化似然）
                loss = -action_dist.log_prob(batch_actions).sum(dim=-1).mean()

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # 验证阶段
            self.policy.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_states, batch_actions, batch_behaviors in val_loader:
                    batch_states = batch_states.to(self.device)
                    batch_actions = batch_actions.to(self.device)
                    batch_behaviors = batch_behaviors.to(self.device)

                    action_dist = self.policy(batch_states, batch_behaviors)
                    loss = -action_dist.log_prob(batch_actions).sum(dim=-1).mean()
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # 学习率调度
            scheduler.step(val_loss)

            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss = {train_loss:.4f}, "
                      f"Val Loss = {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.policy.state_dict(),
                          'simple_approach/results/best_policy_bc.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        print("="*70)
        print(f"✓ BEHAVIOR CLONING COMPLETE")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Model saved: simple_approach/results/best_policy_bc.pth")

        # 保存归一化参数
        normalization_params = {
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std
        }
        torch.save(normalization_params, 'simple_approach/results/normalization_params.pth')
        print(f"  Normalization params saved: simple_approach/results/normalization_params.pth")
        print("="*70)

        # 加载最佳模型
        self.policy.load_state_dict(
            torch.load('simple_approach/results/best_policy_bc.pth')
        )

        return train_losses, val_losses

    def evaluate_accuracy(self, states, actions, behavior_ids, threshold=0.1):
        """
        评估模型在测试集上的准确率

        Args:
            states: [N, state_dim]
            actions: [N, action_dim] ground truth actions
            behavior_ids: [N]
            threshold: 动作差异阈值（用于判断"正确"）

        Returns:
            accuracy, mean_error
        """
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states)
        if not isinstance(actions, torch.Tensor):
            actions = torch.FloatTensor(actions)
        if not isinstance(behavior_ids, torch.Tensor):
            behavior_ids = torch.LongTensor(behavior_ids)

        self.policy.eval()

        with torch.no_grad():
            states = states.to(self.device)
            actions = actions.to(self.device)
            behavior_ids = behavior_ids.to(self.device)

            # 归一化states（使用保存的参数）
            if hasattr(self, 'state_mean'):
                states = (states - self.state_mean.to(self.device)) / self.state_std.to(self.device)

            # 归一化actions（用于比较）
            actions_normalized = actions
            if hasattr(self, 'action_mean'):
                actions_normalized = (actions - self.action_mean.to(self.device)) / self.action_std.to(self.device)

            # 预测动作（确定性，返回归一化的动作）
            pred_actions_normalized = self.policy.get_action(states, behavior_ids, deterministic=True)

            # 反归一化预测动作以计算真实误差
            pred_actions = pred_actions_normalized
            if hasattr(self, 'action_mean'):
                pred_actions = pred_actions_normalized * self.action_std.to(self.device) + self.action_mean.to(self.device)

            # 计算误差（在原始尺度上）
            errors = (pred_actions - actions).abs()
            mean_error = errors.mean(dim=0)

            # 计算准确率（所有维度误差都小于阈值才算正确）
            correct = (errors < threshold).all(dim=1)
            accuracy = correct.float().mean().item()

        return accuracy, mean_error.cpu().numpy()


# 辅助函数：从数据集加载数据
def load_data_from_dataset(dataset_obj):
    """
    从 Dataset 对象加载数据并转换格式

    Args:
        dataset_obj: Dataset 对象

    Returns:
        states, actions, behavior_ids (all as numpy arrays)
    """
    data = dataset_obj.generate_dataset(use_sample_weight=False)

    states = []
    actions = []
    behavior_ids = []

    # Behavior label to ID mapping
    behavior_to_id = {
        "straight": 0,
        "left": 1,
        "right": 2
    }

    # Dataset format: (state, action, next_state, done, mode)
    for state, action, next_state, done, behavior_label in data:
        states.append(state)
        actions.append(action)  # action is [ax, ay]
        behavior_ids.append(behavior_to_id[behavior_label])

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    behavior_ids = np.array(behavior_ids, dtype=np.int64)

    return states, actions, behavior_ids
