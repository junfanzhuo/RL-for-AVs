"""
测试训练好的 Conditional Policy
包括：
1. 在测试集上评估准确率
2. 可视化不同 behavior 的动作输出
3. 在简单场景中测试策略行为
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import Dataset
from simple_approach.models.conditional_policy import ConditionalPolicy
from simple_approach.training.trainer import load_data_from_dataset


def load_trained_policy(model_path='simple_approach/results/final_policy.pth'):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path)
    config = checkpoint['config']

    policy = ConditionalPolicy(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        num_behaviors=config['num_behaviors'],
        hidden_dim=config['hidden_dim'],
        embedding_dim=config['embedding_dim'],
        use_learned_embedding=True
    )

    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()

    print(f"✓ Model loaded from {model_path}")
    return policy, config


def test_behavior_conditioning(policy, test_states, n_samples=10):
    """
    测试模型是否真正响应 behavior 条件

    对同一个 state，测试不同 behavior 的输出
    """
    print("\n" + "="*70)
    print("TEST 1: Behavior Conditioning")
    print("="*70)

    behavior_names = ["Straight", "Left", "Right"]

    with torch.no_grad():
        for i in range(n_samples):
            state = torch.FloatTensor(test_states[i]).unsqueeze(0)

            print(f"\nSample {i+1}:")
            print(f"  State: vx={test_states[i][2]:.2f}, vy={test_states[i][3]:.2f}")

            actions = []
            for behavior_id in range(3):
                action = policy.get_action(state, behavior_id, deterministic=True)
                actions.append(action.numpy())
                print(f"  {behavior_names[behavior_id]:10s}: "
                      f"ax={action[0].item():+.3f}, ay={action[1].item():+.3f}")

            # 检查是否有明显差异
            actions = np.array(actions)
            ay_std = actions[:, 1].std()
            if ay_std > 0.1:
                print(f"  ✓ Behaviors are distinct (ay_std={ay_std:.3f})")
            else:
                print(f"  ⚠ Behaviors may be too similar (ay_std={ay_std:.3f})")


def test_temporal_consistency(policy, test_states, test_behavior_ids, n_episodes=5):
    """
    测试策略的时间一致性

    给定一个行为，策略在连续 state 上的输出应该平滑
    """
    print("\n" + "="*70)
    print("TEST 2: Temporal Consistency")
    print("="*70)

    behavior_names = ["Straight", "Left", "Right"]

    for behavior_id in range(3):
        # 找到该 behavior 的连续样本
        mask = test_behavior_ids == behavior_id
        behavior_states = test_states[mask]

        if len(behavior_states) < 10:
            continue

        # 取前10个样本
        states_seq = torch.FloatTensor(behavior_states[:10])

        with torch.no_grad():
            actions = []
            for state in states_seq:
                action = policy.get_action(state, behavior_id, deterministic=True)
                actions.append(action.numpy())

        actions = np.array(actions)

        # 计算动作变化的平滑度
        action_diffs = np.diff(actions, axis=0)
        mean_diff = np.abs(action_diffs).mean(axis=0)

        print(f"\n{behavior_names[behavior_id]}:")
        print(f"  Mean action change: ax={mean_diff[0]:.4f}, ay={mean_diff[1]:.4f}")
        if mean_diff.max() < 0.5:
            print(f"  ✓ Actions are smooth")
        else:
            print(f"  ⚠ Actions may be too jerky")


def visualize_behavior_distribution(policy, test_states, n_samples=1000):
    """
    可视化不同 behavior 的动作分布
    """
    print("\n" + "="*70)
    print("TEST 3: Action Distribution Visualization")
    print("="*70)

    behavior_names = ["Straight", "Left", "Right"]
    colors = ['blue', 'green', 'red']

    # 随机采样
    indices = np.random.choice(len(test_states), size=n_samples, replace=False)
    sample_states = torch.FloatTensor(test_states[indices])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for behavior_id in range(3):
        with torch.no_grad():
            actions = []
            for state in sample_states:
                action = policy.get_action(state.unsqueeze(0), behavior_id, deterministic=True)
                actions.append(action.numpy())

        actions = np.array(actions)

        # 绘制 (ax, ay) 散点图
        ax = axes[behavior_id]
        ax.scatter(actions[:, 0], actions[:, 1], alpha=0.3, c=colors[behavior_id], s=10)
        ax.set_xlabel('ax (m/s²)')
        ax.set_ylabel('ay (m/s²)')
        ax.set_title(f'{behavior_names[behavior_id]} Behavior')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('simple_approach/results/behavior_action_distribution.png', dpi=150)
    print("✓ Saved: simple_approach/results/behavior_action_distribution.png")


def test_specific_scenarios(policy):
    """
    测试特定场景
    """
    print("\n" + "="*70)
    print("TEST 4: Specific Scenarios")
    print("="*70)

    behavior_names = ["Straight", "Left", "Right"]

    # 场景1: 高速直行
    print("\nScenario 1: High-speed straight driving")
    state = torch.zeros(34)
    state[2] = 30.0  # vx = 30 m/s
    state[3] = 0.0   # vy = 0

    for behavior_id in range(3):
        action = policy.get_action(state.unsqueeze(0), behavior_id, deterministic=True)
        print(f"  {behavior_names[behavior_id]:10s}: "
              f"ax={action[0].item():+.3f}, ay={action[1].item():+.3f}")

    # 场景2: 低速准备变道
    print("\nScenario 2: Low-speed, preparing for lane change")
    state = torch.zeros(34)
    state[2] = 15.0  # vx = 15 m/s
    state[3] = 0.5   # vy = 0.5 m/s (already drifting)

    for behavior_id in range(3):
        action = policy.get_action(state.unsqueeze(0), behavior_id, deterministic=True)
        print(f"  {behavior_names[behavior_id]:10s}: "
              f"ax={action[0].item():+.3f}, ay={action[1].item():+.3f}")


def main():
    print("="*80)
    print(" TESTING CONDITIONAL POLICY")
    print("="*80)

    # 加载模型
    policy, config = load_trained_policy()

    # 加载测试数据
    print("\nLoading test data...")
    dataset = Dataset(
        general_files={
            "cf": "high_quality_data/cf.csv",
            "dlc": "high_quality_data/dlc.csv",
            "mlc": "high_quality_data/mlc.csv"
        },
        data_files=["high_quality_data/data1f.csv"],
        cached_origin_file="cached_origin_hq.pkl",
        cached_generated_file="cached_generated_hq.pkl"
    )

    states, actions, behavior_ids = load_data_from_dataset(dataset)

    # 使用后10%作为测试集
    n_test = int(0.1 * len(states))
    test_states = states[-n_test:]
    test_actions = actions[-n_test:]
    test_behavior_ids = behavior_ids[-n_test:]

    print(f"✓ Test data loaded: {n_test:,} samples")

    # 运行测试
    test_behavior_conditioning(policy, test_states, n_samples=5)
    test_temporal_consistency(policy, test_states, test_behavior_ids, n_episodes=5)
    visualize_behavior_distribution(policy, test_states, n_samples=1000)
    test_specific_scenarios(policy)

    print("\n" + "="*80)
    print("✓ ALL TESTS COMPLETE")
    print("="*80)
    print("\nCheck the results in simple_approach/results/")
    print("="*80)


if __name__ == "__main__":
    main()
