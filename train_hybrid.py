"""
混合训练方案: Behavior Cloning + RL Fine-tuning

阶段1: Behavior Cloning (监督学习)
- 使用平衡数据快速预训练
- 获得合理的初始策略

阶段2: PPO Fine-tuning (强化学习)
- 在环境中继续优化
- 通过reward信号改进策略
"""

import sys
import os
import torch
import numpy as np
import random
import pickle
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_approach.models.conditional_policy import ConditionalPolicy, ValueNetwork
from simple_approach.training.trainer import ConditionalPolicyTrainer, load_data_from_dataset
from simple_approach.simple_environment import SimpleEnvironment
from simple_approach.train_rl import PPOTrainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    print("="*80)
    print(" HYBRID TRAINING: BC Pretraining + RL Fine-tuning")
    print("="*80)

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'simple_approach/results/hybrid_{timestamp}'
    log_dir = 'simple_approach/training_logs'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Results will be saved to: {save_dir}")
    print(f"Logs will be saved to: {log_dir}")

    # ========================================================================
    # [1/4] LOAD DATA
    # ========================================================================
    print("\n" + "="*80)
    print("[1/4] LOADING BALANCED DATA")
    print("="*80)

    with open('cached_generated_balanced.pkl', 'rb') as f:
        data = pickle.load(f)

    print(f"  Total samples: {len(data):,}")

    # 提取states, actions, behavior_ids
    states = []
    actions = []
    behavior_ids = []

    behavior_to_id = {"straight": 0, "left": 1, "right": 2}

    for state, action, next_state, done, behavior_label in data:
        states.append(state)
        actions.append(action)
        behavior_ids.append(behavior_to_id[behavior_label])

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    behavior_ids = np.array(behavior_ids, dtype=np.int64)

    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")

    # ========================================================================
    # [2/4] STAGE 1: BEHAVIOR CLONING
    # ========================================================================
    print("\n" + "="*80)
    print("[2/4] STAGE 1: BEHAVIOR CLONING PRETRAINING")
    print("="*80)

    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    # 创建模型
    policy = ConditionalPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        num_behaviors=3,
        hidden_dim=128,
        embedding_dim=32
    )

    print(f"\nPolicy params: {sum(p.numel() for p in policy.parameters()):,}")

    # BC训练
    bc_trainer = ConditionalPolicyTrainer(policy, device=device)

    print("\nTraining with Behavior Cloning...")
    train_losses, val_losses = bc_trainer.behavior_cloning(
        states, actions, behavior_ids,
        epochs=100,
        batch_size=256,
        lr=3e-4,
        weight_decay=1e-5,
        validation_split=0.1
    )

    # 保存BC模型
    bc_model_path = f'{save_dir}/policy_bc.pth'
    torch.save(policy.state_dict(), bc_model_path)
    print(f"\n✓ BC model saved: {bc_model_path}")

    # 保存归一化参数
    norm_params_path = f'{save_dir}/normalization_params.pth'
    normalization_params = {
        'state_mean': bc_trainer.state_mean,
        'state_std': bc_trainer.state_std,
        'action_mean': bc_trainer.action_mean,
        'action_std': bc_trainer.action_std
    }
    torch.save(normalization_params, norm_params_path)
    print(f"✓ Normalization params saved: {norm_params_path}")

    # 评估BC模型
    print("\nEvaluating BC model...")
    n_test = int(len(states) * 0.1)
    test_indices = np.random.permutation(len(states))[:n_test]

    for threshold in [0.5, 1.0, 2.0]:
        accuracy, mean_error = bc_trainer.evaluate_accuracy(
            states[test_indices],
            actions[test_indices],
            behavior_ids[test_indices],
            threshold=threshold
        )
        print(f"  Threshold {threshold:.1f}: Accuracy={100*accuracy:.1f}%, "
              f"Error: ax={mean_error[0]:.2f}, ay={mean_error[1]:.2f}")

    # ========================================================================
    # [3/4] STAGE 2: RL FINE-TUNING
    # ========================================================================
    print("\n" + "="*80)
    print("[3/4] STAGE 2: PPO FINE-TUNING")
    print("="*80)

    # 创建环境
    env = SimpleEnvironment(data=data, max_steps=50)

    # 创建Value网络
    value_net = ValueNetwork(
        state_dim=state_dim,
        num_behaviors=3,
        hidden_dim=128,
        embedding_dim=32
    )

    print(f"Value params: {sum(p.numel() for p in value_net.parameters()):,}")

    # PPO训练器（使用BC预训练的policy）
    ppo_trainer = PPOTrainer(policy, value_net, env, device=device)

    print("\nFine-tuning with PPO...")
    print("  Iterations: 200")
    print("  Episodes per iteration: 20")

    # 开始PPO训练
    for iteration in range(200):
        # 收集轨迹
        trajectories = ppo_trainer.collect_trajectories(n_episodes=20)

        # 计算advantages
        states_rl, actions_rl, behavior_ids_rl, advantages, returns, old_values = \
            ppo_trainer.compute_advantages(trajectories)

        # PPO更新
        policy_loss, value_loss = ppo_trainer.ppo_update(
            states_rl, actions_rl, behavior_ids_rl, advantages, returns, old_values,
            policy_lr=1e-4,  # 降低学习率（因为已经有BC初始化）
            value_lr=5e-4,
            clip_epsilon=0.2,
            n_epochs=5,  # 减少epochs避免overfitting
            batch_size=64
        )

        # 统计
        avg_reward = np.mean([t[2] for t in trajectories])

        # 打印进度
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Iter {iteration+1:3d}/200: "
                  f"Avg Reward = {avg_reward:7.2f}, "
                  f"Policy Loss = {policy_loss:.4f}, "
                  f"Value Loss = {value_loss:.4f}")

        # 定期保存
        if (iteration + 1) % 50 == 0:
            torch.save(policy.state_dict(),
                      f'{save_dir}/policy_rl_iter_{iteration+1}.pth')
            torch.save(value_net.state_dict(),
                      f'{save_dir}/value_rl_iter_{iteration+1}.pth')

    # 保存最终模型
    final_policy_path = f'{save_dir}/policy_final.pth'
    final_value_path = f'{save_dir}/value_final.pth'
    torch.save(policy.state_dict(), final_policy_path)
    torch.save(value_net.state_dict(), final_value_path)

    print(f"\n✓ Final RL model saved:")
    print(f"  Policy: {final_policy_path}")
    print(f"  Value: {final_value_path}")

    # ========================================================================
    # [4/4] EVALUATION & ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("[4/4] FINAL EVALUATION")
    print("="*80)

    # 行为条件化测试
    print("\nBehavior Conditioning Test:")
    print("Testing if different behaviors produce different actions...")

    policy.eval()

    behavior_names = ['Straight', 'Left', 'Right']
    n_test_states = 5

    all_diffs = {'S-L': [], 'S-R': [], 'L-R': []}

    for test_idx in range(n_test_states):
        test_state = torch.randn(state_dim)

        actions_by_behavior = {}
        for behavior_id, behavior_name in enumerate(behavior_names):
            with torch.no_grad():
                action = policy.get_action(test_state, behavior_id, deterministic=True).numpy()
            actions_by_behavior[behavior_name] = action

        # 计算差异
        diff_SL = np.abs(actions_by_behavior['Straight'] - actions_by_behavior['Left'])
        diff_SR = np.abs(actions_by_behavior['Straight'] - actions_by_behavior['Right'])
        diff_LR = np.abs(actions_by_behavior['Left'] - actions_by_behavior['Right'])

        all_diffs['S-L'].append(diff_SL)
        all_diffs['S-R'].append(diff_SR)
        all_diffs['L-R'].append(diff_LR)

        if test_idx == 0:
            print(f"\nExample (test state 1):")
            for name, action in actions_by_behavior.items():
                print(f"  {name:10s}: ax={action[0]:+7.3f}, ay={action[1]:+7.3f}")

    # 统计分析
    print(f"\nAverage Action Differences (across {n_test_states} states):")
    for key, diffs in all_diffs.items():
        diffs = np.array(diffs)
        avg_ax_diff = diffs[:, 0].mean()
        avg_ay_diff = diffs[:, 1].mean()
        print(f"  {key}: ax_diff={avg_ax_diff:.3f}, ay_diff={avg_ay_diff:.3f}")

    # 保存结果总结
    summary_path = f'{save_dir}/TRAINING_SUMMARY.md'
    with open(summary_path, 'w') as f:
        f.write("# Hybrid Training Summary\n\n")
        f.write(f"Training Date: {datetime.now()}\n\n")
        f.write("## Configuration\n")
        f.write(f"- Data samples: {len(data):,}\n")
        f.write(f"- State dim: {state_dim}\n")
        f.write(f"- Action dim: {action_dim}\n")
        f.write(f"- Policy params: {sum(p.numel() for p in policy.parameters()):,}\n\n")
        f.write("## Stage 1: BC\n")
        f.write(f"- Epochs: 100\n")
        f.write(f"- Batch size: 256\n")
        f.write(f"- Best val loss: {min(val_losses):.4f}\n\n")
        f.write("## Stage 2: RL\n")
        f.write(f"- Iterations: 200\n")
        f.write(f"- Episodes per iter: 20\n")
        f.write(f"- Final avg reward: {avg_reward:.2f}\n\n")
        f.write("## Behavior Conditioning Results\n")
        for key, diffs in all_diffs.items():
            diffs = np.array(diffs)
            f.write(f"- {key}: ax={diffs[:, 0].mean():.3f}, ay={diffs[:, 1].mean():.3f}\n")

    print(f"\n✓ Training summary saved: {summary_path}")

    print("\n" + "="*80)
    print("✓ HYBRID TRAINING COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {save_dir}/")
    print(f"Logs saved to: {log_dir}/")


if __name__ == "__main__":
    main()
