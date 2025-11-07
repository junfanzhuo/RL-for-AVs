"""
Simple Approach: Conditional Policy Training
使用 Behavior Cloning 训练条件化策略

优点：
1. 简单、直接、稳定
2. 完全利用标签数据
3. 训练快速（相比 DIAYN）
4. 易于调试和理解
"""

import sys
import os
import torch
import numpy as np
import random

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import Dataset
from simple_approach.models.conditional_policy import ConditionalPolicy, ValueNetwork
from simple_approach.training.trainer import ConditionalPolicyTrainer, load_data_from_dataset


def set_seed(seed=42):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    print("="*80)
    print(" SIMPLE APPROACH: CONDITIONAL POLICY TRAINING")
    print("="*80)

    # 设置随机种子
    set_seed(42)

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # ==================== 加载数据 ====================
    print("\n" + "="*80)
    print("[1/4] LOADING DATASET")
    print("="*80)

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

    print("\nLoading data from dataset...")
    states, actions, behavior_ids = load_data_from_dataset(dataset)

    print(f"\n✓ Data loaded successfully!")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Behavior IDs shape: {behavior_ids.shape}")

    # 统计行为分布
    unique, counts = np.unique(behavior_ids, return_counts=True)
    behavior_names = ["Straight", "Left", "Right"]
    print(f"\nBehavior Distribution:")
    for bid, count in zip(unique, counts):
        print(f"  {behavior_names[bid]:10s}: {count:6,} ({100*count/len(behavior_ids):.1f}%)")

    # ==================== 创建模型 ====================
    print("\n" + "="*80)
    print("[2/4] CREATING MODELS")
    print("="*80)

    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    num_behaviors = 3

    print(f"\nModel Configuration:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Num behaviors: {num_behaviors}")
    print(f"  Hidden dim: 128")
    print(f"  Embedding dim: 32")

    policy = ConditionalPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        num_behaviors=num_behaviors,
        hidden_dim=128,
        embedding_dim=32,
        use_learned_embedding=True
    )

    # 统计参数数量
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # ==================== 训练模型 ====================
    print("\n" + "="*80)
    print("[3/4] TRAINING")
    print("="*80)

    trainer = ConditionalPolicyTrainer(policy, device=device)

    # Behavior Cloning
    train_losses, val_losses = trainer.behavior_cloning(
        states=states,
        actions=actions,
        behavior_ids=behavior_ids,
        epochs=50,
        batch_size=128,
        lr=3e-4,
        weight_decay=1e-5,
        validation_split=0.1
    )

    # ==================== 评估模型 ====================
    print("\n" + "="*80)
    print("[4/4] EVALUATION")
    print("="*80)

    # 使用验证集评估
    n_test = int(0.1 * len(states))
    test_states = states[-n_test:]
    test_actions = actions[-n_test:]
    test_behavior_ids = behavior_ids[-n_test:]

    print(f"\nEvaluating on {n_test:,} test samples...")

    # 不同阈值下的准确率
    thresholds = [0.05, 0.1, 0.2, 0.5]
    print(f"\nAccuracy at different thresholds:")
    for threshold in thresholds:
        accuracy, mean_error = trainer.evaluate_accuracy(
            test_states, test_actions, test_behavior_ids, threshold=threshold
        )
        print(f"  Threshold {threshold:.2f}: {accuracy*100:.1f}%  "
              f"(Mean error: ax={mean_error[0]:.4f}, ay={mean_error[1]:.4f})")

    # 分行为统计准确率
    print(f"\nPer-behavior Accuracy (threshold=0.1):")
    for behavior_id, behavior_name in enumerate(behavior_names):
        mask = test_behavior_ids == behavior_id
        if mask.sum() > 0:
            acc, err = trainer.evaluate_accuracy(
                test_states[mask],
                test_actions[mask],
                test_behavior_ids[mask],
                threshold=0.1
            )
            print(f"  {behavior_name:10s}: {acc*100:.1f}%  "
                  f"(ax_err={err[0]:.4f}, ay_err={err[1]:.4f})")

    # ==================== 保存结果 ====================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # 保存训练曲线
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Behavior Cloning Training')
    plt.legend()
    plt.grid(True)
    plt.savefig('simple_approach/results/training_curves.png', dpi=150)
    print("✓ Training curves saved: simple_approach/results/training_curves.png")

    # 保存最终模型
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'config': {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'num_behaviors': num_behaviors,
            'hidden_dim': 128,
            'embedding_dim': 32
        }
    }, 'simple_approach/results/final_policy.pth')
    print("✓ Final model saved: simple_approach/results/final_policy.pth")

    # ==================== 完成 ====================
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run: python simple_approach/test_simple.py")
    print("     to test the trained policy")
    print("  2. Review PROJECT_LOG.md for CPT integration plan")
    print("="*80)


if __name__ == "__main__":
    main()
