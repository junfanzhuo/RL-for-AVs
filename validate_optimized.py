"""
验证优化后的模型
检查 Left, Right, Straight 的 ay 是否符合预期
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_approach.models.conditional_policy import ConditionalPolicy

def validate_model(model_path, n_test_states=30):
    """
    验证模型的行为条件化

    预期:
    - Straight: ay ≈ 0
    - Left: ay > 0 (向左)
    - Right: ay < 0 (向右)
    """

    print("="*80)
    print("MODEL VALIDATION")
    print("="*80)
    print(f"\nModel: {model_path}")
    print(f"Testing on {n_test_states} random states\n")

    # 加载模型
    policy = ConditionalPolicy(state_dim=34, action_dim=2, num_behaviors=3)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    behavior_names = ['Straight', 'Left', 'Right']

    # 收集所有动作
    actions_by_behavior = {name: [] for name in behavior_names}

    for test_idx in range(n_test_states):
        test_state = torch.randn(34)

        for behavior_id, behavior_name in enumerate(behavior_names):
            with torch.no_grad():
                action = policy.get_action(test_state, behavior_id, deterministic=True).numpy()
            actions_by_behavior[behavior_name].append(action)

    # 分析结果
    print("="*80)
    print("RESULTS")
    print("="*80)

    for behavior_name in behavior_names:
        actions = np.array(actions_by_behavior[behavior_name])
        ax_values = actions[:, 0]
        ay_values = actions[:, 1]

        ax_mean = ax_values.mean()
        ay_mean = ay_values.mean()

        # 统计 ay 的符号
        ay_positive_ratio = (ay_values > 0).mean()
        ay_negative_ratio = (ay_values < 0).mean()

        print(f"\n{behavior_name}:")
        print(f"  ax: mean={ax_mean:+7.3f}, std={ax_values.std():.3f}")
        print(f"  ay: mean={ay_mean:+7.3f}, std={ay_values.std():.3f}")
        print(f"  ay>0: {ay_positive_ratio*100:.1f}%, ay<0: {ay_negative_ratio*100:.1f}%")

    # 验证检查
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)

    straight_ay = np.array(actions_by_behavior['Straight'])[:, 1]
    left_ay = np.array(actions_by_behavior['Left'])[:, 1]
    right_ay = np.array(actions_by_behavior['Right'])[:, 1]

    checks = []

    # Check 1: Left ay 应该是正的
    left_ay_mean = left_ay.mean()
    check1 = left_ay_mean > 0
    checks.append(check1)
    print(f"\n✓ Check 1: Left ay mean > 0")
    print(f"  Result: {left_ay_mean:+.3f} {'✓ PASS' if check1 else '✗ FAIL'}")

    # Check 2: Right ay 应该是负的
    right_ay_mean = right_ay.mean()
    check2 = right_ay_mean < 0
    checks.append(check2)
    print(f"\n✓ Check 2: Right ay mean < 0")
    print(f"  Result: {right_ay_mean:+.3f} {'✓ PASS' if check2 else '✗ FAIL'}")

    # Check 3: Straight ay 应该接近0
    straight_ay_mean = straight_ay.mean()
    check3 = abs(straight_ay_mean) < abs(left_ay_mean) and abs(straight_ay_mean) < abs(right_ay_mean)
    checks.append(check3)
    print(f"\n✓ Check 3: Straight ay closest to 0")
    print(f"  |Straight|={abs(straight_ay_mean):.3f}, |Left|={abs(left_ay_mean):.3f}, |Right|={abs(right_ay_mean):.3f}")
    print(f"  {'✓ PASS' if check3 else '✗ FAIL'}")

    # Check 4: Left vs Right 差异应该大
    left_right_diff = abs(left_ay_mean - right_ay_mean)
    check4 = left_right_diff > 1.0
    checks.append(check4)
    print(f"\n✓ Check 4: |Left - Right| > 1.0")
    print(f"  Result: {left_right_diff:.3f} {'✓ PASS' if check4 else '✗ FAIL'}")

    # Check 5: Left 的 ay>0 比例应该高
    left_positive_ratio = (left_ay > 0).mean()
    check5 = left_positive_ratio > 0.5
    checks.append(check5)
    print(f"\n✓ Check 5: Left ay>0 ratio > 50%")
    print(f"  Result: {left_positive_ratio*100:.1f}% {'✓ PASS' if check5 else '✗ FAIL'}")

    # Check 6: Right 的 ay<0 比例应该高
    right_negative_ratio = (right_ay < 0).mean()
    check6 = right_negative_ratio > 0.5
    checks.append(check6)
    print(f"\n✓ Check 6: Right ay<0 ratio > 50%")
    print(f"  Result: {right_negative_ratio*100:.1f}% {'✓ PASS' if check6 else '✗ FAIL'}")

    # 总结
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    success_rate = sum(checks) / len(checks)
    print(f"\nSuccess Rate: {success_rate*100:.0f}% ({sum(checks)}/{len(checks)} checks passed)")

    if all(checks):
        print("\n🎉 ALL CHECKS PASSED! Model is working correctly.")
    else:
        print(f"\n⚠️  {len(checks) - sum(checks)} check(s) failed. Model needs further tuning.")

    return checks


if __name__ == "__main__":
    model_path = "simple_approach/results/hybrid_20251110_092530/policy_final.pth"
    validate_model(model_path, n_test_states=30)
