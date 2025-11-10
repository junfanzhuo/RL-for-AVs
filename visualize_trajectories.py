"""
可视化训练后模型的轨迹和速度
展示 Straight, Left, Right 三种行为
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_approach.models.conditional_policy import ConditionalPolicy

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def simulate_trajectory(policy, initial_state, behavior_id, n_steps=100):
    """
    模拟一辆车的轨迹

    Args:
        policy: 训练好的策略
        initial_state: 初始状态
        behavior_id: 行为ID (0=straight, 1=left, 2=right)
        n_steps: 仿真步数

    Returns:
        trajectory: 轨迹数据 (n_steps, state_dim)
        actions: 动作序列 (n_steps, action_dim)
    """
    state = initial_state.copy()
    trajectory = [state.copy()]
    actions = []

    dt = 1/30  # 30 FPS

    for step in range(n_steps):
        # 获取动作
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action = policy.get_action(state_tensor, behavior_id, deterministic=True).numpy()

        actions.append(action.copy())

        # 更新状态（简化的运动学模型）
        ax, ay = action[0], action[1]

        # 提取当前状态
        x = state[0]
        vx = state[1]
        y = state[2]
        vy = state[3]

        # 更新
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        x_new = x + vx * dt + 0.5 * ax * dt**2
        y_new = y + vy * dt + 0.5 * ay * dt**2

        # 更新状态
        state[0] = x_new
        state[1] = vx_new
        state[2] = y_new
        state[3] = vy_new

        trajectory.append(state.copy())

    return np.array(trajectory), np.array(actions)


def plot_single_behavior(ax_traj, ax_vel, ax_acc, trajectory, actions, behavior_name, color):
    """
    绘制单个行为的轨迹、速度和加速度
    """
    # 提取数据
    x = trajectory[:, 0]
    y = trajectory[:, 2]
    vx = trajectory[:, 1]
    vy = trajectory[:, 3]

    ax_values = actions[:, 0]
    ay_values = actions[:, 1]

    # 时间轴 - trajectory 比 actions 多一个点（初始状态）
    time_traj = np.arange(len(trajectory)) / 30.0  # 轨迹时间
    time_action = np.arange(len(actions)) / 30.0   # 动作时间

    # 计算总速度
    v_total = np.sqrt(vx**2 + vy**2)

    # === 轨迹图 ===
    ax_traj.plot(x, y, color=color, linewidth=2.5, label=behavior_name, zorder=3)
    ax_traj.scatter(x[0], y[0], color='green', s=200, marker='o',
                    edgecolors='darkgreen', linewidths=2, zorder=4, label='起点')
    ax_traj.scatter(x[-1], y[-1], color='red', s=200, marker='s',
                    edgecolors='darkred', linewidths=2, zorder=4, label='终点')

    # 添加方向箭头（每10个点）
    for i in range(0, len(x)-10, 10):
        dx = x[i+5] - x[i]
        dy = y[i+5] - y[i]
        ax_traj.arrow(x[i], y[i], dx, dy, head_width=2, head_length=3,
                     fc=color, ec=color, alpha=0.6, zorder=2)

    # 绘制车道线（假设3.5m宽）
    lane_width = 3.5
    ax_traj.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='车道中心')
    ax_traj.axhline(y=lane_width, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax_traj.axhline(y=-lane_width, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

    # 添加车辆起始位置的矩形（表示车辆）
    car_length = 4.5
    car_width = 1.8
    car_rect = Rectangle((x[0] - car_length/2, y[0] - car_width/2),
                         car_length, car_width,
                         linewidth=2, edgecolor='darkgreen', facecolor='lightgreen',
                         alpha=0.5, zorder=3)
    ax_traj.add_patch(car_rect)

    ax_traj.set_xlabel('纵向位置 X (m)', fontsize=12, fontweight='bold')
    ax_traj.set_ylabel('横向位置 Y (m)', fontsize=12, fontweight='bold')
    ax_traj.set_title(f'{behavior_name} - 轨迹', fontsize=14, fontweight='bold')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(loc='best', fontsize=10)
    ax_traj.set_aspect('equal', adjustable='box')

    # === 速度图 ===
    ax_vel.plot(time_traj, vx, color='blue', linewidth=2, label='纵向速度 Vx', marker='o', markersize=3)
    ax_vel.plot(time_traj, vy, color='red', linewidth=2, label='横向速度 Vy', marker='s', markersize=3)
    ax_vel.plot(time_traj, v_total, color='purple', linewidth=2.5, label='总速度 V',
               linestyle='--', marker='D', markersize=3)
    ax_vel.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_vel.set_xlabel('时间 (s)', fontsize=12, fontweight='bold')
    ax_vel.set_ylabel('速度 (m/s)', fontsize=12, fontweight='bold')
    ax_vel.set_title(f'{behavior_name} - 速度变化', fontsize=14, fontweight='bold')
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend(loc='best', fontsize=10)

    # === 加速度图 ===
    time_action = np.arange(len(actions)) / 30.0
    ax_acc.plot(time_action, ax_values, color='green', linewidth=2, label='纵向加速度 ax',
               marker='o', markersize=3)
    ax_acc.plot(time_action, ay_values, color='orange', linewidth=2, label='横向加速度 ay',
               marker='s', markersize=3)
    ax_acc.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # 标注平均值
    ax_acc.axhline(y=ay_values.mean(), color='orange', linestyle='--', linewidth=1.5,
                  alpha=0.7, label=f'ay 均值: {ay_values.mean():.2f}')

    ax_acc.set_xlabel('时间 (s)', fontsize=12, fontweight='bold')
    ax_acc.set_ylabel('加速度 (m/s²)', fontsize=12, fontweight='bold')
    ax_acc.set_title(f'{behavior_name} - 加速度变化', fontsize=14, fontweight='bold')
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc='best', fontsize=10)


def main():
    print("="*80)
    print("轨迹可视化")
    print("="*80)

    # 加载模型
    model_path = "simple_approach/results/hybrid_20251110_092530/policy_final.pth"
    print(f"\n加载模型: {model_path}")

    policy = ConditionalPolicy(state_dim=34, action_dim=2, num_behaviors=3)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    # 定义初始状态
    initial_state = np.zeros(34, dtype=np.float32)
    initial_state[0] = 0.0      # x = 0
    initial_state[1] = 10.0     # vx = 10 m/s (36 km/h)
    initial_state[2] = 0.0      # y = 0 (车道中心)
    initial_state[3] = 0.0      # vy = 0

    behaviors = [
        (0, 'Straight (直行)', 'blue'),
        (1, 'Left (左转)', 'green'),
        (2, 'Right (右转)', 'red')
    ]

    n_steps = 100  # 仿真 100 步 (约3.3秒)

    # 为每个行为生成单独的图
    for behavior_id, behavior_name, color in behaviors:
        print(f"\n生成 {behavior_name} 的轨迹...")

        # 模拟轨迹
        trajectory, actions = simulate_trajectory(policy, initial_state, behavior_id, n_steps)

        # 创建图形
        fig = plt.figure(figsize=(18, 5))

        ax_traj = plt.subplot(1, 3, 1)
        ax_vel = plt.subplot(1, 3, 2)
        ax_acc = plt.subplot(1, 3, 3)

        # 绘制
        plot_single_behavior(ax_traj, ax_vel, ax_acc, trajectory, actions, behavior_name, color)

        plt.tight_layout()

        # 保存
        filename = f"simple_approach/visualizations/{behavior_id}_{behavior_name.split()[0].lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  保存到: {filename}")
        plt.close()

    # 生成对比图（三个行为放在一起）
    print(f"\n生成对比图...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for behavior_id, behavior_name, color in behaviors:
        trajectory, actions = simulate_trajectory(policy, initial_state, behavior_id, n_steps)

        x = trajectory[:, 0]
        y = trajectory[:, 2]

        # 轨迹
        axes[0].plot(x, y, color=color, linewidth=2.5, label=behavior_name, alpha=0.8)
        axes[0].scatter(x[0], y[0], color=color, s=150, marker='o', edgecolors='black', linewidths=2, zorder=5)
        axes[0].scatter(x[-1], y[-1], color=color, s=150, marker='s', edgecolors='black', linewidths=2, zorder=5)

        # 速度
        time = np.arange(len(trajectory)) / 30.0
        vy = trajectory[:, 3]
        axes[1].plot(time, vy, color=color, linewidth=2.5, label=behavior_name, alpha=0.8)

        # 加速度
        time_action = np.arange(len(actions)) / 30.0
        ay_values = actions[:, 1]
        axes[2].plot(time_action, ay_values, color=color, linewidth=2.5, label=behavior_name, alpha=0.8)
        axes[2].axhline(y=ay_values.mean(), color=color, linestyle='--', linewidth=1.5, alpha=0.5)

    # 车道线
    lane_width = 3.5
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='车道中心')
    axes[0].axhline(y=lane_width, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    axes[0].axhline(y=-lane_width, color='gray', linestyle=':', linewidth=1, alpha=0.4)

    # 零线
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # 设置
    axes[0].set_xlabel('纵向位置 X (m)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('横向位置 Y (m)', fontsize=13, fontweight='bold')
    axes[0].set_title('轨迹对比', fontsize=15, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=11)
    axes[0].set_aspect('equal', adjustable='box')

    axes[1].set_xlabel('时间 (s)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('横向速度 Vy (m/s)', fontsize=13, fontweight='bold')
    axes[1].set_title('横向速度对比', fontsize=15, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=11)

    axes[2].set_xlabel('时间 (s)', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('横向加速度 ay (m/s²)', fontsize=13, fontweight='bold')
    axes[2].set_title('横向加速度对比', fontsize=15, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best', fontsize=11)

    plt.tight_layout()
    filename = "simple_approach/visualizations/comparison_all_behaviors.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  保存到: {filename}")
    plt.close()

    print("\n" + "="*80)
    print("✓ 可视化完成!")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  1. simple_approach/visualizations/0_straight.png")
    print(f"  2. simple_approach/visualizations/1_left.png")
    print(f"  3. simple_approach/visualizations/2_right.png")
    print(f"  4. simple_approach/visualizations/comparison_all_behaviors.png")
    print("="*80)


if __name__ == "__main__":
    main()
