"""
简化的环境，用于训练 Conditional Policy
不需要 encoder, discriminator 等 DIAYN 组件
"""

import numpy as np
import random


class SimpleEnvironment:
    """
    简化的驾驶环境

    Reward 设计:
    - 基于 behavior_id，奖励policy执行相应的动作
    - Straight: 奖励 |ay| 小（保持直行）
    - Left: 奖励 ay > 0（左转/左变道）
    - Right: 奖励 ay < 0（右转/右变道）
    """

    def __init__(self, data, max_steps=100):
        """
        Args:
            data: list of (state, action, next_state, done, behavior_label)
            max_steps: 每个episode的最大步数
        """
        self.data = data
        self.max_steps = max_steps

        # 获取维度
        sample = data[0]
        self.state_dim = len(sample[0])
        self.action_dim = len(sample[1])

        # 当前状态
        self.current_state = None
        self.current_behavior_id = None  # 当前episode的behavior目标
        self.step_count = 0

        print(f"SimpleEnvironment initialized:")
        print(f"  State dim: {self.state_dim}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Data samples: {len(data):,}")
        print(f"  Max steps per episode: {max_steps}")

    def reset(self, behavior_id=None):
        """
        重置环境

        Args:
            behavior_id: 指定行为目标 (0=straight, 1=left, 2=right)
                        如果为None，则随机选择

        Returns:
            state: 初始状态
        """
        self.step_count = 0

        # 随机选择一个初始状态
        idx = random.randint(0, len(self.data) - 1)
        self.current_state = np.array(self.data[idx][0], dtype=np.float32)

        # 设置行为目标
        if behavior_id is None:
            self.current_behavior_id = random.randint(0, 2)
        else:
            self.current_behavior_id = behavior_id

        return self.current_state.copy()

    def step(self, action):
        """
        执行一步

        Args:
            action: [ax, ay] 加速度

        Returns:
            next_state, reward, done, info
        """
        self.step_count += 1

        # 根据动作更新状态（简化版本：只更新速度和位置）
        ax, ay = action[0], action[1]

        # 当前状态
        x, vx, y_rel, vy = self.current_state[0], self.current_state[1], self.current_state[2], self.current_state[3]

        # 简单的运动学更新
        dt = 1/30  # 30 FPS
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        x_new = x + vx * dt
        y_rel_new = y_rel + vy * dt

        # 更新状态（保持其他维度不变）
        next_state = self.current_state.copy()
        next_state[0] = x_new
        next_state[1] = vx_new
        next_state[2] = y_rel_new
        next_state[3] = vy_new

        # 计算 reward（基于 behavior_id）
        reward = self._compute_reward(action, self.current_behavior_id)

        # 检查是否完成
        done = (self.step_count >= self.max_steps) or self._check_collision(next_state)

        # 更新当前状态
        self.current_state = next_state

        info = {
            'behavior_id': self.current_behavior_id,
            'step': self.step_count
        }

        return next_state, reward, done, info

    def _compute_reward(self, action, behavior_id):
        """
        计算reward

        Args:
            action: [ax, ay]
            behavior_id: 0=straight, 1=left, 2=right

        Returns:
            reward: 标量
        """
        ax, ay = action[0], action[1]

        reward = 0.0

        if behavior_id == 0:  # Straight
            # 奖励横向加速度小（保持直行）
            reward = -abs(ay) * 0.5
            # 奖励合理的纵向加速度
            reward += -abs(ax) * 0.1

        elif behavior_id == 1:  # Left
            # 奖励正的横向加速度（向左）
            if ay > 0:
                reward = ay * 2.0  # 奖励
            else:
                reward = ay * 5.0  # 惩罚

            # 轻微奖励稳定的纵向加速度
            reward += -abs(ax) * 0.1

        elif behavior_id == 2:  # Right
            # 奖励负的横向加速度（向右）
            if ay < 0:
                reward = -ay * 2.0  # 奖励
            else:
                reward = -ay * 5.0  # 惩罚

            # 轻微奖励稳定的纵向加速度
            reward += -abs(ax) * 0.1

        # 惩罚过大的加速度（不安全）
        if abs(ax) > 10.0:
            reward -= 1.0
        if abs(ay) > 10.0:
            reward -= 1.0

        return reward

    def _check_collision(self, state):
        """
        检查是否发生碰撞

        简化版本：检查横向位置是否超出车道
        """
        y_rel = state[2]

        # 假设车道宽度约为 3.5m
        # y_rel 是相对于前车的横向距离
        # 如果 |y_rel| > 5m，认为已经偏离太远
        if abs(y_rel) > 5.0:
            return True

        return False

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim
