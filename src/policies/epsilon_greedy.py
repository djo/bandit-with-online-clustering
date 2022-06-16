from typing import Tuple

import numpy as np
from numpy.random import Generator

from src.policies.policy import Policy


class EpsilonGreedy(Policy):
    def __init__(self, num_actions: int, epsilon: float, rng: Generator):
        """
        Create EpsilonGreed policy playing the empirically best action
        with probability (1 - epsilon), otherwise explores uniformly at random.
        """
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError(f"Epsilon {epsilon} must be in [0.0, 1.0]")
        self._num_actions = num_actions
        self._epsilon = epsilon
        self._rng = rng
        self._current_action = -1
        self.cumulative_rewards = np.zeros(num_actions, dtype=np.float32)
        self.actions_stats = np.zeros(num_actions, dtype=np.int32)

    def select(self, t: int) -> int:
        if self._rng.random() < self._epsilon:
            self._current_action = self._rng.choice(self._num_actions)
            return self._current_action
        self._current_action, _ = self.empirically_best_action()
        return self._current_action

    def feed_reward(self, t: int, action: int, reward: float):
        if action != self._current_action:
            raise ValueError(
                f"Expected the reward for action {self._current_action}, but got for {action}"
            )
        self.cumulative_rewards[action] += reward
        self.actions_stats[action] += 1
        return

    def empirically_best_action(self) -> Tuple[int, float]:
        if np.count_nonzero(self.cumulative_rewards) == 0:
            return self._rng.choice(self._num_actions), 0.0
        idx = np.where(self.actions_stats != 0)
        i = np.argmax(self.cumulative_rewards[idx] / self.actions_stats[idx])
        action = idx[0][i]
        return action, self.cumulative_rewards[action] / self.actions_stats[action]
