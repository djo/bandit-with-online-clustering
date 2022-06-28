import numpy as np
from numpy.random import Generator

from src.policies.policy import Policy


class UniformRandom(Policy):
    def __init__(self, num_actions: int, rng: Generator):
        """
        Create a policy choosing an action uniformly at random.
        """
        self.num_actions = num_actions
        self._rng = rng
        self._current_action = -1
        self.cumulative_rewards = np.zeros(num_actions, dtype=np.float32)
        self.action_stats = np.zeros(num_actions, dtype=np.int32)

    def select(self, t: int) -> int:
        self._current_action = self._rng.choice(self.num_actions)
        return self._current_action

    def update(self, t: int, action: int, reward: float, phi: float):
        if action != self._current_action:
            raise ValueError(
                f"Expected the reward for action {self._current_action}, but got for {action}"
            )
        self.cumulative_rewards[action] += reward
        self.action_stats[action] += 1
        return
