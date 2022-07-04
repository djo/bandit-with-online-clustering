import numpy as np
from numpy.random import Generator

from src.policies.policy import Policy


class UniformRandom(Policy):
    def __init__(self, num_actions: int, horizon: int, rng: Generator):
        """
        Create a policy choosing an action uniformly at random.
        """
        self.num_actions = num_actions
        self.horizon = horizon
        self._rng = rng
        self.rewards = np.zeros(horizon, dtype=np.float32)
        self.selected_actions = np.full(horizon, fill_value=-1, dtype=np.int32)

    def select(self, t: int) -> int:
        self.selected_actions[t] = self._rng.choice(self.num_actions)
        return self.selected_actions[t]

    def update(self, t: int, action: int, reward: float, phi: float):
        if action != self.selected_actions[t]:
            raise ValueError(
                f"Expected the reward for action {self.selected_actions[t]}, but got for {action}"
            )
        self.selected_actions[t] = action
        self.rewards[t] = reward
        return
