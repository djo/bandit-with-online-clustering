import numpy as np
from numpy.random import Generator

from src.policies.policy import Policy


class DiscountedBetaThompsonSampling(Policy):
    def __init__(self, num_actions: int, horizon: int, gamma: float, rng: Generator):
        """
        Create a discounted Thompson Sampling policy with Beta distribution as a prior one.
        """
        assert 0.0 <= gamma <= 1.0
        self.num_actions = num_actions
        self.horizon = horizon
        self.gamma = gamma
        self.alpha_0 = 1.0
        self.beta_0 = 1.0
        self.alphas = np.zeros(num_actions, dtype=float)
        self.betas = np.zeros(num_actions, dtype=float)
        self._rng = rng
        self.rewards = np.zeros(horizon, dtype=np.float32)
        self.selected_actions = np.full(horizon, fill_value=-1, dtype=np.int32)

    def select(self, t: int) -> int:
        samples = np.zeros(self.num_actions, dtype=float)
        for i in range(self.num_actions):
            alpha = self.alphas[i] + self.alpha_0
            beta = self.betas[i] + self.beta_0
            samples[i] = self._rng.beta(alpha, beta)
        self.selected_actions[t] = int(np.argmax(samples))
        return self.selected_actions[t]

    def update(self, t: int, action: int, reward: float, phi: float):
        assert 0.0 <= phi <= 1.0
        if action != self.selected_actions[t]:
            raise ValueError(
                f"Expected the reward for action {self.selected_actions[t]}, but got for {action}"
            )

        self.selected_actions[t] = action
        self.rewards[t] = reward

        for i in range(self.num_actions):
            self.alphas[i] = self.gamma * self.alphas[i]
            self.betas[i] = self.gamma * self.betas[i]
        self.alphas[action] += phi * reward
        self.betas[action] += phi * (1.0 - reward)

        return
