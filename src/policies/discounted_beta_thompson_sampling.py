import numpy as np
from numpy.random import Generator

from src.policies.policy import Policy


class DiscountedBetaThompsonSampling(Policy):
    def __init__(self, num_actions: int, gamma: float, rng: Generator):
        """
        Create a discounted Thompson Sampling policy with Beta distribution as a prior one.
        """
        assert 0.0 <= gamma <= 1.0
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha_0 = 1.0
        self.beta_0 = 1.0
        self.alphas = np.zeros(num_actions, dtype=float)
        self.betas = np.zeros(num_actions, dtype=float)
        self.current_action = -1
        self._rng = rng
        self.cumulative_rewards = np.zeros(num_actions, dtype=np.float32)
        self.action_stats = np.zeros(num_actions, dtype=np.int32)

    def select(self, t: int) -> int:
        samples = np.zeros(self.num_actions, dtype=float)
        for i in range(self.num_actions):
            alpha = self.alphas[i] + self.alpha_0
            beta = self.betas[i] + self.beta_0
            samples[i] = self._rng.beta(alpha, beta)
        self.current_action = int(np.argmax(samples))
        return self.current_action

    def update(self, t: int, action: int, reward: float, phi: float):
        assert 0.0 <= phi <= 1.0
        if action != self.current_action:
            raise ValueError(
                f"Expected the reward for action {self.current_action}, but got for {action}"
            )

        self.cumulative_rewards[action] += reward
        self.action_stats[action] += 1

        for i in range(self.num_actions):
            self.alphas[i] = self.gamma * self.alphas[i]
            self.betas[i] = self.gamma * self.betas[i]
        self.alphas[action] += phi * reward
        self.betas[action] += phi * (1.0 - reward)

        return
