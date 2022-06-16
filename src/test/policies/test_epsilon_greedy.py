import numpy as np
import pytest
from numpy.random import default_rng, PCG64, Generator

from src.policies.epsilon_greedy import EpsilonGreedy


class TestEpsilonGreedy:
    def test_select_action_to_explore(self):
        greedy = EpsilonGreedy(num_actions=2, epsilon=1.0, rng=Generator(PCG64(40)))
        greedy.cumulative_rewards = np.array([1.0, 1.0], dtype=np.float32)
        greedy.actions_stats = np.array([2, 1])
        assert greedy.select(0) == 0

    def test_select_action_to_exploit(self):
        greedy = EpsilonGreedy(num_actions=2, epsilon=0.0, rng=default_rng())
        greedy.cumulative_rewards = np.array([1.0, 1.0], dtype=np.float32)
        greedy.actions_stats = np.array([2, 1])
        assert greedy.select(0) == 1

    def test_feed_reward(self):
        greedy = EpsilonGreedy(num_actions=2, epsilon=0.0, rng=Generator(PCG64(42)))
        assert greedy.select(0) == 1
        with pytest.raises(ValueError, match=r"Expected the reward for action 1, but got for 0"):
            greedy.feed_reward(t=0, action=0, reward=0.0)
        greedy.feed_reward(t=0, action=1, reward=0.0)

    def test_empirically_best_action_uniformly_when_no_rewards_yet(self):
        greedy = EpsilonGreedy(num_actions=3, epsilon=0.1, rng=Generator(PCG64(42)))
        assert greedy.empirically_best_action() == (0, 0.0)

    def test_empirically_best_action(self):
        greedy = EpsilonGreedy(num_actions=3, epsilon=0.1, rng=default_rng())
        greedy.cumulative_rewards = np.array([0.0, 10.0, 1.0], dtype=np.float32)

        greedy.actions_stats = np.array([0, 11, 1])
        assert greedy.empirically_best_action() == (2, 1.0)

        greedy.actions_stats = np.array([0, 10, 1])
        assert greedy.empirically_best_action() == (1, 1.0)

        greedy.actions_stats = np.array([0, 2, 1])
        assert greedy.empirically_best_action() == (1, 5.0)
