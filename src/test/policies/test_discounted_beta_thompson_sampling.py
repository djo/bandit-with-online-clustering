import pytest
from numpy.random import Generator, PCG64
from numpy.testing import assert_array_equal

from src.policies.discounted_beta_thompson_sampling import DiscountedBetaThompsonSampling


class TestDiscountedTestBetaThompsonSampling:
    def test_select_action_to_explore(self):
        ts = DiscountedBetaThompsonSampling(num_actions=2, gamma=1.0, rng=Generator(PCG64(40)))
        assert ts.select(t=0) == 0

    def test_update(self):
        ts = DiscountedBetaThompsonSampling(num_actions=2, gamma=1.0, rng=Generator(PCG64(42)))
        assert ts.select(t=0) == 1
        with pytest.raises(ValueError, match=r"Expected the reward for action 1, but got for 0"):
            ts.update(t=0, action=0, reward=0.0, phi=1.0)
        ts.update(t=0, action=1, reward=0.0, phi=1.0)
        assert_array_equal(ts.cumulative_rewards, [0.0, 0.0])
        assert_array_equal(ts.action_stats, [0, 1])

    def test_no_discounting(self):
        ts = DiscountedBetaThompsonSampling(num_actions=2, gamma=1.0, rng=Generator(PCG64(52)))

        assert ts.select(t=0) == 0
        ts.update(t=0, action=0, reward=1.0, phi=1.0)

        assert ts.select(t=1) == 1
        ts.update(t=1, action=1, reward=1.0, phi=1.0)
        assert ts.select(t=2) == 1
        ts.update(t=2, action=1, reward=1.0, phi=1.0)
        assert ts.select(t=3) == 1
        ts.update(t=3, action=1, reward=0.0, phi=1.0)

        assert_array_equal(ts.alphas, [1.0, 2.0])
        assert_array_equal(ts.betas, [0.0, 1.0])

    def test_with_gamma_discounting(self):
        ts = DiscountedBetaThompsonSampling(num_actions=2, gamma=0.5, rng=Generator(PCG64(52)))

        assert ts.select(t=0) == 0
        ts.update(t=0, action=0, reward=1.0, phi=1.0)

        assert ts.select(t=1) == 1
        ts.update(t=1, action=1, reward=1.0, phi=1.0)
        assert ts.select(t=2) == 1
        ts.update(t=2, action=1, reward=0.0, phi=1.0)

        assert_array_equal(
            ts.alphas, [(0.0 * 0.5 + 1.0) * 0.5 * 0.5, (0.0 * 0.5 * 0.5 + 1.0) * 0.5 + 0.0]
        )
        assert_array_equal(
            ts.betas, [(0.0 * 0.5 + 0.0) * 0.5 * 0.5, (0.0 * 0.5 * 0.5 + 0.0) * 0.5 + 1.0]
        )

    def test_with_gamma_phi_discounting(self):
        ts = DiscountedBetaThompsonSampling(num_actions=2, gamma=0.5, rng=Generator(PCG64(52)))

        assert ts.select(t=0) == 0
        ts.update(t=0, action=0, reward=1.0, phi=0.9)

        assert ts.select(t=1) == 1
        ts.update(t=1, action=1, reward=1.0, phi=0.8)
        assert ts.select(t=2) == 1
        ts.update(t=2, action=1, reward=0.0, phi=0.7)

        assert_array_equal(
            ts.alphas,
            [(0.0 * 0.5 + 0.9 * 1.0) * 0.5 * 0.5, (0.0 * 0.5 * 0.5 + 0.8 * 1.0) * 0.5 + 0.7 * 0.0],
        )
        assert_array_equal(
            ts.betas, [(0.0 * 0.5 + 0.0) * 0.5 * 0.5, (0.0 * 0.5 * 0.5 + 0.0) * 0.5 + 0.7 * 1.0]
        )

        assert_array_equal(ts.cumulative_rewards, [1.0, 1.0])
        assert_array_equal(ts.action_stats, [1, 2])
