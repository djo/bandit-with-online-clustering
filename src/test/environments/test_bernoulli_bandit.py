import numpy as np
from numpy.random import default_rng, Generator, PCG64
from numpy.testing import assert_array_equal, assert_array_almost_equal

from src.environments.bernoulli_bandit import BernoulliBandit


class TestBernoulliBandit:
    def test_draw(self):
        with_certainty = BernoulliBandit(means=[0.0, 1.0], horizon=2, rng=default_rng())
        assert with_certainty.draw(t=0, action=0) == 0
        assert with_certainty.draw(t=1, action=1) == 1

        env = BernoulliBandit(means=[0.5], horizon=2, rng=Generator(PCG64(42)))
        assert env.draw(t=0, action=0) == 1
        assert env.draw(t=0, action=0) == 0

    def test_means_tracking(self):
        env = BernoulliBandit(means=[0.4, 0.6], horizon=4, rng=default_rng())
        env.draw(t=0, action=0)
        env.draw(t=1, action=1)

        env.means = np.array([0.7, 0.3])
        env.draw(t=2, action=0)
        env.draw(t=3, action=1)

        assert_array_equal(env.largest_means, [0.6, 0.6, 0.7, 0.7])
        assert_array_equal(env.optimal_actions, [1, 1, 0, 0])
        assert_array_almost_equal(env.suboptimality, [0.2, 0.0, 0.0, 0.4])
