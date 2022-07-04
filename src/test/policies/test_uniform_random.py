import pytest
from numpy.random import PCG64, Generator
from numpy.testing import assert_array_equal

from src.policies.uniform_random import UniformRandom


class TestUniformRandom:
    def test_select(self):
        uniform = UniformRandom(num_actions=2, horizon=1, rng=Generator(PCG64(40)))
        assert uniform.select(t=0) == 1

    def test_update(self):
        uniform = UniformRandom(num_actions=2, horizon=1, rng=Generator(PCG64(40)))
        assert uniform.select(t=0) == 1
        with pytest.raises(ValueError, match=r"Expected the reward for action 1, but got for 0"):
            uniform.update(t=0, action=0, reward=0.0, phi=1.0)
        uniform.update(t=0, action=1, reward=1.0, phi=1.0)
        assert_array_equal(uniform.rewards, [1.0])
        assert_array_equal(uniform.selected_actions, [1])
