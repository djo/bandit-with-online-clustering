from numpy.random import default_rng, Generator, PCG64

from src.environments.bernoulli_bandit import BernoulliBandit


class TestBernoulliBandit:
    def test_draw(self):
        with_certainty = BernoulliBandit(means=[0.0, 1.0], rng=default_rng())
        assert with_certainty.draw(action=0) == 0
        assert with_certainty.draw(action=1) == 1

        b42 = BernoulliBandit(means=[0.5], rng=Generator(PCG64(42)))
        assert b42.draw(0) == 1
        assert b42.draw(0) == 0
