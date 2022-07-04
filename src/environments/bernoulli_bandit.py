import numpy as np
from numpy.random import Generator

from src.environments.non_stationary_stochastic_environment import (
    NonStationaryStochasticEnvironment,
)


class BernoulliBandit(NonStationaryStochasticEnvironment):
    def __init__(self, init_means: np.ndarray, horizon: int, rng: Generator):
        self._rng = rng
        self.largest_means = np.zeros(horizon, dtype=float)
        self.optimal_actions = np.full(horizon, fill_value=-1, dtype=int)
        self.means = init_means

    def draw(self, t: int, action: int) -> float:
        self.optimal_actions[t] = np.argmax(self.means)
        self.largest_means[t] = self.means[self.optimal_actions[t]]
        p = self.means[action]
        return self._rng.binomial(n=1, p=p)
