from typing import List

import numpy as np
from numpy.random import Generator

from src.environments.non_stationary_stochastic_environment import (
    NonStationaryStochasticEnvironment,
)


class BernoulliBandit(NonStationaryStochasticEnvironment):
    def __init__(self, means: List[float], horizon: int, rng: Generator):
        self._rng = rng
        self.largest_means = np.zeros(horizon, dtype=float)
        self.suboptimality = np.ones(horizon, dtype=float)
        self.optimal_actions = np.full(horizon, fill_value=-1, dtype=int)
        self.means = np.array(means)
        if np.any((self.means < 0) | (self.means > 1)):
            raise ValueError(f"Means {means} must be in [0, 1]")

    def draw(self, t: int, action: int) -> float:
        self.optimal_actions[t] = np.argmax(self.means)
        self.largest_means[t] = self.means[self.optimal_actions[t]]
        self.suboptimality[t] = self.largest_means[t] - self.means[action]

        p = self.means[action]
        return self._rng.binomial(n=1, p=p)
