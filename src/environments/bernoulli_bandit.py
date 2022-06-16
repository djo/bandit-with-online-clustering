from typing import List

import numpy as np
from numpy.random import Generator

from src.environments.environment import Environment


class BernoulliBandit(Environment):
    def __init__(self, means: List[float], rng: Generator):
        self._rng = rng
        self.means = np.array(means)
        if np.any((self.means < 0) | (self.means > 1)):
            raise ValueError(f"Means {means} must be in [0, 1]")

    def draw(self, action: int) -> float:
        p = self.means[action]
        return self._rng.binomial(n=1, p=p)
