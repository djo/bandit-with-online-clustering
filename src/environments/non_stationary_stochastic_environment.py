from typing import Protocol

import numpy as np


class NonStationaryStochasticEnvironment(Protocol):
    # current means which can be updated directly on each step
    means: np.ndarray
    # largest mean on each step t
    largest_means: np.ndarray
    # an action with the largest mean on each step t
    optimal_actions: np.ndarray
    # suboptimality on each step t
    suboptimality: np.ndarray

    def draw(self, t: int, action: int) -> float:
        """
        Draw the reward from a given action.

        :param t: current round [0, T), where T is the horizon
        :param action: an action index
        :return: a realised reward
        """
        ...
