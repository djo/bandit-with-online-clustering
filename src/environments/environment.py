from typing import Protocol

import numpy as np


class Environment(Protocol):
    means: np.ndarray

    def draw(self, action: int) -> float:
        """
        Draw the reward from a given action.

        :param action: an action index
        :return: a realised reward
        """
        ...
