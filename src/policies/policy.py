from typing import Protocol

import numpy as np


class Policy(Protocol):
    num_actions: int
    cumulative_rewards: np.ndarray
    action_stats: np.ndarray

    def select(self, t: int) -> int:
        """
        Select next action to play on the given round.

        :param t: current round [0, T), where T is the horizon
        :return: an action index to play [0, N), where N is the number of actions
        """
        ...

    def update(self, t: int, action: int, reward: float, phi: float):
        """
        Feed the realised reward to the policy.

        :param t: a given round [0, T)
        :param action: an action index to play [0, N)
        :param reward: realised reward
        :param phi: discounting factor for the observed reward
        :return: nothing
        """
        ...
