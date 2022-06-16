from typing import Protocol


class Policy(Protocol):
    def select(self, t: int) -> int:
        """
        Select next action to play on the given round.

        :param t: current round [0, T), where T is the horizon
        :return: an action index to play [0, N), where N is the number of actions
        """
        ...

    def feed_reward(self, t: int, action: int, reward: float):
        """
        Feed the realised reward to the policy.

        :param t: a given round [0, T)
        :param action: an action index to play [0, N)
        :param reward: realised reward
        :return: nothing
        """
        ...
