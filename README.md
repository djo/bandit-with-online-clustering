# Contextual multi-armed bandit problem with online clustering

![Build](https://github.com/djo/bandit-with-online-clustering/workflows/Python%20application/badge.svg)

Provides the framework for numerical experiments to simulate the contextual multi-armed bandit problem
in the environment with online clustering.

The context comes from the stream of data points observed in a sequence in real-time. We consider the case with good clustering of these data points (see cluster validation statistics, the goodness of clustering). Rewards on each step are dependent on a context (cluster), we consider iid assumption. As new points arrive (representing a new data or an updated state), the cluster centers drift, and so do the rewards with them. As usual, the objective is to maximize the cumulative total reward.

### Development

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
./pychecks.sh
```

MIT License

Copyright (c) 2022 Andrii Dzhoha
