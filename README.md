# Contextual multi-armed bandit problem with online clustering

![Build](https://github.com/djo/bandit-with-online-clustering/workflows/Python%20application/badge.svg)

The framework for numerical experiments to simulate the contextual multi-armed bandit problem
in the environment with online clustering.
Part of the paper [Multi-armed bandit problem with online clustering as side information](https://wiki.helsinki.fi/download/attachments/406850783/dzhoha-abstract.pdf).

Structure of the project and currently implemented algorithms:

||Files|
|-|-|
|Environments|[Protocol](src/environments/non_stationary_stochastic_environment.py)|
||[Bernoulli MAB](src/environments/bernoulli_bandit.py)|
|Policies|[Protocol](src/policies/policy.py)|
||[Uniform Random](src/policies/uniform_random.py)|
||[Discounted Thompson Sampling (Beta distribution)](src/policies/discounted_beta_thompson_sampling.py)|
|Tests|[Test module](src/test/)|

### Development

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
./pychecks.sh
```

MIT License

Copyright (c) 2022 Andrii Dzhoha
