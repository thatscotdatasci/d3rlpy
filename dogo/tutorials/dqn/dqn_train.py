import torch
# Dataset
from d3rlpy.datasets import get_cartpole 
# Algorithm
from d3rlpy.algos import DQN
# Metrics
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment

from sklearn.model_selection import train_test_split

########################################################################
# Getting started tutorial:
# https://d3rlpy.readthedocs.io/en/v1.1.0/tutorials/getting_started.html
########################################################################

print(f"GPU available: {torch.cuda.is_available()}")

# Here, we use the CartPole dataset to instantly check training results.
dataset, env = get_cartpole()
train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)
dqn = DQN(use_gpu=torch.cuda.is_available())

# initialize neural networks with the given observation shape and action size.
# this is not necessary when you directly call fit or fit_online method.
dqn.build_with_dataset(dataset)

# calculate metrics with test dataset
td_error = td_error_scorer(dqn, test_episodes)

# set environment in scorer function
evaluate_scorer = evaluate_on_environment(env)

# evaluate algorithm on the environment
rewards = evaluate_scorer(dqn)

# start training
dqn.fit(train_episodes,
    eval_episodes=test_episodes,
    n_epochs=10,
    scorers={
        'td_error': td_error_scorer,
        'value_scale': average_value_estimation_scorer,
        'environment': evaluate_scorer
    }
)

# make decisions
observation = env.reset()

# return actions based on the greedy-policy
action = dqn.predict([observation])[0]

# estimate action-values
value = dqn.predict_value([observation], [action])[0]
