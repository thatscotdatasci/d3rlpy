import gym
import torch
from d3rlpy.datasets import MDPDataset 
from d3rlpy.algos import DQN
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment

from sklearn.model_selection import train_test_split

########################################################################
# Getting started tutorial - but updated to train mujoco:
# https://d3rlpy.readthedocs.io/en/v1.1.0/tutorials/getting_started.html
########################################################################

print(f"GPU available: {torch.cuda.is_available()}")

env = gym.make("InvertedPendulum-v2")

MDPDataset.load()

dqn = DQN(use_gpu=torch.cuda.is_available())

dqn.build_with_env(env)

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
