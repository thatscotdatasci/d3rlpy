import os
import argparse

import gym
import torch
import d3rlpy
from d3rlpy.datasets import MDPDataset
from d3rlpy.metrics import dynamics_observation_prediction_error_scorer
from d3rlpy.metrics import dynamics_reward_prediction_error_scorer
from sklearn.model_selection import train_test_split

from dogo.paths import MODELS_BASEDIR, DATASET_BASEDIR, DYNAMICS_MODEL_DIR
from dogo.utils.datetime import get_current_timestamp_str


##########
# Settings
##########

SEED = 1
ALGORITHM = "sac"
ENV = "HalfCheetah-v2"
POLICY_TIMESTAMP_DIR = "2022.05.2022-18:13:40"
DATASET_TIMESTAMP = "2022.05.2022-11:17:04"
DATASET_TIMESTAMP_FILE = "{DATASET_TIMESTAMP}}.h5"
DATASET_PATH = os.path.join(DATASET_BASEDIR, ALGORITHM, ENV, POLICY_TIMESTAMP_DIR, DATASET_TIMESTAMP_FILE)


#############################################
# Training according to d3rlpy example script
# Modified to use training trajectories
#############################################

PARAMETER_TABLE = {
    'halfcheetah-random-v0': (5, 0.5),
    'hopper-random-v0': (5, 1),
    'walker2d-random-v0': (1, 1),
    'halfcheetah-medium-v0': (1, 1),
    'hopper-medium-v0': (5, 5),
    'walker2d-medium-v0': (5, 5),
    'halfcheetah-medium-replay-v0': (5, 1),
    'hopper-medium-replay-v0': (5, 1),
    'walker2d-medium-replay-v0': (1, 1),
    'halfcheetah-medium-expert-v0': (5, 1),
    'hopper-medium-expert-v0': (5, 1),
    'walker2d-medium-expert-v0': (1, 2)
}



# Load dataset
dataset = MDPDataset.load(DATASET_PATH)

# Load environment
env = gym.make(ENV)

# fix seed
d3rlpy.seed(SEED)
env.seed(SEED)

_, test_episodes = train_test_split(dataset, test_size=0.2)

# prepare dynamics model
dynamics_encoder = d3rlpy.models.encoders.VectorEncoderFactory(
    hidden_units=[200, 200, 200, 200],
    activation='swish',
)
dynamics_optim = d3rlpy.models.optimizers.AdamFactory(weight_decay=2.5e-5)
dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(
    encoder_factory=dynamics_encoder,
    optim_factory=dynamics_optim,
    learning_rate=1e-3,
    n_ensembles=5,
    use_gpu=torch.cuda.is_available(),
)

# path for the dynamics model logs and final model
dynamics_model_path = os.path.join(MODELS_BASEDIR, DYNAMICS_MODEL_DIR, ALGORITHM, ENV, DATASET_TIMESTAMP, get_current_timestamp_str())

# train dynamics model
dynamics.fit(
    dataset.episodes,
    eval_episodes=test_episodes,
    n_steps=100000,
    scorers={
        "obs_error": dynamics_observation_prediction_error_scorer,
        "rew_error": dynamics_reward_prediction_error_scorer,
    },
    logdir=dynamics_model_path
)

# save the dynamics model
dynamics.save_model(os.path.join(dynamics_model_path, "model.pt"))

