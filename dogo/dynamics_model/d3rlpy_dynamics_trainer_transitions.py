import os
import argparse

import torch
import d3rlpy
from d3rlpy.datasets import MDPDataset
from d3rlpy.metrics import dynamics_observation_prediction_error_scorer
from d3rlpy.metrics import dynamics_reward_prediction_error_scorer

from dogo.paths import (
    MODELS_BASEDIR, DATASET_BASEDIR, DYNAMICS_MODEL_DIR
)
from dogo.utils.datetime import get_current_timestamp_str
from dogo.dataset.transitions import get_transitions_from_numpy_arr, get_episodes_from_numpy_arrs


##########
# Settings
##########

SEED = None 
USE_GPU = torch.cuda.is_available()

ENV = "HalfCheetah-v2"
POLICY_ALGORITHM = "SAC-PAP1"
DATASET_NAME = "SAC-PAP1"
DATATSET_NUMPY_ARR_PATH = "/home/ajc348/rds/hpc-work/d3rlpy/datasets/sac/HalfCheetah-v2/SAC-PAP1.npy"
TEST_EPISODES_NUMPY_ARRS_PATH_MASK = "/home/ajc348/rds/hpc-work/d3rlpy/datasets/sac/HalfCheetah-v2/sac_2022.05.10-18:13:40/D3RLPY-PEP1-TEST/rollout_1000_*.npy"
POLICY_TIMESTAMP = "2022.05.10-18:13:40"
DATASET_TIMESTAMP = "2022.05.19-11:30:00"

###############
# Derived Paths
###############

# path for the dynamics model logs and final model
cur_timestamp = get_current_timestamp_str()
dynamics_model_dir = os.path.join(
    MODELS_BASEDIR,
    DYNAMICS_MODEL_DIR,
    POLICY_ALGORITHM,
    ENV,
    f"{POLICY_ALGORITHM}_{POLICY_TIMESTAMP}",
    f"data_{DATASET_TIMESTAMP}",
    cur_timestamp
)
dynamics_model_path = os.path.join(dynamics_model_dir, f"model_{cur_timestamp}.pt")

# Create results directory
if os.path.isdir(dynamics_model_dir):
    raise FileExistsError('Target directory already exists')
else:
    os.makedirs(dynamics_model_dir)

#############################################
# Training according to d3rlpy example script
# Modified to use training trajectories
#############################################

# Load dataset
transitions = get_transitions_from_numpy_arr(DATATSET_NUMPY_ARR_PATH)

# Load test episodes
test_episodes = get_episodes_from_numpy_arrs(TEST_EPISODES_NUMPY_ARRS_PATH_MASK)

# fix seed
if SEED:
    d3rlpy.seed(SEED)

# TODO: Need to have test_episodes for evaluation
# train_transitions, test_transitions = train_test_split(transitions, test_size=0.2)

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
    use_gpu=USE_GPU,
)

# train dynamics model
dynamics.fit(
    transitions,
    eval_episodes=test_episodes,
    n_steps=100000,
    n_steps_per_epoch=10000,
    scorers={
        "obs_error": dynamics_observation_prediction_error_scorer,
        "rew_error": dynamics_reward_prediction_error_scorer,
    },
    experiment_name=f"Dynamics_{SEED}",
    logdir=dynamics_model_dir
)

# save the dynamics model
dynamics.save_model(dynamics_model_path)

