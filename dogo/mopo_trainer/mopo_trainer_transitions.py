import os
from datetime import datetime

import gym
import torch
import d3rlpy
from d3rlpy.datasets import MDPDataset
from d3rlpy.dynamics import ProbabilisticEnsembleDynamics

from dogo.paths import (
    DATASET_BASEDIR,
    MODELS_BASEDIR,
    DYNAMICS_MODEL_DIR,
    SAVED_MODEL_PARAMTERS_FILENAME
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
DYNAMICS_MODEL_TIMESTAMP = "2022.05.19-14:34:34"
DYNAMICS_MODEL_d3rlpy_DIR = "Dynamics_None_20220519143435"

# d3rlpy parameters
ROLLOUT_HORIZON, LAM = 5, 1

###############
# Derived Paths
###############

dynamics_model_dir = os.path.join(
    MODELS_BASEDIR,
    DYNAMICS_MODEL_DIR,
    POLICY_ALGORITHM,
    ENV,
    f"{POLICY_ALGORITHM}_{POLICY_TIMESTAMP}",
    f"data_{DATASET_TIMESTAMP}",
    DYNAMICS_MODEL_TIMESTAMP
)
dynamics_model_path = os.path.join(dynamics_model_dir, f"model_{DYNAMICS_MODEL_TIMESTAMP}.pt")
dynamics_model_params_path = os.path.join(dynamics_model_dir, DYNAMICS_MODEL_d3rlpy_DIR, SAVED_MODEL_PARAMTERS_FILENAME)

cur_timestamp = get_current_timestamp_str()
mopo_policy_dir = os.path.join(
    MODELS_BASEDIR,
    'mopo',
    ENV,
    f"{POLICY_ALGORITHM}_{POLICY_TIMESTAMP}",
    f'dynamics_{DYNAMICS_MODEL_TIMESTAMP}',
    f"data_{DATASET_TIMESTAMP}",
    cur_timestamp
)
mopo_policy_model_path = os.path.join(mopo_policy_dir, f"model_{cur_timestamp}.pt")

# Create results directory
if os.path.isdir(mopo_policy_dir):
    raise FileExistsError('Target directory already exists')
else:
    os.makedirs(mopo_policy_dir)

############
# Train MOPO
############

# Load dataset
transitions = get_transitions_from_numpy_arr(DATATSET_NUMPY_ARR_PATH)

# Load test episodes
test_episodes = get_episodes_from_numpy_arrs(TEST_EPISODES_NUMPY_ARRS_PATH_MASK)

# Load environment
env = gym.make(ENV)

# fix seed
if SEED:
    d3rlpy.seed(SEED)
    env.seed(SEED)

# Load the dynamics model
dynamics = ProbabilisticEnsembleDynamics.from_json(dynamics_model_params_path)
dynamics.load_model(dynamics_model_path)

# prepare mopo
mopo = d3rlpy.algos.MOPO(
    dynamics=dynamics,
    rollout_horizon=ROLLOUT_HORIZON,
    lam=LAM,
    use_gpu=USE_GPU,
)

# train mopo
mopo.fit(
    transitions,
    eval_episodes=test_episodes,
    n_steps=500000,
    n_steps_per_epoch=1000,
    save_interval=10,
    scorers={
        "environment": d3rlpy.metrics.evaluate_on_environment(env),
        'value_scale': d3rlpy.metrics.average_value_estimation_scorer
    },
    experiment_name=f"MOPO_{SEED}",
    logdir=mopo_policy_dir
)

# Save the model
mopo.save_model(mopo_policy_model_path)
