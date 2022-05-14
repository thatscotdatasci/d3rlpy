import os

import gym
import torch
import d3rlpy
from d3rlpy.datasets import MDPDataset
from d3rlpy.dynamics import ProbabilisticEnsembleDynamics
from sklearn.model_selection import train_test_split

from dogo.dynamics_model.dynamics_model import Gauss_Dynamics_Model, Trainer
from dogo.paths import (
    DATASET_BASEDIR,
    MODELS_BASEDIR,
    DYNAMICS_MODEL_DIR,
    SAVED_MODEL_FILENAME,
    SAVED_MODEL_PARAMTERS_FILENAME
)
from dogo.utils.datetime import get_current_timestamp_str
from dogo.utils.pytorch_setup import DEVICE


##########
# Settings
##########

SEED = 1
ALGORITHM = "sac"
ENV = "HalfCheetah-v2"
POLICY_TIMESTAMP_DIR = "2022.05.10-18:13:40"
DATASET_TIMESTAMP = "2022.05.11-11:17:04"
DATASET_TIMESTAMP_FILE = f"{DATASET_TIMESTAMP}.h5"
DYNAMICS_MODEL_TIMESTAMP_DIR = "2022.05.12-17:56:07"
DYNAMICS_MODEL_d3rlpy_DIR = "ProbabilisticEnsembleDynamics_20220512175607"

DATASET_PATH = os.path.join(DATASET_BASEDIR, ALGORITHM, ENV, POLICY_TIMESTAMP_DIR, DATASET_TIMESTAMP_FILE)

DYNAMICS_MODEL_BASE_DIR = os.path.join(MODELS_BASEDIR, DYNAMICS_MODEL_DIR, ALGORITHM, ENV, DATASET_TIMESTAMP, DYNAMICS_MODEL_TIMESTAMP_DIR)
DYNAMICS_MODEL_FILE = os.path.join(DYNAMICS_MODEL_BASE_DIR, SAVED_MODEL_FILENAME)
DYNAMICS_MODEL_PARAMS_PATH = os.path.join(DYNAMICS_MODEL_BASE_DIR, DYNAMICS_MODEL_d3rlpy_DIR, SAVED_MODEL_PARAMTERS_FILENAME)

# d3rlpy parameters
ROLLOUT_HORIZON, LAM = 5, 1


############
# Train MOPO
############

# Load dataset
dataset = MDPDataset.load(DATASET_PATH)

# Load environment
env = gym.make(ENV)

# fix seed
d3rlpy.seed(SEED)
env.seed(SEED)

# Load the dynamics model
dynamics = ProbabilisticEnsembleDynamics.from_json(DYNAMICS_MODEL_PARAMS_PATH)
dynamics.load_model(DYNAMICS_MODEL_FILE)

# prepare combo
mopo = d3rlpy.algos.MOPO(
    dynamics=dynamics,
    rollout_horizon=ROLLOUT_HORIZON,
    lam=LAM,
    use_gpu=torch.cuda.is_available()
)

_, test_episodes = train_test_split(dataset, test_size=0.2)

# train combo
mopo.fit(
    dataset.episodes,
    eval_episodes=test_episodes,
    n_steps=500000,
    n_steps_per_epoch=1000,
    save_interval=10,
    scorers={
        "environment": d3rlpy.metrics.evaluate_on_environment(env),
        'value_scale': d3rlpy.metrics.average_value_estimation_scorer
    },
    experiment_name=f"MOPO_{SEED}"
)
