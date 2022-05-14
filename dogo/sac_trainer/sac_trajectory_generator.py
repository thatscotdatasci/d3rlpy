import os
from datetime import datetime

import gym
import d3rlpy
from d3rlpy.algos import SAC

from dogo.paths import (
    MODELS_BASEDIR, DATASET_BASEDIR
)
from dogo.utils.datetime import get_current_timestamp_str

##########
# Settings
##########

SEED = None 

ENV = "HalfCheetah-v2"
# SAC_POLICY_TIMESTAMP = "2022.05.10-18:13:40"
SAC_POLICY_TIMESTAMP = "2022.05.10-18:13:41"
# MODEL_TIMESTAMP_DIR = "2022.05.10-18:13:42"

###############
# Derived Paths
###############

sac_policy_model_path = os.path.join(
    MODELS_BASEDIR,
    'sac',
    ENV,
    SAC_POLICY_TIMESTAMP,
    f"model_{SAC_POLICY_TIMESTAMP}.pt"
)

cur_timestamp = get_current_timestamp_str()
dataset_dir = os.path.join(
    DATASET_BASEDIR,
    'sac', ENV,
    f"sac_{SAC_POLICY_TIMESTAMP}"
)
dataset_path = os.path.join(dataset_dir, f"data_{cur_timestamp}.h5")

# Create results directory
if not os.path.isdir(dataset_dir):
    os.makedirs(dataset_dir)

##############################
# Generate and Save Trajectory
##############################

# Load environment
env = gym.make(ENV)

if SEED:
    d3rlpy.seed(SEED)
    env.seed(SEED)

# Load algorithm
sac = SAC()
sac.build_with_env(env)

# Load model
sac.load_model(sac_policy_model_path)

# prepare experience replay buffer
buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)

# start data collection
sac.collect(env, buffer, n_steps=100000)

# export as MDPDataset
dataset = buffer.to_mdp_dataset()

# save MDPDataset
dataset.dump(dataset_path)
