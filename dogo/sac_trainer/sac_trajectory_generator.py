import os
from datetime import datetime

import numpy as np
import gym
import d3rlpy
from d3rlpy.algos import SAC
from d3rlpy.dataset import MDPDataset

from dogo.paths import (
    MODELS_BASEDIR, DATASET_BASEDIR
)
from dogo.utils.datetime import get_current_timestamp_str

##########
# Settings
##########

SEED = 42 

ENV = "HalfCheetah-v2"
SAC_POLICY_TIMESTAMP = "2022.05.10-18:13:40"
# SAC_POLICY_TIMESTAMP = "2022.05.10-18:13:41"
# SAC_POLICY_TIMESTAMP = "2022.05.10-18:13:42"

N_EPISODES = 500
EPISODE_LENGTH = 1000
N_POLICIES = 5
SAVE_POLICY = 3.

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
    'sac',
    ENV,
    f"sac_{SAC_POLICY_TIMESTAMP}",
    'D3RLPY-PAP5-P3_100000'
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

final_dataset = None
final_dataset_arr = None
for e in range(N_EPISODES):
    # prepare experience replay buffer
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=EPISODE_LENGTH, env=env)

    # start data collection
    sac.collect(env, buffer, n_steps=EPISODE_LENGTH+3)

    # export as MDPDataset
    dataset = buffer.to_mdp_dataset()

    # extract the individual features from the dataset
    observations = dataset.observations[:EPISODE_LENGTH,:]
    actions = dataset.actions[:EPISODE_LENGTH,:]
    next_observations = dataset.observations[1:EPISODE_LENGTH+1,:]
    rewards = dataset.rewards[:EPISODE_LENGTH][:,None]
    terminals = dataset.terminals[:EPISODE_LENGTH][:,None]
    
    # Split into two
    # policies = np.vstack((np.zeros((int(EPISODE_LENGTH/2),1)), np.ones((int(EPISODE_LENGTH/2),1))))

    # Split into five
    policies = np.vstack((
        np.full((int(EPISODE_LENGTH/5),1), 0.),
        np.full((int(EPISODE_LENGTH/5),1), 1.),
        np.full((int(EPISODE_LENGTH/5),1), 2.),
        np.full((int(EPISODE_LENGTH/5),1), 3.),
        np.full((int(EPISODE_LENGTH/5),1), 4.),
        ))

    # create and save a numpy array
    dataset_arr = np.hstack((observations, actions, next_observations, rewards, terminals, policies))

    # optionally select only a single policy
    if SAVE_POLICY is not None:
        dataset_arr = dataset_arr[np.squeeze(np.argwhere(dataset_arr[:,-1]==SAVE_POLICY)),:]
        assert dataset_arr.shape[0] == int(EPISODE_LENGTH/N_POLICIES)
    else:
        assert dataset_arr.shape[0] == EPISODE_LENGTH
    
    np.save(os.path.join(dataset_dir, f'rollout_{EPISODE_LENGTH}_{e}.npy'), dataset_arr)

    # create or update the final dataset and array
    if e == 0:
        final_dataset = dataset
        final_dataset_arr = dataset_arr
    else:
        final_dataset.extend(dataset)
        final_dataset_arr = np.vstack((final_dataset_arr, dataset_arr))

# check the datasets
if SAVE_POLICY is not None:
    assert final_dataset_arr.shape[0] == N_EPISODES*int(EPISODE_LENGTH/N_POLICIES)
else:
    assert final_dataset_arr.shape[0] == N_EPISODES*EPISODE_LENGTH

# save MDPDataset
dataset.dump(dataset_path)

# save final array
np.save(os.path.join(dataset_dir, f'combined_data.npy'), final_dataset_arr)
