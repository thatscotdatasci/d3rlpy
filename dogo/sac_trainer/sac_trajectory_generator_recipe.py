import os
from datetime import datetime
from collections import namedtuple

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

RecipeIngredient = namedtuple('RecipeIngredient', 'policy_path n_episodes episode_length')
RECIPE = [
    RecipeIngredient("/home/ajc348/rds/hpc-work/d3rlpy/models/sac/HalfCheetah-v2/2022.05.10-18:13:40/SAC_online_20220510181343/model_25000.pt", 26,  1000),
    RecipeIngredient("/home/ajc348/rds/hpc-work/d3rlpy/models/sac/HalfCheetah-v2/2022.05.10-18:13:40/SAC_online_20220510181343/model_50000.pt", 25,  1000),
    RecipeIngredient("/home/ajc348/rds/hpc-work/d3rlpy/models/sac/HalfCheetah-v2/2022.05.10-18:13:40/SAC_online_20220510181343/model_75000.pt", 25,  1000),
    RecipeIngredient("/home/ajc348/rds/hpc-work/d3rlpy/models/sac/HalfCheetah-v2/2022.05.10-18:13:40/SAC_online_20220510181343/model_100000.pt", 25, 1000),
]

###############
# Derived Paths
###############

cur_timestamp = get_current_timestamp_str()
dataset_dir = os.path.join(
    DATASET_BASEDIR,
    'sac',
    ENV,
    f"sac_{SAC_POLICY_TIMESTAMP}",
    'D3RLPY-PEP3'
)
dataset_path = os.path.join(dataset_dir, f"data_{cur_timestamp}.h5")

# Create results directory
if not os.path.isdir(dataset_dir):
    os.makedirs(dataset_dir)
else:
    raise FileExistsError('Delete dest dir before proceeding')

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

# Initialise the final arrays
final_dataset = None
final_dataset_arr = None

# Loop through the recipe
for i, ing in enumerate(RECIPE):
    # Load model
    sac.load_model(ing.policy_path)

    for e in range(ing.n_episodes):
        # prepare experience replay buffer
        buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=ing.episode_length, env=env)

        # start data collection
        sac.collect(env, buffer, n_steps=ing.episode_length+3)

        # export as MDPDataset
        dataset = buffer.to_mdp_dataset()

        # extract the individual features from the dataset
        observations = dataset.observations[:ing.episode_length,:]
        actions = dataset.actions[:ing.episode_length,:]
        next_observations = dataset.observations[1:ing.episode_length+1,:]
        rewards = dataset.rewards[:ing.episode_length][:,None]
        terminals = dataset.terminals[:ing.episode_length][:,None]
        # policies = np.vstack((np.zeros((int(ing.episode_length/2),1)), np.ones((int(ing.episode_length/2),1))))
        policies = np.full((ing.episode_length,1), i)

        # create and save a numpy array
        dataset_arr = np.hstack((observations, actions, next_observations, rewards, terminals, policies))
        assert dataset_arr.shape[0] == ing.episode_length
        np.save(os.path.join(dataset_dir, f'rollout_{i}_{ing.episode_length}_{e}.npy'), dataset_arr)

        # create or update the final dataset and array
        if final_dataset is None:
            final_dataset = dataset
            final_dataset_arr = dataset_arr
        else:
            final_dataset.extend(dataset)
            final_dataset_arr = np.vstack((final_dataset_arr, dataset_arr))

# check the datasets
assert final_dataset_arr.shape[0] == sum([ing.n_episodes*ing.episode_length for ing in RECIPE])

# save MDPDataset
dataset.dump(dataset_path)

# save final array
np.save(os.path.join(dataset_dir, f'combined_data.npy'), final_dataset_arr)
