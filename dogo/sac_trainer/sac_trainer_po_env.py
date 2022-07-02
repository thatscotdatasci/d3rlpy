import sys
import os
import time
from datetime import datetime

import gym
import torch
import numpy as np
import d3rlpy
from d3rlpy.algos import SAC

from dogo.paths import MODELS_BASEDIR
from dogo.utils.datetime import get_current_timestamp_str
from dogo.environments.wrappers import HalfCheetahPO

##########
# Settings
##########

SEED = None 

USE_GPU = torch.cuda.is_available()

ENV = "HalfCheetah-v2"
STATE_DIMS = 17

EPOCH_LENGTH = 20000
N_EPOCHS = 150


def main(masked_indices_str: str, seed: int = SEED):
    # ####################################
    # # Decide on the indices to be masked
    # ####################################
    # state_space_idxs = list(range(STATE_DIMS))
    # np.random.shuffle(state_space_idxs)
    # masked_indices = state_space_idxs[:num_masked_indices]
    # masked_indices_str = ','.join([str(i) for i in masked_indices])

    masked_indices = [int(i) for i in masked_indices_str.split(',')]

    ###############
    # Derived Paths
    ###############

    # Path that the results will be saved to
    sac_policy_dir_gen = lambda ts: os.path.join(MODELS_BASEDIR, 'sac', f'{ENV}-PO', ts)
    cur_timestamp = get_current_timestamp_str()
    sac_policy_dir = sac_policy_dir_gen(cur_timestamp)

    # Create results directory
    while os.path.isdir(sac_policy_dir):
        time.sleep(np.random.rand())
        cur_timestamp = get_current_timestamp_str()
        sac_policy_dir = sac_policy_dir_gen(cur_timestamp)
    os.mkdir(sac_policy_dir)

    sac_policy_model_path = os.path.join(
        sac_policy_dir,
        f"model_{cur_timestamp}.pt"
    )

    # Record the maked indices being used
    with open(os.path.join(sac_policy_dir, 'masked_indices.txt'), 'w') as f:
        f.write(masked_indices_str)

    # Record the seed being used
    with open(os.path.join(sac_policy_dir, 'seed.txt'), 'w') as f:
        f.write(str(seed))

    ######################
    # Load the Environment
    ######################

    env = HalfCheetahPO(gym.make(ENV), masked_indices=masked_indices)
    eval_env = gym.make(ENV)

    ######################
    # Set Environment Seed
    ######################

    if SEED:
        d3rlpy.seed(seed)
        env.seed(seed)
        eval_env.seed(seed)

    ###########################
    # Instantiate the Algorithm
    ###########################

    sac = SAC(
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=3e-4,
        use_gpu=USE_GPU,
    )

    ##########################
    # Define the replay buffer
    ##########################
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=EPOCH_LENGTH*N_EPOCHS, env=env)

    #######
    # Train
    #######
    sac.fit_online(
        env,
        buffer,
        eval_env=eval_env,
        n_steps=EPOCH_LENGTH*N_EPOCHS,
        n_steps_per_epoch=EPOCH_LENGTH,
        update_interval=1,
        update_start_step=10000,
        experiment_name=f"SAC_{seed}_{masked_indices_str}",
        logdir=sac_policy_dir,
    )

    ################
    # Save the Model
    ################
    sac.save_model(sac_policy_model_path)


if __name__ == '__main__':
    # Extract from command line arguments
    seed = int(sys.argv[1])
    masked_indices_str = sys.argv[2]

    # For use when debugging
    # seed = 1443
    # masked_indices_str = '1,2'

    main(seed=seed, masked_indices_str=masked_indices_str)
