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
from dogo.environments.wrappers import HalfCheetahObsNoise

##########
# Settings
##########

SEED = None 
NOISE_STD = 0.001

USE_GPU = torch.cuda.is_available()

ENV = "HalfCheetah-v2"

EPOCH_LENGTH = 20000
N_EPOCHS = 150


def main(seed: int = SEED, noise_std: float = NOISE_STD):
    ###############
    # Derived Paths
    ###############

    # Path that the results will be saved to
    sac_policy_dir_gen = lambda ts: os.path.join(MODELS_BASEDIR, 'sac', f'{ENV}-Noise', ts)
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

    # Record the noise std being used
    with open(os.path.join(sac_policy_dir, 'noise_std.txt'), 'w') as f:
        f.write(str(noise_std))

    # Record the seed being used
    with open(os.path.join(sac_policy_dir, 'seed.txt'), 'w') as f:
        f.write(str(seed))

    ######################
    # Load the Environment
    ######################

    env = HalfCheetahObsNoise(gym.make(ENV), noise_std=noise_std)
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
        experiment_name=f"SAC_{seed}_{noise_std}",
        logdir=sac_policy_dir,
    )

    ################
    # Save the Model
    ################
    sac.save_model(sac_policy_model_path)


if __name__ == '__main__':
    seed = int(sys.argv[1])
    noise_std = float(sys.argv[2])
    main(seed=seed, noise_std=noise_std)
