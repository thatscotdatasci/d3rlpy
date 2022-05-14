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

##########
# Settings
##########

SEED = None 
USE_GPU = torch.cuda.is_available()

ENV = "HalfCheetah-v2"

###############
# Derived Paths
###############

# Path that the results will be saved to
sac_policy_dir_gen = lambda ts: os.path.join(MODELS_BASEDIR, 'sac', ENV, ts)
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

######################
# Load the Environment
######################

env = gym.make(ENV)
eval_env = gym.make(ENV)

######################
# Set Environment Seed
######################

if SEED:
    d3rlpy.seed(SEED)
    env.seed(SEED)
    eval_env.seed(SEED)

###########################
# Instantiate the Algorithm
###########################

sac = SAC(
    batch_size=256,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    temp_learning_rate=3e-4,
    use_gpu=USE_GPU,
)

##########################
# Define the replay buffer
##########################
buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)

#######
# Train
#######
sac.fit_online(
    env,
    buffer,
    eval_env=eval_env,
    n_steps=100000,
    n_steps_per_epoch=1000,
    update_interval=1,
    update_start_step=1000,
    experiment_name=f"SAC_{SEED}",
    logdir=sac_policy_dir,
)

################
# Save the Model
################
sac.save_model(sac_policy_model_path)
