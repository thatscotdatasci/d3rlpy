import os
import time
from datetime import datetime

import gym
import torch
import numpy as np
import d3rlpy
from d3rlpy.algos import SAC
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment

from dogo.paths import MODELS_BASEDIR

print(f"GPU available: {torch.cuda.is_available()}")

##########
# Settings
##########

ENV = "HalfCheetah-v2"
USE_GPU = torch.cuda.is_available()
SEED = None 

#####################################
# Create Directory to Save Results In
#####################################

model_dir_gen = lambda: os.path.join(MODELS_BASEDIR, 'sac', ENV, datetime.now().strftime("%Y.%m.%Y-%H:%M:%S"))
model_dir = model_dir_gen()
while os.path.isdir(model_dir):
    time.sleep(np.random.rand())
    model_dir = model_dir_gen()
os.mkdir(model_dir)

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
    logdir=model_dir,
)

################
# Save the Model
################
sac.save_model(os.path.join(model_dir, 'model.pt'))
