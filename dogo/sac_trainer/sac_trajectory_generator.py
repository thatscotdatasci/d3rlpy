import os
from datetime import datetime

import gym
import d3rlpy
from d3rlpy.algos import SAC

from dogo.paths import MODELS_BASEDIR, DATASET_BASEDIR

##########
# Settings
##########

ENV = "HalfCheetah-v2"
# MODEL_TIMESTAMP_DIR = "2022.05.2022-18:13:40"
MODEL_TIMESTAMP_DIR = "2022.05.2022-18:13:41"
# MODEL_TIMESTAMP_DIR = "2022.05.2022-18:13:42"
MODEL_FILENAME = "model.pt"

##################
# Save the Dataset
##################

# Load environment
env = gym.make(ENV)

# Load algorithm
sac = SAC()
sac.build_with_env(env)

# Load model
sac.load_model(os.path.join(MODELS_BASEDIR, 'sac', ENV, MODEL_TIMESTAMP_DIR, MODEL_FILENAME))

# prepare experience replay buffer
buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)

# start data collection
sac.collect(env, buffer, n_steps=100000)

# export as MDPDataset
dataset = buffer.to_mdp_dataset()

# save MDPDataset
dataset_dir = os.path.join(DATASET_BASEDIR, 'sac', ENV, MODEL_TIMESTAMP_DIR)
if not os.path.isdir(dataset_dir):
    os.makedirs(dataset_dir)
dataset.dump(os.path.join(dataset_dir, f'{datetime.now().strftime("%Y.%m.%Y-%H:%M:%S")}.h5'))
