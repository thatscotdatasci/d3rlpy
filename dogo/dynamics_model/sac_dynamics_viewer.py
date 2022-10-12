import os

import gym
from d3rlpy.algos import SAC
from d3rlpy.dynamics import ProbabilisticEnsembleDynamics

from dogo.paths import (
    MODELS_BASEDIR,
    DYNAMICS_MODEL_DIR,
    SAVED_MODEL_FILENAME,
    SAVED_MODEL_PARAMTERS_FILENAME
)


##########
# Settings
##########

ALGORITHM = "sac"
ENV = "HalfCheetah-v2"
POLICY_TIMESTAMP_DIR = "2022.05.10-18:13:40"
# MODEL_TIMESTAMP_DIR = "2022.05.10-18:13:41"
# MODEL_TIMESTAMP_DIR = "2022.05.10-18:13:42"
DATASET_TIMESTAMP = "2022.05.11-11:17:04"
DYNAMICS_MODEL_TIMESTAMP_DIR = "2022.05.12-17:56:07"
DYNAMICS_MODEL_d3rlpy_DIR = "ProbabilisticEnsembleDynamics_20220512175607"

DYNAMICS_MODEL_BASE_DIR = os.path.join(MODELS_BASEDIR, DYNAMICS_MODEL_DIR, ALGORITHM, ENV, DATASET_TIMESTAMP, DYNAMICS_MODEL_TIMESTAMP_DIR)
DYNAMICS_MODEL_FILE = os.path.join(DYNAMICS_MODEL_BASE_DIR, SAVED_MODEL_FILENAME)
DYNAMICS_MODEL_PARAMS_PATH = os.path.join(DYNAMICS_MODEL_BASE_DIR, DYNAMICS_MODEL_d3rlpy_DIR, SAVED_MODEL_PARAMTERS_FILENAME)

###########
# Execution
###########

# Load environment
# Note that this is only being loaded to instantiate the SAC algorithm
# TODO: Is there a way this can be avoided? Does it particularly matter?
env = gym.make(ENV)

# Load algorithm
sac = SAC()
sac.build_with_env(env)

# Load model
sac.load_model(os.path.join(MODELS_BASEDIR, 'sac', ENV, POLICY_TIMESTAMP_DIR, SAVED_MODEL_FILENAME))

# Load the dynamics model
dynamics = ProbabilisticEnsembleDynamics.from_json(DYNAMICS_MODEL_PARAMS_PATH)
dynamics.load_model(DYNAMICS_MODEL_FILE)

# Loop through however many episodes
for i_episode in range(1):
    observation = env.reset()
    for t in range(10000):

        # Cannot render from HPC terminal
        env.render()

        print(observation)
        
        # The sample_action methods requires a batch dimension
        # In this case we are only passing in a single observation
        action = sac.sample_action(observation[None,:])
        observation, reward = dynamics(x=observation, )

env.close()
