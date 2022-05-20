import os

import gym
from d3rlpy.algos import MOPO

from dogo.paths import MODELS_BASEDIR


##########
# Settings
##########

SEED = None 

ENV = "HalfCheetah-v2"
POLICY_ALGORITHM = "sac"
POLICY_TIMESTAMP = "2022.05.10-18:13:40"
DATASET_TIMESTAMP = "2022.05.11-11:17:04"
DYNAMICS_MODEL_TIMESTAMP = "2022.05.12-17:56:07"
MOPO_POLICY_TIMESTAMP = "2022.05.14-12:36:43"
MOPO_POLICY_d3rlpy_DIR = "MOPO_None_20220514123645"
MOPO_POLICY_CHECKPOINT = "80000"

# ENV = "HalfCheetah-v2"
# POLICY_ALGORITHM = "sac"
# POLICY_TIMESTAMP = "2022.05.10-18:13:40"
# DATASET_TIMESTAMP = "2022.05.11-11:17:04"
# DYNAMICS_MODEL_TIMESTAMP = "2022.05.12-17:56:07"
# MOPO_POLICY_TIMESTAMP = "2022.05.14-13:44:51"
# MOPO_POLICY_d3rlpy_DIR = "MOPO_None_20220514134455"
# MOPO_POLICY_CHECKPOINT = "80000"

# ENV = "HalfCheetah-v2"
# POLICY_ALGORITHM = "SAC-PAP1"
# POLICY_TIMESTAMP = "2022.05.10-18:13:40"
# DATASET_TIMESTAMP = "2022.05.19-11:30:00"
# DYNAMICS_MODEL_TIMESTAMP = "2022.05.19-14:34:34"
# MOPO_POLICY_TIMESTAMP = "2022.05.19-15:24:10"
# MOPO_POLICY_d3rlpy_DIR = "MOPO_None_20220519152413"
# MOPO_POLICY_CHECKPOINT = "80000"


###############
# Derived Paths
###############

mopo_policy_dir = os.path.join(
    MODELS_BASEDIR,
    'mopo',
    ENV,
    f"{POLICY_ALGORITHM}_{POLICY_TIMESTAMP}",
    f'dynamics_{DYNAMICS_MODEL_TIMESTAMP}',
    f"data_{DATASET_TIMESTAMP}",
    MOPO_POLICY_TIMESTAMP
)
mopo_policy_model_path = os.path.join(mopo_policy_dir, MOPO_POLICY_d3rlpy_DIR, f"model_{MOPO_POLICY_CHECKPOINT}.pt")

###########
# Execution
###########

# Load environment
env = gym.make(ENV)

if SEED:
    env.seed(SEED)

# Load algorithm
mopo = MOPO()
mopo.build_with_env(env)

# Load model
mopo.load_model(mopo_policy_model_path)

# Loop through however many episodes
for i_episode in range(1):
    observation = env.reset()
    for t in range(10000):

        # Cannot render from HPC terminal
        env.render()

        print(observation)
        
        # The sample_action methods requires a batch dimension
        # In this case we are only passing in a single observation
        action = mopo.sample_action(observation[None,:])
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
