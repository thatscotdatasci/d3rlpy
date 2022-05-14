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
mopo_policy_model_path = os.path.join(mopo_policy_dir, f"model_{MOPO_POLICY_TIMESTAMP}.pt")

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
