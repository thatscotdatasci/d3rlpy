import os

import gym
from d3rlpy.algos import SAC

from dogo.paths import MODELS_BASEDIR


##########
# Settings
##########

SEED = None 

ENV = "HalfCheetah-v2"

SAC_POLICY_TIMESTAMP = "2022.05.10-18:13:40"
SAC_POLICY_d3rlpy_DIR = "SAC_online_20220510181343"
SAC_POLICY_CHECKPOINT = "100000"

# SAC_POLICY_TIMESTAMP = "2022.05.10-18:13:41"
# SAC_POLICY_d3rlpy_DIR = "SAC_online_20220510181345"
# SAC_POLICY_CHECKPOINT = "80000"

# SAC_POLICY_TIMESTAMP = "2022.05.10-18:13:42"
# SAC_POLICY_d3rlpy_DIR = "SAC_online_20220510181344"
# SAC_POLICY_CHECKPOINT = "80000"

###############
# Derived Paths
###############

sac_policy_model_path = os.path.join(
    MODELS_BASEDIR,
    "sac",
    ENV,
    SAC_POLICY_TIMESTAMP,
    SAC_POLICY_d3rlpy_DIR,
    f"model_{SAC_POLICY_CHECKPOINT}.pt"
)

###########
# Execution
###########

# Load environment
env = gym.make(ENV)
# env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True, force=True)

if SEED:
    env.seed(SEED)

# Load algorithm
sac = SAC()
sac.build_with_env(env)

# Load model
sac.load_model(sac_policy_model_path)

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
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
