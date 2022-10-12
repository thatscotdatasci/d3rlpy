import os

import gym
from d3rlpy.algos import SAC

from dogo.paths import MODELS_BASEDIR


##########
# Settings
##########

SEED = None 

#*********
# Baslines
#*********

ENV = "HalfCheetah-v2"
ENV_FOLDER = ENV

# SAC_POLICY_TIMESTAMP = "2022.05.10-18:13:40"
# SAC_POLICY_d3rlpy_DIR = "SAC_online_20220510181343"
# SAC_POLICY_CHECKPOINT = "100000"

# SAC_POLICY_TIMESTAMP = "2022.05.10-18:13:41"
# SAC_POLICY_d3rlpy_DIR = "SAC_online_20220510181345"
# SAC_POLICY_CHECKPOINT = "100000"

# SAC_POLICY_TIMESTAMP = "2022.05.10-18:13:42"
# SAC_POLICY_d3rlpy_DIR = "SAC_online_20220510181344"
# SAC_POLICY_CHECKPOINT = "40000"

#***********
# Noisy Envs
#***********

# ENV = "HalfCheetah-v2"
# ENV_FOLDER = "HalfCheetah-v2-Noise"

# 1443 - 0.1
# SAC_POLICY_TIMESTAMP = "2022.06.28-19:38:29"
# SAC_POLICY_d3rlpy_DIR = "SAC_1443_0.1_20220628193830"
# SAC_POLICY_CHECKPOINT = "100000"

# 1443 - 0.01
# SAC_POLICY_TIMESTAMP = "2022.06.28-19:38:26"
# SAC_POLICY_d3rlpy_DIR = "SAC_1443_0.01_20220628193827"
# SAC_POLICY_CHECKPOINT = "100000"

# 1443 - 0.001
# SAC_POLICY_TIMESTAMP = "2022.06.28-19:38:36"
# SAC_POLICY_d3rlpy_DIR = "SAC_1443_0.001_20220628193838"
# SAC_POLICY_CHECKPOINT = "100000"

#********
# PO Envs
#********

# ENV = "HalfCheetah-v2"
# ENV_FOLDER = "HalfCheetah-v2-PO"

# 1443 - 3
# SAC_POLICY_TIMESTAMP = "2022.06.28-20:29:36"
# SAC_POLICY_d3rlpy_DIR = "SAC_1443_3_20220628202937"
# SAC_POLICY_CHECKPOINT = "100000"

# 1443 - 12
# SAC_POLICY_TIMESTAMP = "2022.06.28-20:40:18"
# SAC_POLICY_d3rlpy_DIR = "SAC_1443_12_20220628204019"
# SAC_POLICY_CHECKPOINT = "100000"

# 1443 - 9,10
# SAC_POLICY_TIMESTAMP = "2022.06.28-20:29:27"
# SAC_POLICY_d3rlpy_DIR = "SAC_1443_9,10_20220628202929"
# SAC_POLICY_CHECKPOINT = "100000"

# 1443 - 3,16
# SAC_POLICY_TIMESTAMP = "2022.06.28-20:29:10"
# SAC_POLICY_d3rlpy_DIR = "SAC_1443_3,16_20220628202911"
# SAC_POLICY_CHECKPOINT = "100000"

#*******
# Hopper
#*******

ENV = "Hopper-v2"
ENV_FOLDER = ENV

SAC_POLICY_TIMESTAMP = "2022.08.31-14:32:54"
SAC_POLICY_d3rlpy_DIR = "SAC_1234_20220831143300"
SAC_POLICY_CHECKPOINT = "200000"

# SAC_POLICY_TIMESTAMP = "2022.08.31-14:32:55"
# SAC_POLICY_d3rlpy_DIR = "SAC_1443_20220831143300"
# SAC_POLICY_CHECKPOINT = "1000000"

# SAC_POLICY_TIMESTAMP = "2022.08.31-14:34:23"
# SAC_POLICY_d3rlpy_DIR = "SAC_4321_20220831143424"
# SAC_POLICY_CHECKPOINT = "1000000"

#*********
# Walker2d
#*********

# ENV = "Walker2d-v2"
# ENV_FOLDER = ENV

# SAC_POLICY_TIMESTAMP = "2022.08.31-15:46:49"
# SAC_POLICY_d3rlpy_DIR = "SAC_1443_20220831154653"
# SAC_POLICY_CHECKPOINT = "100000"

# SAC_POLICY_TIMESTAMP = "2022.08.31-15:47:43"
# SAC_POLICY_d3rlpy_DIR = "SAC_4321_20220831154743"
# SAC_POLICY_CHECKPOINT = "3000000"

# SAC_POLICY_TIMESTAMP = "2022.08.31-15:50:29"
# SAC_POLICY_d3rlpy_DIR = "SAC_1234_20220831155031"
# SAC_POLICY_CHECKPOINT = "3000000"


###############
# Derived Paths
###############

sac_policy_model_path = os.path.join(
    MODELS_BASEDIR,
    "sac",
    ENV_FOLDER,
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

        # Cannot render from standard HPC terminal
        # Comment out if not running in a VNC session
        env.render()

        # print(observation)
        
        # The sample_action methods requires a batch dimension
        # In this case we are only passing in a single observation
        action = sac.sample_action(observation[None,:])
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
