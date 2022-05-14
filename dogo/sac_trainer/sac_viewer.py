import os

import gym
from d3rlpy.algos import SAC

from dogo.paths import MODELS_BASEDIR


##########
# Settings
##########

ENV = "HalfCheetah-v2"
MODEL_TIMESTAMP_DIR = "2022.05.2022-18:13:40"
# MODEL_TIMESTAMP_DIR = "2022.05.2022-18:13:41"
# MODEL_TIMESTAMP_DIR = "2022.05.2022-18:13:42"
MODEL_FILENAME = "model.pt"


###########
# Execution
###########

# Load environment
env = gym.make(ENV)

# Load algorithm
sac = SAC()
sac.build_with_env(env)

# Load model
sac.load_model(os.path.join(MODELS_BASEDIR, 'sac', ENV, MODEL_TIMESTAMP_DIR, MODEL_FILENAME))

# Loop through however many episodes
for i_episode in range(1):
    observation = env.reset()
    for t in range(10000):

        # Cannot render from HPC terminal
        env.render()

        print(observation)
        
        action = sac.sample_action(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
