import os

import numpy as np
import gym
from mujoco_py import MjSimState

########
# Config
########

ENVIRONMENT = "HalfCheetah-v2"
INPUT_DATA = f"/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1-P4_100000.npy"

STATE_DIMS = 17
ACTION_DIMS = 6

# Load gym environment
env = gym.make(ENVIRONMENT)

# Load the input dataset - this will provide starting locations
data = np.load(INPUT_DATA)
states = data[:,:STATE_DIMS]
actions = data[:,STATE_DIMS:STATE_DIMS+ACTION_DIMS]

# Reset to begin a new episode
env.reset()

for idx in range(1000):
    # Use the action to take a step
    # env.step(actions[idx,:])

    # # Setting the state
    qpos = np.hstack((np.zeros(1),states[idx,:8]))
    qvel = states[idx,8:]
    env.set_state(qpos=qpos, qvel=qvel)

    # Render
    env.render()
