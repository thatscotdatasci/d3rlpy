import sys
import os

import numpy as np
import gym
import d3rlpy
from d3rlpy.algos import SAC

from dogo.paths import (
    DATASET_BASEDIR
)

##########
# Settings
##########

SEED = 42 
N_POLICIES = 5
N_EPISODES = 125*N_POLICIES
EPISODE_LENGTH = 1000
SAVE_INDIVIDUAL = True

def main(dataset_name, env_name, sac_policy_model_path):
    ###############
    # Derived Paths
    ###############
    dataset_dir = os.path.join(
        DATASET_BASEDIR,
        'sac',
        env_name,
        dataset_name
    )

    # Create results directory
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    # Save the model path that was used to create the data
    with open(os.path.join(dataset_dir, 'sac_policy_model_path.txt'), 'w') as f:
        f.write(sac_policy_model_path)

    ##############################
    # Generate and Save Trajectory
    ##############################

    # Load environment
    env = gym.make(env_name)

    if SEED:
        d3rlpy.seed(SEED)
        env.seed(SEED)

    # Load algorithm
    sac = SAC()
    sac.build_with_env(env)

    # Load model
    sac.load_model(sac_policy_model_path)

    final_dataset_arr = {}
    for e in range(N_EPISODES):
        print(f'Episode: {e}')

        observations = []
        actions = []
        next_observations = []
        rewards = []
        terminals = []
        
        cur_obs = env.reset()
        for _ in range(EPISODE_LENGTH):
            action = sac.sample_action(cur_obs[None,:])
            next_obs, reward, done, _ = env.step(action)

            observations.append(cur_obs)
            actions.append(action)
            next_observations.append(next_obs)
            rewards.append(reward)
            terminals.append(done)

            # Render, if running on terminal
            # env.render()

            cur_obs = next_obs

        observations = np.vstack(observations)
        actions = np.vstack(actions)
        next_observations = np.vstack(next_observations)
        rewards = np.vstack(rewards)
        terminals = np.vstack(terminals)
        
        # Set the policy identifier
        policies = np.vstack([np.full((int(EPISODE_LENGTH/N_POLICIES),1), float(i)) for i in range(N_POLICIES)])

        # Create the numpy array
        dataset_arr = np.hstack((observations, actions, next_observations, rewards, terminals, policies))

        # Optionally select only a single policy
        if SAVE_INDIVIDUAL:
            for i in range(N_POLICIES):
                pol_dataset_arr = dataset_arr[np.squeeze(np.argwhere(dataset_arr[:,-1]==float(i))),:]
                assert pol_dataset_arr.shape[0] == int(EPISODE_LENGTH/N_POLICIES)
                # np.save(os.path.join(dataset_dir, f'rollout_{i}_{EPISODE_LENGTH}_{e}.npy'), pol_dataset_arr)

                if e == 0:
                    final_dataset_arr[i] = pol_dataset_arr
                else:
                    final_dataset_arr[i] = np.vstack((final_dataset_arr[i], pol_dataset_arr))
        else:
            assert dataset_arr.shape[0] == EPISODE_LENGTH
            # np.save(os.path.join(dataset_dir, f'rollout_{EPISODE_LENGTH}_{e}.npy'), dataset_arr)
            if e == 0:
                final_dataset_arr[0] = dataset_arr
            else:
                final_dataset_arr[0] = np.vstack((final_dataset_arr[0], dataset_arr))

    # Check and save the dataset
    if SAVE_INDIVIDUAL:
        for i in range(N_POLICIES):
            assert final_dataset_arr[i].shape[0] == N_EPISODES*int(EPISODE_LENGTH/N_POLICIES)
            np.save(os.path.join(dataset_dir, f'{dataset_name}_P{i}_{final_dataset_arr[i].shape[0]}.npy'), final_dataset_arr[i])
    else:
        assert final_dataset_arr[0].shape[0] == N_EPISODES*EPISODE_LENGTH
        np.save(os.path.join(dataset_dir, f'{dataset_name}_{final_dataset_arr[0].shape[0]}.npy'), final_dataset_arr[0])

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    env_name = sys.argv[2]
    sac_policy_model_path = sys.argv[3]
    main(dataset_name=dataset_name, env_name=env_name, sac_policy_model_path=sac_policy_model_path)
