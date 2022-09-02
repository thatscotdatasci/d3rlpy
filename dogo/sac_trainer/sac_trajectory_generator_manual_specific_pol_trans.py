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
N_TRANS = 20000
EPISODE_LENGTH = 1000

def main(dataset_name, env_name, sac_policy_model_path, policy_identifier):
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
    with open(os.path.join(dataset_dir, f'sac_policy_model_path_P{int(policy_identifier)}.txt'), 'w') as f:
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
    e_count = 0
    t_count = 0
    while t_count < N_TRANS:
        print(f'Episode: {e_count}')

        observations = []
        actions = []
        next_observations = []
        rewards = []
        terminals = []
        
        cur_obs = env.reset()
        for i in range(EPISODE_LENGTH):
            action = sac.sample_action(cur_obs[None,:])
            next_obs, reward, done, _ = env.step(action)

            observations.append(cur_obs)
            actions.append(action)
            next_observations.append(next_obs)
            rewards.append(reward)
            terminals.append(done)
            t_count += 1

            if done or t_count == N_TRANS:
                print(f'Finished at: {i}')
                break

            # Render, if running on terminal
            # env.render()

            cur_obs = next_obs

        observations = np.vstack(observations)
        actions = np.vstack(actions)
        next_observations = np.vstack(next_observations)
        rewards = np.vstack(rewards)
        terminals = np.vstack(terminals)
        
        # Set the policy identifier
        policies = np.full((len(observations),1), policy_identifier)

        # Create the numpy array
        dataset_arr = np.hstack((observations, actions, next_observations, rewards, terminals, policies))

        if e_count == 0:
            final_dataset_arr[0] = dataset_arr
        else:
            final_dataset_arr[0] = np.vstack((final_dataset_arr[0], dataset_arr))

        e_count += 1

    assert final_dataset_arr[0].shape[0] == N_TRANS
    np.save(os.path.join(dataset_dir, f'{dataset_name}-P{int(policy_identifier)}_{final_dataset_arr[0].shape[0]}.npy'), final_dataset_arr[0])

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    policy_identifier = float(sys.argv[2])
    env_name = sys.argv[3]
    sac_policy_model_path = sys.argv[4]
    main(dataset_name=dataset_name, env_name=env_name, sac_policy_model_path=sac_policy_model_path, policy_identifier=policy_identifier)
