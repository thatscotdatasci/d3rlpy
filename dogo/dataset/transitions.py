import os
from glob import glob
from typing import Sequence, List, Tuple

import numpy as np
from d3rlpy.datasets import Transition, Episode


def load_numpy_arr(nump_arr_path: str, dataset_dimensions: Sequence = (17, 6, 17, 1, 1, 1)) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    data = np.load(nump_arr_path)
    transition_elements = np.split(data, np.cumsum(dataset_dimensions), axis=1)[:-1]
    for i, e in enumerate(transition_elements):
        assert e.shape[1] == dataset_dimensions[i]
    observations, actions, next_states, rewards, dones, policies = transition_elements
    return observations, actions, next_states, rewards, dones, policies

def get_transitions_from_numpy_arr(nump_arr_path: str, dataset_dimensions: Sequence = (17, 6, 17, 1, 1, 1)) -> List[Transition]:
    if not os.path.isfile(nump_arr_path):
        raise FileNotFoundError('Input array does not exist.')

    observations, actions, next_states, rewards, dones, _ = load_numpy_arr(nump_arr_path, dataset_dimensions)

    obs_dim = (observations.shape[1],)
    action_dim = actions.shape[1]

    transitions = [
        Transition(
            observation_shape=obs_dim,
            action_size=action_dim,
            observation=observations[i,:],
            action=actions[i,:],
            next_observation=next_states[i,:],
            reward= rewards[i,:],
            terminal=dones[i,:],
        )
        for i in range(observations.shape[0])
    ]
    
    assert len(transitions) == observations.shape[0]
    return transitions
    
def get_episodes_from_numpy_arrs(nump_arr_path_mask: str, dataset_dimensions: Sequence = (17, 6, 17, 1, 1, 1)) -> List[Episode]:
    arr_paths = glob(nump_arr_path_mask)
    if len(arr_paths) == 0:
        raise FileNotFoundError('Did not find any Numpy arrays')
    
    episodes = []
    for arr_path in arr_paths:
        observations, actions, _, rewards, dones, _ = load_numpy_arr(arr_path, dataset_dimensions)

        obs_dim = (observations.shape[1],)
        action_dim = actions.shape[1]

        episode = Episode(
            observation_shape=obs_dim,
            action_size=action_dim,
            observations=observations,
            actions=actions,
            rewards=np.squeeze(rewards),
            terminal=True
        )

        episodes.append(episode)
    return episodes
