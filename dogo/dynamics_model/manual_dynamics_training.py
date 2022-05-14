import os

import numpy as np
import torch
from torch import Tensor
from d3rlpy.dataset import MDPDataset

from dogo.dynamics_model.manual_dynamics_model import Gauss_Dynamics_Model, Trainer
from dogo.paths import MODELS_BASEDIR, DATASET_BASEDIR, DYNAMICS_MODEL_DIR
from dogo.utils.datetime import get_current_timestamp_str
from dogo.utils.pytorch_setup import DEVICE


##########
# Settings
##########

ALGORITHM = "sac"
ENV = "HalfCheetah-v2"
MODEL_TIMESTAMP_DIR = "2022.05.2022-18:13:40"
DATASET_TIMESTAMP_FILE = "2022.05.2022-11:17:04"
DATASET_PATH = os.path.join(DATASET_BASEDIR, ALGORITHM, ENV, MODEL_TIMESTAMP_DIR, DATASET_TIMESTAMP_FILE)


########################
# Simple manual training
########################

data = MDPDataset.load(DATASET_PATH)

X = Tensor(np.hstack((data.observations[:-1], data.actions[:-1])))
Y = Tensor(np.hstack((data.observations[1:,:], data.rewards[:-1][:, None])))

dynamics_model = Gauss_Dynamics_Model(
    input_dims=X.shape[1],
    output_dims=Y.shape[1],
    hidden_units=10,
)
dynamics_model.to(DEVICE)

trainer = Trainer(
    model=dynamics_model,
    n_steps=10000,
)

trainer.train(X,Y)

# path for the dynamics model logs and final model
dynamics_model_path = os.path.join(MODELS_BASEDIR, DYNAMICS_MODEL_DIR, 'manual', ALGORITHM, ENV, DATASET_TIMESTAMP_FILE, get_current_timestamp_str())

# make the directory
if not os.path.isdir(dynamics_model_path):
    os.makedirs(dynamics_model_path)

# save the dynamics mode
torch.save(trainer.model.state_dict(), os.path.join(dynamics_model_path, "model.pt"))
