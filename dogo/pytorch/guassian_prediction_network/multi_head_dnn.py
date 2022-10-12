from tqdm import tqdm


from tqdm.auto import trange
import numpy as np
import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from dogo.utils.pytorch_setup import DEVICE

class DNN(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int
    ) -> None:
        super().__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims

        self.shared_l1 = nn.Linear(
            in_features=input_dims,
            out_features=hidden_dims,
        )

        self.mu_layer = nn.Linear(
            in_features=hidden_dims,
            out_features=input_dims,
        )

        self.log_sigma_layer = nn.Linear(
            in_features=hidden_dims,
            out_features=int(input_dims*(input_dims+1)/2),
        )

    def forward(self, X: Tensor):
        X = X.to(DEVICE)
        Y_1 = F.relu(self.shared_l1(X))
        
        mu = F.relu(self.mu_layer(Y_1))

        log_sigma = torch.sigmoid(self.log_sigma_layer(Y_1))
        sigma = torch.exp(log_sigma)
        cov_mat = torch.zeros(X.shape[0], self.input_dims, self.input_dims)
        i, j = torch.tril_indices(self.input_dims, self.input_dims)
        cov_mat[:, i, j] = sigma

        norm = MultivariateNormal(loc=mu, scale_tril=cov_mat)
        return norm.log_prob(X)

class Trainer:
    def __init__(self, model: nn.Module, n_steps: int) -> None:
        self.model = model
        self.n_steps = n_steps

        self.criterion = nn.MSELoss()
        self.optimiser = optim.Adam(
            params=self.model.parameters(),
            lr=1e-3
        )

    def train(self, X: Tensor, Y: Tensor):
        self.model.train()

        t_range = trange(self.n_steps)
        for i in t_range:
            self.optimiser.zero_grad()
            preds = self.model(X)
            loss = self.criterion(preds, Y)
            loss.backward()
            self.optimiser.step()
            t_range.set_postfix({'loss': loss.item()})

    def predict(self, X: Tensor):
        self.model.eval()
        return self.model(X)


# # Set the dimensions
# N = 1
# M = 3

# # Create the distribution
# mu = np.random.random(M)
# sigma = np.random.random(int((M*(M+1))/2))
# cov = np.zeros((M, M))
# cov[np.tril_indices(M)] = sigma
# cov.T[np.tril_indices(M)] = sigma

# X = np.random.multivariate_normal()
