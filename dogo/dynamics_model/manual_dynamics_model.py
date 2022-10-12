from typing import Tuple

from tqdm.auto import trange
import torch
from torch import nn, Tensor, optim
from torch.functional import F
from torch.distributions import MultivariateNormal

from dogo.utils.pytorch_setup import DEVICE

#######################################################################################
# Custom dynamics model - based on MBPO, but with fewer layers
# The purpose of this was simply to ensure that custom models could be used with D3RLPY
#######################################################################################

class Gauss_Dynamics_Model(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_units: int,
        diagonal: bool = True,
    ) -> None:
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_units = hidden_units
        self.diagonal = diagonal

        self.shared_l1 = nn.Linear(
            in_features=self.input_dims,
            out_features=self.hidden_units
        )

        self.mu_l = nn.Linear(
            in_features=self.hidden_units,
            out_features=self.output_dims
        )

        if self.diagonal:
            sigma_1_dims = int(self.output_dims)
        else:
            sigma_1_dims = int(self.output_dims*(self.output_dims+1)/2)
        self.sigma_l = nn.Linear(
            in_features=self.hidden_units,
            out_features=sigma_1_dims,
        )


    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        X = X.to(DEVICE)
        Y_1 = F.relu(self.shared_l1(X))

        mu = F.relu(self.mu_l(Y_1))

        log_sigma = torch.sigmoid(self.sigma_l(Y_1))
        sigma = torch.exp(log_sigma)

        if self.diagonal:
            cov_mat = torch.diag_embed(sigma)
        else:
            cov_mat = torch.zeros(X.shape[0], self.output_dims, self.output_dims)
            i, j = torch.tril_indices(self.output_dims, self.output_dims)
            cov_mat[:, i, j] = sigma
        
        return mu, cov_mat


class Trainer:
    def __init__(self, model: nn.Module, n_steps: int) -> None:
        self.model = model
        self.n_steps = n_steps

        self.criterion = self.loss
        self.optimiser = optim.Adam(
            params=self.model.parameters(),
            lr=1e-3
        )

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        mu, cov_mat = output
        norm = MultivariateNormal(loc=mu, scale_tril=cov_mat)
        return -norm.log_prob(target).sum()

    def train(self, X: Tensor, Y: Tensor):
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)

        self.model.train()

        t_range = trange(self.n_steps)
        for _ in t_range:
            self.optimiser.zero_grad()
            mu, cov_mat = self.model(X)
            loss = self.criterion((mu, cov_mat), Y)
            loss.backward()
            self.optimiser.step()
            t_range.set_postfix({'loss': loss.item()})

    def sample(self, X: Tensor):
        self.model.eval()
        mu, cov_mat = self.model(X)
        norm = MultivariateNormal(loc=mu, scale_tril=cov_mat)
        return norm.sample()
