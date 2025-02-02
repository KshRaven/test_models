
from torch import Tensor
from typing import Union

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bias_enabled = False

    def dist(self, mean: Tensor, std: Tensor, latent: Tensor = None):
        if self.distribution == 'discrete':
            mean = mean.unsqueeze(-2)
            distribution = torch.distributions.Categorical(torch.softmax(mean, -1))
        elif self.distribution == 'normal':
            distribution = torch.distributions.Normal(mean, std)
        elif self.distribution == 'mult_var_normal':
            if latent is not None:
                # Create and cache the correlation layer and its lower-triangular indices, if needed.
                if not hasattr(self, 'corr'):
                    embedding, features = latent.shape[-1], std.shape[-1]
                    tril_params_num = (features * (features - 1)) // 2
                    self.corr = torch.nn.Linear(embedding, tril_params_num, device=std.device, dtype=std.dtype,
                                                bias=self.bias_enabled)
                    self.corr_indices = torch.tril_indices(features, features, offset=-1, device=std.device)

                # Compute the lower-triangular correlation parameters and reduce them to (-1, 1)
                corr = torch.tanh(self.corr(latent))

                # Determine the batch shape from the correlation output (could be one or more batch dims)
                features_out = std.shape[-1]  # number of features (must equal 'features')

                # Create a full identity matrix of shape (F, F) and expand it to the batch shape.
                identity = torch.eye(features_out, device=corr.device, dtype=corr.dtype)
                # Clone after expansion to ensure a writable (contiguous) tensor.
                corr_matrix = identity.expand(*std.shape[:-1], features_out, features_out).clone()

                # Fill the lower-triangular part (excluding the diagonal) with the computed correlations.
                corr_matrix[..., self.corr_indices[0], self.corr_indices[1]] = corr

                # Enforce symmetry: copy the lower triangle to the upper triangle.
                # We add the transpose and subtract the duplicate diagonal.
                corr_matrix = corr_matrix + corr_matrix.transpose(-1, -2) - torch.diag_embed(
                    torch.diagonal(corr_matrix, dim1=-2, dim2=-1))

                # Build the covariance matrix: std_diag * corr_matrix * std_diag.
                std_diag = torch.diag_embed(std)
                cov_matrix = std_diag @ corr_matrix @ std_diag
            else:
                cov_matrix = torch.diag_embed(std ** 2)
            distribution = torch.distributions.MultivariateNormal(mean, cov_matrix)
        else:
            raise NotImplementedError(f"Unsupported distribution")
        return distribution

    def get_mean(self, latent: Tensor) -> Tensor:
        raise NotImplementedError(f"No 'get_mean' method")

    def get_std(self, latent: Tensor) -> Tensor:
        raise NotImplementedError(f"No 'get_std' method")

    def get_action(self, state: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(f"No 'get_action' method")

    def evaluate_action(self, state: Tensor, action: Tensor) -> [Tensor, Union[Tensor, None]]:
        raise NotImplementedError(f"No 'evaluate_action' method")

    def get_policy(self, state: Tensor, **options) -> Tensor:
        raise NotImplementedError(f"No 'get_policy' method")

    def get_value(self, state: Tensor) -> Tensor:
        raise NotImplementedError(f"No 'get_value' method")
