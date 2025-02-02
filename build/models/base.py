
from torch import Tensor
from torch.distributions import Distribution
from typing import Union, Any

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bias_enabled = False

    def dist(self, mean: Tensor, std: Tensor, latent: Tensor = None, verbose: int = None
             ) -> Union[tuple[Distribution, dict[str, Any]], Distribution]:
        extra = {}
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
                corr_params = torch.sigmoid(self.corr(latent))

                # Determine the batch shape from the correlation output (could be one or more batch dims)
                features_out = std.shape[-1]  # number of features (must equal 'features')

                # Create a full identity matrix of shape (F, F) and expand it to the batch shape.
                identity = torch.eye(features_out, device=corr_params.device, dtype=corr_params.dtype)
                # Clone after expansion to ensure a writable (contiguous) tensor.
                corr_matrix = identity.expand(*std.shape[:-1], features_out, features_out).clone()
                epsilon     = corr_matrix * 1e-8

                # Fill the lower-triangular part (excluding the diagonal) with the computed correlations.
                # Enforce symmetry: copy the lower triangle to the upper triangle.
                # We add the transpose and subtract the duplicate diagonal.
                corr_matrix[..., self.corr_indices[0], self.corr_indices[1]] = corr_params
                corr_matrix[..., self.corr_indices[1], self.corr_indices[0]] = corr_params

                # Build the covariance matrix: std_diag * corr_matrix * std_diag.
                assert torch.all(std >= 0)
                std_diag = torch.diag_embed(std)
                cov_matrix = std_diag @ corr_matrix @ std_diag
                cov_matrix = cov_matrix + epsilon

                if verbose and verbose >= 2:
                    extra['corr_params'] = corr_params
                    extra['corr_matrix'] = corr_matrix
                    extra['std_diag'] = std_diag
            else:
                cov_matrix = torch.diag_embed(std ** 2)
            try:
                distribution = torch.distributions.MultivariateNormal(mean, cov_matrix)
            except ValueError as e:
                if verbose is None or verbose < 2:
                    raise e
                else:
                    distribution = None
        else:
            raise NotImplementedError(f"Unsupported distribution")
        if verbose and verbose >= 2:
            return distribution, extra
        else:
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
