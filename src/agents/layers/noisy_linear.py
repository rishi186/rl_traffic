"""Noisy Dense layer for exploration without epsilon-greedy.

Implements Factorised Gaussian Noise as in Fortunato et al., 2017
("Noisy Networks for Exploration"). The noise is learned and automatically
adapts — no epsilon schedule required.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorised NoisyNet dense layer.

    Replaces a standard Linear layer. During training the noise encourages
    exploration; during inference `self.training` is False to suppress it for
    deterministic evaluation.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        sigma_init: Initial noise standard deviation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable mean parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # Learnable noise magnitude
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Non-learnable noise buffers
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize parameters using Uniform distribution."""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        sigma_val = self.sigma_init / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma_val)
        self.bias_sigma.data.fill_(sigma_val)

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        """Factorised noise function: sign(x) * sqrt(|x|)."""
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """Generate and update weight and bias noise matrices."""
        device = self.weight_mu.device
        epsilon_in = torch.randn(self.in_features, device=device)
        epsilon_out = torch.randn(self.out_features, device=device)

        f_in = self._f(epsilon_in)
        f_out = self._f(epsilon_out)

        # Outer product of f_out and f_in
        self.weight_epsilon.copy_(torch.outer(f_out, f_in))
        self.bias_epsilon.copy_(f_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)
