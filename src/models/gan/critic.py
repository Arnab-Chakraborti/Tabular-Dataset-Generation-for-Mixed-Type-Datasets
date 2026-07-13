"""
Critic network for WGAN-GP.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.configs.gan import GANConfig


class Critic(nn.Module):
    """
    Critic network used in WGAN-GP.

    Parameters
    ----------
    config : GANConfig
        GAN configuration.

    cond_dim : int
        Dimension of conditional vector.

    data_dim : int
        Dimension of transformed data.
    """

    def __init__(
        self,
        config: GANConfig,
        cond_dim: int,
        data_dim: int,
    ):
        super().__init__()

        self.config = config

        input_dim = data_dim + cond_dim

        self.network = nn.Sequential(

            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Real or fake transformed samples.

        cond : torch.Tensor, optional
            Conditional vector.

        Returns
        -------
        torch.Tensor
            Critic score.
        """

        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        return self.network(x)