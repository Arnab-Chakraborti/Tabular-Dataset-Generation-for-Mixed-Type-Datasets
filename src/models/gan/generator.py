"""
Generator network for the CTGAN-inspired WGAN-GP model.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configs.gan import GANConfig


class Generator(nn.Module):
    """
    Generator network.

    Parameters
    ----------
    config : GANConfig
        GAN configuration.

    cond_dim : int
        Dimension of the conditional vector.

    data_dim : int
        Dimension of the transformed data.

    activation_layout : list
        Activation layout returned by DataTransformer.
    """

    def __init__(
        self,
        config: GANConfig,
        cond_dim: int,
        data_dim: int,
        activation_layout: List[dict],
    ):
        super().__init__()

        self.config = config
        self.activation_layout = activation_layout
        self.cond_dim = cond_dim

        input_dim = config.latent_dim + cond_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, data_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        z : torch.Tensor
            Latent noise.

        cond : torch.Tensor, optional
            Conditional vector.

        hard : bool, default=False
            Whether to use hard Gumbel-Softmax.

        Returns
        -------
        generated_data : torch.Tensor
            Generated transformed samples.

        categorical_logits : List[torch.Tensor]
            Raw logits of every categorical segment.
            These are used for conditional loss.
        """

        if cond is not None:
            gen_input = torch.cat([z, cond], dim=1)
        else:
            gen_input = z

        raw_output = self.network(gen_input)

        outputs = []
        categorical_logits = []

        for segment in self.activation_layout:

            start = segment["start"]
            end = segment["end"]

            if segment["type"] == "continuous":

                # alpha
                alpha = torch.tanh(
                    raw_output[:, start:start + 1]
                )

                # beta
                beta_logits = raw_output[:, start + 1:end]

                beta = F.softmax(
                    beta_logits,
                    dim=-1,
                )

                outputs.append(alpha)
                outputs.append(beta)

            else:

                logits = raw_output[:, start:end]

                categorical_logits.append(logits)

                categorical = F.gumbel_softmax(
                    logits,
                    tau=self.config.temperature,
                    hard=hard,
                    dim=-1,
                )

                outputs.append(categorical)

        generated_data = torch.cat(
            outputs,
            dim=1,
        )

        return generated_data, categorical_logits