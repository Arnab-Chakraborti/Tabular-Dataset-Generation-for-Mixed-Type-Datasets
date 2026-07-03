'''
No diffusion logic: This file only defines the neural network.
It does not sample timesteps, add noise, compute the diffusion loss, or perform reverse sampling.
Those belong in diffusion_trainer.py and sampler.py.
'''

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (batch_size,) integer or float timesteps

        Returns:
            (batch_size, embedding_dim)
        """
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=device) * exponent
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=1
        )
        if self.embedding_dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=1
            )
        return embedding


class ResidualMLPBlock(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: str = "SiLU",
        use_layernorm: bool = True
    ):
        super().__init__()

        activation_map = {
            "ReLU": nn.ReLU,
            "LeakyReLU": lambda: nn.LeakyReLU(0.2),
            "GELU": nn.GELU,
            "SiLU": nn.SiLU,
        }
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")
        layers = []
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation_map[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class DiffusionMLP(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list = [512, 512, 256],
        time_embedding_dim: int = 128,
        activation: str = "SiLU",
        dropout: float = 0.0,
        use_layernorm: bool = True,
        prediction_type: str = "noise"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.time_embedding_dim = time_embedding_dim
        self.activation = activation
        self.dropout = dropout
        self.use_layernorm = use_layernorm
        self.prediction_type = prediction_type
        self.time_embedding = SinusoidalTimeEmbedding(
            time_embedding_dim
        )
        input_dim = latent_dim + time_embedding_dim
        self.input_projection = nn.Linear(
            input_dim,
            hidden_dims[0]
        )
        blocks = []
        current_dim = hidden_dims[0]
        for next_dim in hidden_dims:
            if next_dim != current_dim:
                blocks.append(
                    nn.Linear(current_dim, next_dim)
                )
                current_dim = next_dim
            blocks.append(
                ResidualMLPBlock(
                    current_dim,
                    dropout=dropout,
                    activation=activation,
                    use_layernorm=use_layernorm
                )
            )

        self.backbone = nn.Sequential(*blocks)
        self.output_projection = nn.Linear(
            current_dim,
            latent_dim
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        t_embed = self.time_embedding(timesteps)
        x = torch.cat(
            [noisy_latent, t_embed],
            dim=1
        )
        x = self.input_projection(x)
        x = self.backbone(x)
        prediction = self.output_projection(x)

        return prediction

    def get_config(self):

        return {
            "latent_dim": self.latent_dim,
            "hidden_dims": str(self.hidden_dims),
            "time_embedding_dim": self.time_embedding_dim,
            "activation": self.activation,
            "dropout": self.dropout,
            "use_layernorm": self.use_layernorm,
            "prediction_type": self.prediction_type,
        }
