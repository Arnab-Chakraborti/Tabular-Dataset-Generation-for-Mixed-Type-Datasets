import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, latent_dim, data_dim):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, data_dim),

            # Continuous outputs are approximately in [-1,1]
            nn.Tanh()
        )

    def forward(self, z):
        return self.network(z)