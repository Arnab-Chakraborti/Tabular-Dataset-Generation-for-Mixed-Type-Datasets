import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, data_dim):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(data_dim, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 1)

            # No Sigmoid here
        )

    def forward(self, x):
        return self.network(x)