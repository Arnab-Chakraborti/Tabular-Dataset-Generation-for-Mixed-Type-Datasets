import math
import torch
import torch.nn as nn

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TabSynDenoisingMLP(nn.Module):
    def __init__(self, num_columns, d_token=4, d_hidden=1024):
        super().__init__()
        # Flattened input configuration dimension size: M * d
        self.in_dim = num_columns * d_token
        
        # 1. Input Layer Projection
        self.input_layer = nn.Linear(self.in_dim, d_hidden)
        
        # 2. Sinusoidal Time Conditioning Track
        self.time_layer = nn.Sequential(
            SinusoidalEmbedding(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        
        # 3. Core 3-Layer Dense Hidden Backbone utilizing SiLU blocks
        self.fc1 = nn.Linear(d_hidden, 2 * d_hidden)
        self.fc2 = nn.Linear(2 * d_hidden, 2 * d_hidden)
        self.fc3 = nn.Linear(2 * d_hidden, d_hidden)
        
        self.act = nn.SiLU()
        
        # 4. Final Target Estimation Projection
        self.output_layer = nn.Linear(d_hidden, self.in_dim)

    def forward(self, z_t, t):
        # Base representation matrix extraction
        h0 = self.input_layer(z_t)
        
        # Embedding time coordinates and merge via addition
        t_emb = self.time_layer(t)
        h_in = h0 + t_emb
        
        # Forward execution through deep hidden structural layers
        h1 = self.act(self.fc1(h_in))
        h2 = self.act(self.fc2(h1))
        h3 = self.act(self.fc3(h2))
        
        # Map back to latent noise prediction limits
        return self.output_layer(h3)
