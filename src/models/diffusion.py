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


class FiLMResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
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
            
        self.use_layernorm = use_layernorm
        
        # Main data transformation
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
        if self.use_layernorm:
            self.norm = nn.LayerNorm(hidden_dim)
            
        # FiLM Projection: Maps the conditioning vector to Gamma (scale) and Beta (shift)
        # We multiply hidden_dim by 2 so we can chunk it cleanly in the forward pass
        self.film_projection = nn.Linear(cond_dim, hidden_dim * 2)
        
        self.act = activation_map[activation]()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, cond):
        # 1. Base transformation
        h = self.fc(x)
        
        # 2. Normalize
        if self.use_layernorm:
            h = self.norm(h)
            
        # 3. FiLM Modulation
        # Project conditioning vector and split it into scale (gamma) and shift (beta)
        film_params = self.film_projection(cond)
        gamma, beta = film_params.chunk(2, dim=-1)
        
        # Apply the modulation: h = (gamma * h) + beta
        # We use (1 + gamma) so initializing the linear layer near 0 defaults to an identity mapping
        h = (1.0 + gamma) * h + beta 
        
        # 4. Activate and Dropout
        h = self.act(h)
        h = self.dropout(h)
        
        # 5. Residual Connection
        return x + h


class DiffusionMLP(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_dim: int = 0,
        hidden_dims: list = [512, 512, 256],
        time_embedding_dim: int = 128,
        activation: str = "SiLU",
        dropout: float = 0.0,
        use_layernorm: bool = True,
        prediction_type: str = "noise"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim 
        self.hidden_dims = hidden_dims
        self.time_embedding_dim = time_embedding_dim
        self.activation = activation
        self.dropout = dropout
        self.use_layernorm = use_layernorm
        self.prediction_type = prediction_type
        
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        
        # Determine the total size of the conditioning vector
        self.cond_dim = time_embedding_dim
        if self.context_dim > 0:
            self.context_projection = nn.Linear(self.context_dim, time_embedding_dim)
            # We will concatenate Time and Context, so the dimension doubles safely
            self.cond_dim = time_embedding_dim * 2 
            
        # The input projection ONLY takes the noisy latent now
        self.input_projection = nn.Linear(latent_dim, hidden_dims[0])
        
        # Build the deep FiLM network
        self.blocks = nn.ModuleList()
        current_dim = hidden_dims[0]
        
        for next_dim in hidden_dims:
            if next_dim != current_dim:
                self.blocks.append(nn.Linear(current_dim, next_dim))
                current_dim = next_dim
                
            self.blocks.append(
                FiLMResidualBlock(
                    hidden_dim=current_dim,
                    cond_dim=self.cond_dim, # Pass the total conditioning dimension
                    dropout=dropout,
                    activation=activation,
                    use_layernorm=use_layernorm
                )
            )

        self.output_projection = nn.Linear(current_dim, latent_dim)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor = None 
    ) -> torch.Tensor:
        
        # 1. Get Time Embedding
        t_embed = self.time_embedding(timesteps)
        
        # 2. Build the cleanly concatenated Conditioning Vector (cond)
        if self.context_dim > 0 and context is not None:
            c_embed = self.context_projection(context)
            # Concatenation prevents destructive interference!
            cond = torch.cat([t_embed, c_embed], dim=1) 
        else:
            cond = t_embed
            
        # 3. Process Latent Data
        x = self.input_projection(noisy_latent)
        
        # 4. Route through FiLM Blocks
        for layer in self.blocks:
            if isinstance(layer, FiLMResidualBlock):
                x = layer(x, cond) # Pass the clean conditioning vector into every block
            else:
                x = layer(x)
                
        # 5. Output Prediction
        prediction = self.output_projection(x)

        return prediction

    def get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "context_dim": self.context_dim, 
            "hidden_dims": str(self.hidden_dims),
            "time_embedding_dim": self.time_embedding_dim,
            "activation": self.activation,
            "dropout": self.dropout,
            "use_layernorm": self.use_layernorm,
            "prediction_type": self.prediction_type,
        }
