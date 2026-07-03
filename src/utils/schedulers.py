# src/utils/schedulers.py
import numpy as np
import torch

class MMDAnnealingScheduler:
    def __init__(self, total_epochs, max_beta=1.0, annealing_type="linear", warm_up_ratio=0.3):
        """
        Args:
            total_epochs (int): Total training epochs.
            max_beta (float): Maximum weight multiplier for MMD loss.
            annealing_type (str): "linear", "sigmoid", or "none".
            warm_up_ratio (float): Fraction of total epochs dedicated to annealing.
        """
        self.total_epochs = total_epochs
        self.max_beta = max_beta
        self.annealing_type = annealing_type
        self.anneal_until_epoch = int(total_epochs * warm_up_ratio)

    def get_beta(self, current_epoch):
        if self.annealing_type == "none" or current_epoch == 0:
            return self.max_beta
        
        if current_epoch >= self.anneal_until_epoch:
            return self.max_beta

        if self.annealing_type == "linear":
            # Linear ramp from 0 up to max_beta
            return self.max_beta * (current_epoch / self.anneal_until_epoch)
            
        elif self.annealing_type == "sigmoid":
            # Sigmoidal curve progression centered in the middle of warm_up zone
            center = self.anneal_until_epoch / 2
            gain = 6.0 / (self.anneal_until_epoch) # Control slope sharpness
            return self.max_beta / (1.0 + np.exp(-gain * (current_epoch - center)))
            
        return self.max_beta
    
class DiffusionScheduler:
    def __init__(
        self,
        num_timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule: str = "linear",
        device: str = "cpu",
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule
        self.device = device
        if schedule != "linear":
            raise NotImplementedError(
                f"{schedule} schedule not implemented."
            )
        # βt
        self.betas = torch.linspace(
            beta_start,
            beta_end,
            num_timesteps,
            device=device
        )
        # αt = 1 - βt
        self.alphas = 1.0 - self.betas
        # ᾱt = ∏ αt
        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0
        )
        # Precompute constants
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(
            1.0 - self.alpha_bars
        )
    def sample_timesteps(
        self,
        batch_size: int
    ):
        return torch.randint(
            low=0,
            high=self.num_timesteps,
            size=(batch_size,),
            device=self.device,
        )
    def add_noise(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
    ):
        noise = torch.randn_like(z)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].unsqueeze(1)
        sqrt_one_minus_alpha_bar = (
            self.sqrt_one_minus_alpha_bars[t].unsqueeze(1)
        )
        noisy_latent = (
            sqrt_alpha_bar * z
            + sqrt_one_minus_alpha_bar * noise
        )
        return noisy_latent, noise

    def get_config(self):

        return {
            "num_timesteps": self.num_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "schedule": self.schedule,
        }
