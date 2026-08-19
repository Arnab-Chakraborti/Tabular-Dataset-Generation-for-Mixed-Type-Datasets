import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.schedulers import MMDAnnealingScheduler

def fast_mmd(real, fake, sigmas=[0.05, 0.1, 0.7, 5.0, 10.0]):
    # Vectorized distance calculation: ||x-y||^2 = ||x||^2 + ||y||^2 - 2x*y.T
    x_sq = torch.sum(real**2, dim=1).unsqueeze(1)
    y_sq = torch.sum(fake**2, dim=1).unsqueeze(0)

    dist_xx = torch.clamp(x_sq + x_sq.T - 2 * torch.mm(real, real.T), min=0.0)
    dist_yy = torch.clamp(y_sq + y_sq.T - 2 * torch.mm(fake, fake.T), min=0.0)
    dist_xy = torch.clamp(x_sq + y_sq - 2 * torch.mm(real, fake.T), min=0.0)
    total_mmd=0

    for s in sigmas:
        gamma = 1.0 / (2 * s**2)
        k_xx = torch.exp(-dist_xx * gamma)
        k_yy = torch.exp(-dist_yy * gamma)
        k_xy = torch.exp(-dist_xy * gamma)

        total_mmd += (k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())

    return total_mmd

def boundary_loss(fake_data, real_min, real_max):
    low_penalty = torch.relu(real_min - fake_data)
    high_penalty = torch.relu(fake_data - real_max)
    return torch.mean(low_penalty + high_penalty)

class TabVAELoss(nn.Module):
    def __init__(self, continuous_dim: int, cardinalities: list, mmd_policy: dict, total_epochs: int = 1000, continuous_cols: list = None, missing_indicators: list = None):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.cardinalities = cardinalities
        
        self.continuous_cols = continuous_cols or []
        self.missing_indicators = missing_indicators or []
        self.mask_pairs = []
        
        # Map the exact index of every base column to its specific _is_missing indicator
        for ind_col in self.missing_indicators:
            base_col = ind_col.replace('_is_missing', '')
            if base_col in self.continuous_cols and ind_col in self.continuous_cols:
                base_idx = self.continuous_cols.index(base_col)
                ind_idx = self.continuous_cols.index(ind_col)
                self.mask_pairs.append((base_idx, ind_idx))
        
        # --- PARSE THE UI DICTIONARY ---
        mode = mmd_policy.get("mode", "constant").lower()
        max_beta = 500.0
        warm_up_ratio = 0.25

        if mode == "constant":
            max_beta = mmd_policy.get("weight", 500.0)
            warm_up_ratio = 0.0
        elif mode == "linear":
            max_beta = mmd_policy.get("end_weight", 500.0)
            warmup_epochs = mmd_policy.get("warmup_epochs", 250)
            warm_up_ratio = warmup_epochs / max(1, total_epochs)
        elif mode == "cosine":
            max_beta = mmd_policy.get("max_weight", 500.0)
            warm_up_ratio = 0.0
        
        self.mmd_scheduler = MMDAnnealingScheduler(
            total_epochs=total_epochs, 
            max_beta=float(max_beta), 
            annealing_type=mode,
            warm_up_ratio=warm_up_ratio
        )

    def forward(self, recon_x: torch.Tensor, target_x: torch.Tensor, z: torch.Tensor, mu: torch.Tensor, logvar:torch.Tensor, current_epoch: int = 1):
        batch_size = recon_x.size(0)
        recon_loss = 0.0
        
        if self.continuous_dim > 0:
            pred_cont = recon_x[:, :self.continuous_dim]
            true_cont = target_x[:, :self.continuous_dim]
            
            raw_mse = F.mse_loss(pred_cont, true_cont, reduction='none')
            
            if self.mask_pairs:
                weight_matrix = torch.ones_like(raw_mse)
                for base_idx, ind_idx in self.mask_pairs:
                    is_missing_mask = true_cont[:, ind_idx]
                    # If target is missing (1), weight becomes 0. If valid (0), weight becomes 1.
                    weight_matrix[:, base_idx] = 1.0 - is_missing_mask
                    
                masked_mse = raw_mse * weight_matrix
                
                # Average only over valid cells and indicator columns to prevent artificial loss deflation
                recon_loss += masked_mse.sum() / torch.clamp(weight_matrix.sum(), min=1.0)
            else:
                recon_loss += raw_mse.mean()
                
        cat_loss = 0.0    
        current_idx = self.continuous_dim
        for card in self.cardinalities:
            pred_cat = recon_x[:, current_idx : current_idx + card]
            true_cat = target_x[:, current_idx : current_idx + card]
            true_cat_idx = torch.argmax(true_cat, dim=1)
            cat_loss += F.cross_entropy(pred_cat, true_cat_idx, reduction='mean')
            current_idx += card

        if len(self.cardinalities)>0:
            recon_loss+= (cat_loss/len(self.cardinalities))
            
        z_prior = torch.randn_like(z)
        mmd_loss = fast_mmd(z, z_prior)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        active_weight = self.mmd_scheduler.get_beta(current_epoch)
        total_loss = recon_loss + (active_weight * mmd_loss)+0.05*kl_loss

        return total_loss, recon_loss, mmd_loss, active_weight


def compute_diffusion_loss(denoising_model, z_0, t_max=1.0):
    """
    Implements Score Matching for Variance-Exploding Linear Noise Schedules.
    """
    batch_size = z_0.size(0)
    
    # Sample random times t and match them linearly to the noise level sigma(t) = t
    t = torch.rand(batch_size, device=z_0.device) * t_max
    sigma = t.view(-1, 1)
    
    # Generate ground truth Gaussian noise targets
    epsilon = torch.randn_like(z_0)
    
    # Add linear noise in the flattened space: z_t = z_0 + sigma(t) * epsilon
    z_t = z_0 + sigma * epsilon
    
    # Generate the prediction guess
    epsilon_pred = denoising_model(z_t, t)
    
    # Standard Mean Squared Error optimization checking matching
    loss = F.mse_loss(epsilon_pred, epsilon)
    return loss
