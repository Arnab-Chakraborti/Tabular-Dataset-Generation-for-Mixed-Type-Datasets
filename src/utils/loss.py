import torch
import torch.nn as nn
import torch.nn.functional as F

def fast_mmd(real, fake, sigmas=[0.05, 0.1, 0.7, 5.0, 10.0]):
    # Vectorized distance calculation: ||x-y||^2 = ||x||^2 + ||y||^2 - 2x*y.T
    x_sq = torch.sum(real**2, dim=1).unsqueeze(1)
    y_sq = torch.sum(fake**2, dim=1).unsqueeze(0)

    dist_xx = x_sq + x_sq.T - 2 * torch.mm(real, real.T)
    dist_yy = y_sq + y_sq.T - 2 * torch.mm(fake, fake.T)
    dist_xy = x_sq + y_sq - 2 * torch.mm(real, fake.T)
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
    def __init__(self, continuous_dim: int, cardinalities: list, mmd_weight: float =1.0):
        super().__init__()
        self.continuous_dim= continuous_dim
        self.cardinalities= cardinalities
        self.mmd_weight= mmd_weight

    def forward(self, recon_x: torch.Tensor, target_x: torch.Tensor, z: torch.Tensor,mmd_weight=None):
        batch_size= recon_x.size(0)
        recon_loss=0.0
        if self.continuous_dim>0:
            pred_cont=recon_x[:, :self.continuous_dim]
            true_cont=target_x[:, :self.continuous_dim]

            recon_loss+=F.mse_loss(pred_cont, true_cont, reduction='mean')
            #swd_loss=sliced_wasserstein_distance(pred_cont, true_cont,device=recon_x.device)
            #recon_loss+=5.0*swd_loss
        current_idx= self.continuous_dim
        for card in self.cardinalities:
            pred_cat=recon_x[:, current_idx : current_idx + card]
            true_cat=target_x[:, current_idx : current_idx + card]
            true_cat_idx= torch.argmax(true_cat, dim=1)
            recon_loss+=F.cross_entropy(pred_cat, true_cat_idx, reduction= 'mean')
            current_idx+=card
        z_prior= torch.randn_like(z)
        mmd_loss= fast_mmd(z, z_prior)
        active_weight = mmd_weight if mmd_weight is not None else self.mmd_weight
        total_loss = recon_loss + (active_weight * mmd_loss)

        return total_loss, recon_loss, mmd_loss


class TabSynLossEngine:
    def __init__(self, beta_max=0.01, beta_min=1e-5, lambda_decay=0.7, patience=5):
        self.beta = beta_max
        self.beta_min = beta_min
        self.lambda_decay = lambda_decay
        self.patience = patience
        
        # Stabilization tracking metrics
        self.best_recon_loss = float('inf')
        self.patience_counter = 0

    def compute_vae_loss(self, x_num, x_cat_true_labels, x_num_hat, x_cat_logits_hat, mu, logvar):
        """
        Calculates Beta-VAE loss metrics incorporating multi-headed variable definitions.
        """
        # 1. Numerical Verification Loss (MSE)
        recon_num = F.mse_loss(x_num_hat, x_num) if x_num is not None else 0.0
        
        # 2. Categorical Verification Loss (Cross Entropy)
        recon_cat = 0.0
        for i, logits in enumerate(x_cat_logits_hat):
            recon_cat += F.cross_entropy(logits, x_cat_true_labels[:, i])
            
        if len(x_cat_logits_hat) > 0:
            recon_cat = recon_cat / len(x_cat_logits_hat)
            
        total_recon = recon_num + recon_cat
        
        # 3. Latent Regularization Loss Space Kullback-Leibler tracking
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / mu.size(0) # Standard Normalize by Batch size
        
        # Apply current adaptive weight scalar constraint
        total_loss = total_recon + self.beta * kl_loss
        
        return total_loss, total_recon, kl_loss

    def update_beta_schedule(self, current_epoch_recon_loss):
        """
        Executes adaptive weight updating if reconstruction progress plateaus.
        """
        if current_epoch_recon_loss < self.best_recon_loss:
            self.best_recon_loss = current_epoch_recon_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            self.beta = max(self.beta_min, self.beta * self.lambda_decay)
            self.patience_counter = 0
            print(f"[VAE Telemetry] Decay triggered. New Adaptive Beta Constraint: {self.beta:.6f}")


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
