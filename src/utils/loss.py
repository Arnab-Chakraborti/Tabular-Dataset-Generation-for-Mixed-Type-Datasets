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

    def forward(self, recon_x: torch.Tensor, target_x: torch.Tensor, z: torch.Tensor):
        batch_size= recon_x.size(0)
        recon_loss=0.0
        if self.continuous_dim>0:
            pred_cont=recon_x[:, :self.continuous_dim]
            true_cont=target_x[:, :self.continuous_dim]

            recon_loss+= F.mse_loss(pred_cont, true_cont, reduction='mean')
        current_idx= self.continuous_dim
        for card in self.cardinalities:
            pred_cat=recon_x[:, current_idx : current_idx + card]
            true_cat=target_x[:, current_idx : current_idx + card]
            true_cat_idx= torch.argmax(true_cat, dim=1)
            recon_loss+=F.cross_entropy(pred_cat, true_cat_idx, reduction= 'mean')
            current_idx+=card
        z_prior= torch.randn_like(z)
        mmd_loss= fast_mmd(z, z_prior)
        total_loss= recon_loss + (self.mmd_weight * mmd_loss)

        return total_loss, recon_loss, mmd_loss

def sliced_wasserstein_distance(X, Y, num_projections=1000, p=2, device="cpu"):    
    projections = torch.randn(X.size(1), num_projections, device=device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)
    X_projected = torch.matmul(X, projections)
    Y_projected = torch.matmul(Y, projections)
    
    # Sort
    X_sorted, _ = torch.sort(X_projected, dim=0)
    Y_sorted, _ = torch.sort(Y_projected, dim=0)
    
    # L2 (or p-norm)
    delta = X_sorted - Y_sorted
    loss = torch.pow(torch.abs(delta), p).mean()
    
    return loss
