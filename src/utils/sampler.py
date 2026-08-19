import torch

class DiffusionSampler:
    def __init__(
        self,
        diffusion_model,
        scheduler,
        device="cpu",
    ):
        self.model = diffusion_model
        self.scheduler = scheduler
        self.device = device

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        latent_dim: int,
        context: torch.Tensor = None,
        guidance_scale: float = 1.5 # The strength of the UI steering
    ):
        self.model.eval()

        if context is not None:
            context = context.to(self.device)
            num_samples = context.size(0)

        z = torch.randn(num_samples, latent_dim, device=self.device)

        for timestep in reversed(range(self.scheduler.num_timesteps)):
            t = torch.full((num_samples,), timestep, device=self.device, dtype=torch.long)

            # --- CFG Double Forward Pass ---
            if context is not None and guidance_scale > 1.0:
                # 1. Duplicate inputs for a single batched pass (for speed)
                z_in = torch.cat([z, z], dim=0)
                t_in = torch.cat([t, t], dim=0)
                
                # the unconditional zeros mask
                uncond_context = torch.zeros_like(context)
                context_in = torch.cat([context, uncond_context], dim=0)
                
                #  Predict both at once
                noise_pred = self.model(noisy_latent=z_in, timesteps=t_in, context=context_in)
                
                #  Splitting predictions back apart
                noise_cond, noise_uncond = noise_pred.chunk(2, dim=0)
                
                #  CFG Mathematical Formula
                predicted_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                
            else:
                # Standard fallback (unconditional or scale = 1.0)
                predicted_noise = self.model(noisy_latent=z, timesteps=t, context=context)

            beta = self.scheduler.betas[timestep]
            alpha = self.scheduler.alphas[timestep]
            alpha_bar = self.scheduler.alpha_bars[timestep]

            # DDPM mean estimate
            z = (1.0 / torch.sqrt(alpha)) * (
                z - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise
            )

            if timestep > 0:
                noise = torch.randn_like(z)
                z = z + torch.sqrt(beta) * noise

        return z
