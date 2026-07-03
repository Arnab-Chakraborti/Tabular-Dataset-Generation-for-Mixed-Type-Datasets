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
    ):
        """
        Generates latent vectors by reverse diffusion.

        Returns
        -------
        z0 : torch.Tensor
            Clean latent vectors of shape
            (num_samples, latent_dim)
        """

        self.model.eval()

        # Initial Gaussian latent
        z = torch.randn(
            num_samples,
            latent_dim,
            device=self.device,
        )

        # Reverse diffusion
        for timestep in reversed(range(self.scheduler.num_timesteps)):
            t = torch.full(
                (num_samples,),
                timestep,
                device=self.device,
                dtype=torch.long,
            )
            predicted_noise = self.model(
                noisy_latent=z,
                timesteps=t,
            )
            beta = self.scheduler.betas[timestep]
            alpha = self.scheduler.alphas[timestep]
            alpha_bar = self.scheduler.alpha_bars[timestep]

            # DDPM mean estimate
            z = (1.0 / torch.sqrt(alpha)) * (z-((1 - alpha) / torch.sqrt(1 - alpha_bar))* predicted_noise)

            # Add stochastic noise except at t=0
            if timestep > 0:
                noise = torch.randn_like(z)
                z = z + torch.sqrt(beta) * noise

        return z
