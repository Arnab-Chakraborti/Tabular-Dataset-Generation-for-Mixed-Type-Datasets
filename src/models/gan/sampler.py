import torch


class Sampler:

    def __init__(
        self,
        generator,
        latent_dim,
        device,
    ):

        self.generator = generator
        self.latent_dim = latent_dim
        self.device = device

    ####################################################################
    # Generate Synthetic Data
    ####################################################################

    def sample(self, num_samples):

        # Put generator in evaluation mode
        self.generator.eval()

        with torch.no_grad():

            # Generate random latent vectors
            z = torch.randn(
                num_samples,
                self.latent_dim,
                device=self.device
            )

            # Generate synthetic transformed samples
            synthetic_transformed = self.generator(z)

        # Move to CPU
        synthetic_transformed = synthetic_transformed.cpu().numpy()

        return synthetic_transformed