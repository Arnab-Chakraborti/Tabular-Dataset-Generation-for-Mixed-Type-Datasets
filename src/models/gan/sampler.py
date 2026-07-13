"""
Sampler for trained GAN models.
"""

from __future__ import annotations

import numpy as np
import torch

from src.configs.gan import GANConfig
from src.models.gan.generator import Generator


class Sampler:
    """
    Sample synthetic data from a trained Generator.

    Parameters
    ----------
    config : GANConfig
        GAN configuration.

    generator : Generator
        Trained generator.

    conditional_sampler
        Conditional sampler used during generation.

    device : str
        Device to run generation on.
    """

    def __init__(
        self,
        config: GANConfig,
        generator: Generator,
        conditional_sampler,
        device: str = "cuda",
    ):
        self.config = config
        self.generator = generator
        self.conditional_sampler = conditional_sampler
        self.device = device

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        hard: bool = True,
    ) -> np.ndarray:
        """
        Generate transformed synthetic samples.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.

        hard : bool
            Whether to use hard Gumbel Softmax.

        Returns
        -------
        np.ndarray
            Synthetic transformed samples.
        """

        self.generator.eval()

        generated = []

        remaining = num_samples

        while remaining > 0:

            batch_size = min(
                self.config.batch_size,
                remaining,
            )

            cond = self.conditional_sampler.sample_generation_condvec(
                batch_size
            )

            if cond is not None:

                cond = torch.tensor(
                    cond,
                    dtype=torch.float32,
                    device=self.device,
                )

            z = torch.randn(
                batch_size,
                self.config.latent_dim,
                device=self.device,
            )

            fake, _ = self.generator(
                z,
                cond,
                hard=hard,
            )

            generated.append(
                fake.cpu().numpy()
            )

            remaining -= batch_size

        self.generator.train()

        return np.concatenate(
            generated,
            axis=0,
        )