"""
Trainer for CTGAN-inspired WGAN-GP.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.configs.gan import GANConfig
from src.models.gan.generator import Generator
from src.models.gan.critic import Critic
from src.models.gan.losses import (
    critic_loss,
    generator_loss,
    gradient_penalty,
    conditional_loss,
)


class GANTrainer:

    def __init__(
        self,
        config: GANConfig,
        generator: Generator,
        critic: Critic,
        conditional_sampler,
        device: str = "cuda",
    ):

        self.config = config

        self.generator = generator.to(device)
        self.critic = critic.to(device)

        self.conditional_sampler = conditional_sampler

        self.device = device

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )

        self.c_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )

        self.history = {
            "critic_loss": [],
            "generator_loss": [],
            "gradient_penalty": [],
            "wasserstein": [],
            "conditional_loss": [],
        }

    ##################################################################
    # Noise
    ##################################################################

    def sample_noise(
        self,
        batch_size,
    ):

        return torch.randn(
            batch_size,
            self.config.latent_dim,
            device=self.device,
        )

    ##################################################################
    # Train Critic
    ##################################################################

    def train_critic(
        self,
        real_batch,
    ):

        batch_size = real_batch.size(0)

        cond, col, opt = self.conditional_sampler.sample_train_condvec(
            batch_size
        )

        cond = torch.tensor(
            cond,
            dtype=torch.float32,
            device=self.device,
        )

        z = self.sample_noise(batch_size)

        with torch.no_grad():

            fake_batch, _ = self.generator(
                z,
                cond,
            )

        critic_real = self.critic(
            real_batch,
            cond,
        )

        critic_fake = self.critic(
            fake_batch.detach(),
            cond,
        )

        wasserstein = critic_loss(
            critic_real,
            critic_fake,
        )

        gp = gradient_penalty(
            self.critic,
            real_batch,
            fake_batch.detach(),
            cond,
        )

        loss = (
            wasserstein
            + self.config.lambda_gp * gp
        )

        self.c_optimizer.zero_grad()

        loss.backward()

        self.c_optimizer.step()

        return (
            loss.item(),
            wasserstein.item(),
            gp.item(),
        )

    ##################################################################
    # Train Generator
    ##################################################################

    def train_generator(
        self,
        batch_size,
    ):

        cond, _, _ = self.conditional_sampler.sample_train_condvec(
            batch_size
        )

        cond = torch.tensor(
            cond,
            dtype=torch.float32,
            device=self.device,
        )

        z = self.sample_noise(batch_size)

        fake_batch, logits = self.generator(
            z,
            cond,
        )

        fake_score = self.critic(
            fake_batch,
            cond,
        )

        w_loss = generator_loss(
            fake_score,
        )

        c_loss = conditional_loss(
            logits,
            cond,
            self.conditional_sampler.discrete_columns,
        )

        loss = (
            w_loss
            + self.config.conditional_loss_weight
            * c_loss
        )

        self.g_optimizer.zero_grad()

        loss.backward()

        self.g_optimizer.step()

        return (
            loss.item(),
            w_loss.item(),
            c_loss.item(),
        )

    ##################################################################
    # Training Loop
    ##################################################################

    def fit(
        self,
        train_loader: DataLoader,
    ):

        self.generator.train()
        self.critic.train()

        for epoch in range(self.config.epochs):

            critic_epoch_loss = 0.0
            generator_epoch_loss = 0.0
            wasserstein_epoch = 0.0
            gp_epoch = 0.0
            cond_epoch = 0.0

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.epochs}",
                leave=False,
            )

            for real_batch in progress:

                real_batch = real_batch.to(self.device)

                ########################################################
                # Train Critic
                ########################################################

                for _ in range(self.config.n_critic):

                    (
                        critic_loss_value,
                        wasserstein,
                        gp,
                    ) = self.train_critic(real_batch)

                    critic_epoch_loss += critic_loss_value
                    wasserstein_epoch += wasserstein
                    gp_epoch += gp

                ########################################################
                # Train Generator
                ########################################################

                (
                    generator_loss_value,
                    w_loss,
                    cond_loss_value,
                ) = self.train_generator(
                    real_batch.size(0)
                )

                generator_epoch_loss += generator_loss_value
                cond_epoch += cond_loss_value

                progress.set_postfix(
                    {
                        "C": f"{critic_loss_value:.3f}",
                        "G": f"{generator_loss_value:.3f}",
                        "GP": f"{gp:.3f}",
                    }
                )

            ############################################################
            # Epoch Statistics
            ############################################################

            num_batches = len(train_loader)

            critic_epoch_loss /= (
                num_batches * self.config.n_critic
            )

            wasserstein_epoch /= (
                num_batches * self.config.n_critic
            )

            gp_epoch /= (
                num_batches * self.config.n_critic
            )

            generator_epoch_loss /= num_batches
            cond_epoch /= num_batches

            self.history["critic_loss"].append(
                critic_epoch_loss
            )

            self.history["generator_loss"].append(
                generator_epoch_loss
            )

            self.history["wasserstein"].append(
                wasserstein_epoch
            )

            self.history["gradient_penalty"].append(
                gp_epoch
            )

            self.history["conditional_loss"].append(
                cond_epoch
            )

            print(
                f"Epoch {epoch + 1:03d} | "
                f"Critic: {critic_epoch_loss:.4f} | "
                f"Generator: {generator_epoch_loss:.4f} | "
                f"GP: {gp_epoch:.4f}"
            )

    ##################################################################
    # Generate Samples
    ##################################################################

    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int,
        hard: bool = True,
    ):
        """
        Generate synthetic samples.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.

        hard : bool
            Whether to use hard Gumbel-Softmax.

        Returns
        -------
        torch.Tensor
            Generated transformed samples.
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

            z = self.sample_noise(batch_size)

            fake, _ = self.generator(
                z,
                cond,
                hard=hard,
            )

            generated.append(
                fake.cpu()
            )

            remaining -= batch_size

        self.generator.train()

        return torch.cat(
            generated,
            dim=0,
        )

    ##################################################################
    # Save Checkpoint
    ##################################################################

    def save_checkpoint(
        self,
        path,
    ):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "generator": self.generator.state_dict(),
                "critic": self.critic.state_dict(),
                "generator_optimizer": self.g_optimizer.state_dict(),
                "critic_optimizer": self.c_optimizer.state_dict(),
                "history": self.history,
                "config": self.config,
            },
            path,
        )

    ##################################################################
    # Load Checkpoint
    ##################################################################

    def load_checkpoint(
        self,
        path,
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.generator.load_state_dict(
            checkpoint["generator"]
        )

        self.critic.load_state_dict(
            checkpoint["critic"]
        )

        self.g_optimizer.load_state_dict(
            checkpoint["generator_optimizer"]
        )

        self.c_optimizer.load_state_dict(
            checkpoint["critic_optimizer"]
        )

        self.history = checkpoint["history"]

    ##################################################################
    # Get Training History
    ##################################################################

    def get_history(self):
        """
        Return training history.
        """

        return self.history

