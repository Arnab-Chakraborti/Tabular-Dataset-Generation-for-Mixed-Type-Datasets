import torch
from src.models.gan.losses import get_criterion


class GANTrainer:

    def __init__(
        self,
        generator,
        discriminator,
        train_loader,
        latent_dim,
        device,
        learning_rate=2e-4,
    ):

        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        self.train_loader = train_loader

        self.latent_dim = latent_dim

        self.device = device

        self.criterion = get_criterion()

        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.999),
        )

        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.999),
        )

    ####################################################################
    # Train
    ####################################################################

    def train(self, num_epochs):

        for epoch in range(num_epochs):

            self.generator.train()
            self.discriminator.train()

            d_epoch_loss = 0.0
            g_epoch_loss = 0.0

            for batch in self.train_loader:

                ####################################
                # Train Discriminator
                ####################################

                real_batch = batch[0].to(self.device)

                batch_size = real_batch.size(0)

                real_labels = torch.ones(
                    batch_size,
                    1,
                    device=self.device,
                )

                fake_labels = torch.zeros(
                    batch_size,
                    1,
                    device=self.device,
                )

                z = torch.randn(
                    batch_size,
                    self.latent_dim,
                    device=self.device,
                )

                fake_batch = self.generator(z)

                real_output = self.discriminator(real_batch)

                fake_output = self.discriminator(
                    fake_batch.detach()
                )

                real_loss = self.criterion(
                    real_output,
                    real_labels,
                )

                fake_loss = self.criterion(
                    fake_output,
                    fake_labels,
                )

                d_loss = real_loss + fake_loss

                self.discriminator_optimizer.zero_grad()

                d_loss.backward()

                self.discriminator_optimizer.step()

                ####################################
                # Train Generator
                ####################################

                z = torch.randn(
                    batch_size,
                    self.latent_dim,
                    device=self.device,
                )

                fake_batch = self.generator(z)

                output = self.discriminator(fake_batch)

                g_loss = self.criterion(
                    output,
                    real_labels,
                )

                self.generator_optimizer.zero_grad()

                g_loss.backward()

                self.generator_optimizer.step()

                d_epoch_loss += d_loss.item()

                g_epoch_loss += g_loss.item()

            print(
                f"Epoch {epoch + 1:03d} | "
                f"D Loss: {d_epoch_loss / len(self.train_loader):.4f} | "
                f"G Loss: {g_epoch_loss / len(self.train_loader):.4f}"
            )