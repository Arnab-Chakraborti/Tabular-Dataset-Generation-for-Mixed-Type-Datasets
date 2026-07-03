import mlflow
import torch
import torch.nn as nn


def train_diffusion(
    model,
    scheduler,
    latent_loader,
    optimizer,
    epochs,
    device,
):
    """
    Train diffusion model on VAE latent vectors.

    Args:
        model: DiffusionMLP
        scheduler: DiffusionScheduler
        latent_loader: DataLoader over latent vectors
        optimizer: Torch optimizer
        epochs: Number of diffusion epochs
        device: cpu/cuda

    Returns:
        model
        training_loss_history
    """

    criterion = nn.MSELoss()
    loss_history = []
    model.train()
    for epoch in range(epochs):

        epoch_loss = 0.0

        for (z,) in latent_loader:
            z = z.to(device)
            optimizer.zero_grad()

            # ----------------------------
            # Sample diffusion timestep
            # ----------------------------

            t = scheduler.sample_timesteps(
                batch_size=z.size(0)
            )

            # ----------------------------
            # Add forward noise
            # ----------------------------

            z_t, noise = scheduler.add_noise(
                z,
                t
            )

            predicted_noise = model(
                noisy_latent=z_t,
                timesteps=t
            )

            loss = criterion(
                predicted_noise,
                noise
            )

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(latent_loader)
        loss_history.append(epoch_loss)
        mlflow.log_metric(
            "Diffusion Training Loss",
            epoch_loss,
            step=epoch
        )
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Diffusion Loss: {epoch_loss:.6f}"
        )

    return model, loss_history
