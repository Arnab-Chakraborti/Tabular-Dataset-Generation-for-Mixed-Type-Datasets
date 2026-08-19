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
    cfg_drop_prob=0.15  
):
    criterion = nn.MSELoss()
    loss_history = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in latent_loader:
            z = batch[0].to(device)
            context = batch[1].to(device) if len(batch) > 1 else None
            
            # --- CFG Context Dropout Logic ---
            if context is not None:
                # Create a boolean mask where True means "drop the context"
                drop_mask = torch.rand(z.size(0), device=device) < cfg_drop_prob
                
                # Replacing dropped contexts with absolute zeros
                context = torch.where(
                    drop_mask.unsqueeze(1), 
                    torch.zeros_like(context), 
                    context
                )

            optimizer.zero_grad()

            t = scheduler.sample_timesteps(batch_size=z.size(0))
            z_t, noise = scheduler.add_noise(z, t)

            predicted_noise = model(
                noisy_latent=z_t,
                timesteps=t,
                context=context
            )

            loss = criterion(predicted_noise, noise)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= len(latent_loader)
        loss_history.append(epoch_loss)
        
        mlflow.log_metric("Diffusion Training Loss", epoch_loss, step=epoch)
        print(f"Epoch [{epoch+1}/{epochs}] Diffusion Loss: {epoch_loss:.6f}")

    return model, loss_history
