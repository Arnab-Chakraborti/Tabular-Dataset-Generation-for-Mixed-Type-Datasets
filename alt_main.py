import os
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import argparse
import yaml
import mlflow
import mlflow.pytorch
from src.utils.sampler import DiffusionSampler
from src.models.diffusion import DiffusionMLP
from src.trainers.diffusion_trainer import train_diffusion
from src.utils.schedulers import DiffusionScheduler
from src.models.vae import MixedTabularVAE
from src.data_processing.data_preprocessing import TabularDataPreprocessor
from src.data_processing.data_postprocessing import TabularDataPostprocessor
from src.utils.loss import TabVAELoss
from src.Evaluations.eval_metrics import evaluate_generator_performance


def train_vae(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    """
    Handles only the VAE optimization and validation loops.
    """
    history = {
        "train_loss": [], "val_loss": [], 
        "train_recon": [], "val_recon": [], 
        "train_mmd": [], "val_mmd": [], 
        "tau": []
    }
    
    tau_min = 0.1
    tau_max = 1.0
    best_val_loss = float('inf')
    
    print("Starting Optimization:")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_recon, mmd_loss = 0.0, 0.0, 0.0
        cos_inner = math.pi * (epoch - 1) / epochs
        current_tau = tau_min + 0.5 * (tau_max - tau_min) * (1.0 + math.cos(cos_inner))
        model.decoder.tau = current_tau
        history["tau"].append(current_tau)
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            recon_x, z = model(batch_x)
            loss, recon_loss, train_mmd = criterion(recon_x, batch_x, z)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item()
            train_recon += recon_loss.item()
            mmd_loss += train_mmd.item()
        avg_train_loss = train_loss / len(train_loader)
        avg_train_recon = train_recon / len(train_loader)
        avg_train_mmd = mmd_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss, val_recon, val_mmd = 0.0, 0.0, 0.0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                recon_x, z = model(batch_x)
                loss, recon_loss, batch_mmd = criterion(recon_x, batch_x, z)
                
                val_loss += loss.item()
                val_recon += recon_loss.item()
                val_mmd += batch_mmd.item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon = val_recon / len(val_loader)
        avg_val_mmd = val_mmd / len(val_loader)
        
        # --- Logging & Checkpointing ---
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_recon"].append(avg_train_recon)
        history["val_recon"].append(avg_val_recon)
        history["train_mmd"].append(avg_train_mmd)
        history["val_mmd"].append(avg_val_mmd)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), weight_path)
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | MMD Loss: {avg_train_mmd:.4f}")
    return model, history


if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'adult', 'adult.data')
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find '{DATA_PATH}'.")
        exit(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_params.yaml")
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    epochs = config["vae_training"]["VAE epochs"]
    batch_size = config["vae_training"]["batch_size"]
    latent_dim = config["vae_training"]["latent_dim"]
    mmd_weight = config["vae_training"]["MMD Weight"]
    la=config["architecture"]["Encoder_dims"]
    lr = config["vae_training"]["lr"]
    val_split = 0.2
    weight_path = "best_vae_weights.pt"
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training engine initialised on device: {device}")
    print(f"Raw Tabular data loaded from {DATA_PATH}")
    raw_df = pd.read_csv(DATA_PATH)
    train_df, temp_df = train_test_split(raw_df, test_size=val_split, shuffle=True, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, shuffle=True, random_state=42)
    print(f"Split: {len(train_df)} Train | {len(val_df)} Val | {len(test_df)} Test")
    continuous_cols = raw_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = raw_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    preprocessor = TabularDataPreprocessor(
        continuous_cols=continuous_cols, 
        categorical_cols=categorical_cols,
        continuous_scaler="standard",  
        clip_outliers=False,        
        impute_missing=True
    )
    train_matrix = preprocessor.fit_transform(train_df)
    val_matrix = preprocessor.transform(val_df)
    test_matrix = preprocessor.transform(test_df)  
    train_loader = DataLoader(TensorDataset(torch.tensor(train_matrix, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_matrix, dtype=torch.float32)), batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_matrix, dtype=torch.float32)), batch_size=batch_size * 2, shuffle=False)
    # model initialisation
    cardinalities = preprocessor.cardinalities
    input_dim = train_matrix.shape[1]
    continuous_dim = len(continuous_cols)
    model = MixedTabularVAE(input_dim, continuous_dim, cardinalities, la , latent_dim).to(device)
    criterion = TabVAELoss(continuous_dim, cardinalities, mmd_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ---MLflow Tracking & Execution ---
    mlflow.set_experiment("Mixed_Tabular_VAE_Synthesis")
    
    with mlflow.start_run(run_name=f"VAE_latent{latent_dim}_epochs{epochs}"):
        mlflow.log_params({
            "epochs": epochs, "batch_size": batch_size, "latent_dim": latent_dim,
            "Encoder_dims": la, "Decoder_dims": la[::-1], "MMD Weight": mmd_weight,
        })
        
        # Execute Training
        model, history = train_vae(model=model, train_loader=train_loader, val_loader=val_loader,criterion=criterion, optimizer=optimizer, epochs=epochs, device=device)
        '''
        # Log Training Metrics from History
        for epoch in range(len(history["train_loss"])):
            mlflow.log_metrics({
                "train_loss": history["train_loss"][epoch],
                "val_loss": history["val_loss"][epoch]
            }, step=epoch + 1)
        '''
        # --- Plotting Loss Curves ---
        print("Plotting Loss Curves...")
        fig = plt.figure(figsize=(18, 5))
        
        # Total Loss
        plt.subplot(1, 3, 1)
        plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Total Loss", color='blue')
        plt.plot(range(1, epochs + 1), history["val_loss"], label="Validation Total Loss", color='orange')
        plt.title("Total Objective Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Recon Loss
        plt.subplot(1, 3, 2)
        plt.plot(range(1, epochs + 1), history["train_recon"], label="Train Recon Loss", color='green')
        plt.plot(range(1, epochs + 1), history["val_recon"], label="Validation Recon Loss", color='red')
        plt.title("Reconstruction Accuracy")
        plt.xlabel("Epochs")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # MMD & Tau
        ax1 = plt.subplot(1, 3, 3) 
        p1, = ax1.plot(range(1, epochs + 1), history["train_mmd"], label="Train MMD Loss", color='purple')
        p2, = ax1.plot(range(1, epochs + 1), history["val_mmd"], label="Validation MMD Loss", color='brown')
        ax1.set_title("Latent Alignment & Tau Decay")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("MMD Loss", color='purple')
        ax1.tick_params(axis='y', labelcolor='purple')
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2 = ax1.twinx()  
        p3, = ax2.plot(range(1, epochs + 1), history["tau"], label="Tau Temperature", color='darkgray', linestyle=':')
        ax2.set_ylabel("Tau (Temperature)", color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        lines = [p1, p2, p3]
        ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
        
        os.makedirs("data/plots", exist_ok=True)
        plot_path = "data/plots/loss_curves.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        mlflow.log_artifact(plot_path, artifact_path="plots")

        #-----------------Diffusion Block-------------------#

        model.load_state_dict(torch.load(weight_path))#get best weights from training+validation
        model.eval()#freeze weights
        for p in model.parameters():
            p.requires_grad = False
        train_tensor = torch.tensor(
        train_matrix,
        dtype=torch.float32,
        device=device)
        with torch.no_grad():
            mu, logvar = model.encoder(train_tensor)
            latent_dataset = model.reparameterize(mu, logvar)

        latent_loader = DataLoader(
        TensorDataset(latent_dataset),
        batch_size=batch_size,
        shuffle=True
        )
        diffusion_hidden_dims = config["architecture"]["diffusion_hidden_dims"]
        time_embedding_dim = config["architecture"]["time_embedding_dim"]
        num_diffusion_steps = config["diffusion"]["num_diffusion_steps"]
        beta_start = config["diffusion"]["beta_start"]
        beta_end = config["diffusion"]["beta_end"]
        diffusion_lr = config["diffusion"]["diffusion_lr"]
        
        diffusion_epochs = config["diffusion"]["diffusion_epochs"]

        diffusion_model = DiffusionMLP(
            latent_dim=latent_dim,
            hidden_dims=diffusion_hidden_dims,
            time_embedding_dim=time_embedding_dim
        ).to(device)

        beta_start = 1e-4
        beta_end = 2e-2
        #diffusion noise scheduler
        noise_scheduler = DiffusionScheduler(
            num_timesteps=num_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule="linear",
            device=device
        )

        diffusion_optimizer = torch.optim.AdamW(
            diffusion_model.parameters(),
            lr=diffusion_lr
        )

        # --------------------------------------------------------
        # Log Hyperparameters
        # --------------------------------------------------------
        mlflow.log_param("diffusion_hidden_dims", str(diffusion_hidden_dims))
        mlflow.log_param("time_embedding_dim", time_embedding_dim)
        mlflow.log_params(diffusion_model.get_config())
        mlflow.log_params(noise_scheduler.get_config())
        mlflow.log_param("num_diffusion_steps", num_diffusion_steps)

        # --------------------------------------------------------
        # Train Diffusion Model
        # --------------------------------------------------------

        diffusion_model, diffusion_loss_history = train_diffusion(
            model=diffusion_model,
            scheduler=noise_scheduler,
            latent_loader=latent_loader,
            optimizer=diffusion_optimizer,
            epochs=diffusion_epochs,
            device=device
        )

        diffusion_model.eval()
        for p in diffusion_model.parameters():
            p.requires_grad = False

        gen_size = len(test_df)

        #######################
        sampler = DiffusionSampler(
            diffusion_model=diffusion_model,
            scheduler=noise_scheduler,
            device=device,
        )

        latent_samples = sampler.sample(
            num_samples=gen_size,
            latent_dim=latent_dim,
        )

        synthetic_raw = (
            model.decoder(latent_samples)
            .cpu()
            .numpy()
        )

        # --- Final Evaluation on Unseen Test Dataset ---
        print("Evaluation against unseen test dataset:")
        
        postprocessor = TabularDataPostprocessor(preprocessor)
        synthetic_df_export = postprocessor.inverse_transform(synthetic_raw)
        
        # Save synthetic data to CSV (fixed file extension)
        output_data_path = "data/synthetic_data_export.csv"
        synthetic_df_export.to_csv(output_data_path, index=False)
        
        gen_metrics = evaluate_generator_performance(real_df=test_df, synthetic_df=synthetic_df_export, k=5)
        
        print("\n Final Generative Performance (Unseen Data):")
        print(f"   Shape Error:     {gen_metrics['shape_error_pct']:.2f}%")
        print(f"   Trend Error:     {gen_metrics['trend_error_pct']:.2f}%")
        print(f"   Alpha-Precision: {gen_metrics['alpha_precision_pct']:.2f}%")
        print(f"   Beta-Recall:     {gen_metrics['beta_recall_pct']:.2f}%")

        mlflow.log_metrics({
            "test_shape_error": gen_metrics["shape_error_pct"],
            "test_trend_error": gen_metrics["trend_error_pct"],
            "test_alpha_precision": gen_metrics["alpha_precision_pct"],
            "test_beta_recall": gen_metrics["beta_recall_pct"]
        })
            
        mlflow.log_artifact(output_data_path, artifact_path="synthetic_data")
        mlflow.pytorch.log_model(model, "vae_model")
        print("Pipeline complete.")
   
