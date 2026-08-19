import os
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import argparse
import yaml
import seaborn as sns
import scipy.stats as ss
import mlflow
import mlflow.pytorch
from src.utils.sampler import DiffusionSampler
from src.data_processing.data_preprocessing import objectify
from src.models.diffusion import DiffusionMLP
from src.trainers.diffusion_trainer import train_diffusion
from src.utils.schedulers import DiffusionScheduler
from src.models.vae import MixedTabularVAE
from src.data_processing.data_preprocessing import TabularDataPreprocessor
from src.data_processing.data_postprocessing import TabularDataPostprocessor
from src.utils.loss import TabVAELoss
from src.Evaluations.eval_metrics import evaluate_generator_performance
import gspread
from google.oauth2.service_account import Credentials
import datetime
import requests
import zipfile
import io
import time
import sys
import numpy as np


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denom = min((kcorr-1), (rcorr-1))
    return np.sqrt(phi2corr / denom) if denom > 0 else 0.0

def compute_categorical_corr(df, cat_cols):
    corr = pd.DataFrame(index=cat_cols, columns=cat_cols)
    for col1 in cat_cols:
        for col2 in cat_cols:
            if col1 == col2:
                corr.loc[col1, col2] = 1.0
            else:
                corr.loc[col1, col2] = cramers_v(df[col1], df[col2])
    return corr.astype(float)

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
        t_0 = time.time()
        model.train()
        train_loss, train_recon, mmd_loss = 0.0, 0.0, 0.0
        cos_inner = math.pi * (epoch - 1) / epochs
        current_tau = tau_min + 0.5 * (tau_max - tau_min) * (1.0 + math.cos(cos_inner))
        model.decoder.tau = current_tau
        history["tau"].append(current_tau)
        c = 0
        active_weight_epoch = 0.0
        
        for (batch_x,) in train_loader:
            t0 = time.time()
            c += 1
            batch_x = batch_x.to(device)
            t1 = time.time()
            recon_x, z = model(batch_x)
            t2 = time.time()
            
            # --- Unpack all 4 variables from the updated TabVAELoss ---
            loss, recon_loss, train_mmd, active_weight = criterion(recon_x, batch_x, z, current_epoch=epoch)
            active_weight_epoch = active_weight
            
            t3 = time.time()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            t4 = time.time()
            
            if c == 80:
                print(f" Actually training on: {device.upper()}")
                print(f"\n--- MICRO-PROFILER RESULTS Batch {c} ---")
                print(f"1. CPU -> GPU Transfer:  {(t1-t0)*1000:.2f} ms")
                print(f"2. Model Forward Pass:   {(t2-t1)*1000:.2f} ms")
                print(f"3. Loss Calculation:     {(t3-t2)*1000:.2f} ms")
                print(f"4. Backward Pass & Step: {(t4-t3)*1000:.2f} ms")
                print("------------------------------------------\n")
            
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
                
                loss, recon_loss, batch_mmd, _ = criterion(recon_x, batch_x, z, current_epoch=epoch)
                
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
            print(f"Epoch [{epoch:03d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | MMD Loss: {avg_train_mmd:.4f} | MMD Wgt: {active_weight_epoch:.2f}")
            
        t = time.time()
        if epoch % 20 == 1:
            print(f"Wall clock time per epoch: {(t - t_0) * 1000:.2f} ms")
            
    return model, history

def download_and_extract(url, extract_folder):
    os.makedirs(extract_folder, exist_ok=True)
    print(f"Downloading data from {url}...")
    response = requests.get(url)
    response.raise_for_status() 
    
    print("Extracting files...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(extract_folder)
        print(f"Extraction complete. Files saved to: {extract_folder}")

def find_largest_data_file(extract_folder):
    valid_extensions = ['.csv', '.data', '.xls', '.xlsx', '.txt']
    largest_file = None
    max_size = -1
    
    print(f"Scanning '{extract_folder}' for data files...")
    for root, _, files in os.walk(extract_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                if file_size > max_size:
                    max_size = file_size
                    largest_file = file_path
                    
    if not largest_file:
        raise FileNotFoundError(f"No valid data files {valid_extensions} found in {extract_folder}")
        
    print(f"Selected primary dataset file: {os.path.basename(largest_file)}")
    return largest_file

if __name__ == "__main__":
    physical_cores = max(1, os.cpu_count() // 2)
    torch.set_num_threads(physical_cores)
    print(f"CPU Optimization: Restricted PyTorch to {physical_cores} threads.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_params.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    # ==========================================
    # 1. PARSE GENERAL CONFIGS 
    # ==========================================
    url = config["dataset"]["url"]
    name = config["dataset"]["name"]
    epochs = config["vae"]["epochs"]
    batch_size = config["vae"]["batch_size"]
    lr = config["vae"]["learning_rate"]
    
    raw_mmd_dict = config["vae"]["mmd"]

    val_split = 0.2
    weight_path = "best_vae_weights.pt"
    
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', name)
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"Dataset '{name}' missing. Initiating download...")
        download_and_extract(url, DATA_DIR)

    DATA_PATH = find_largest_data_file(DATA_DIR)
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training engine initialised on device: {device}")
    print(f"Raw Tabular data loaded from {DATA_PATH}")
    
    if DATA_PATH.lower().endswith(('.xls', '.xlsx')):
        raw_df = pd.read_excel(DATA_PATH)
    elif DATA_PATH.lower().endswith(('.csv', '.data', '.txt')):
        raw_df = pd.read_csv(DATA_PATH, skipinitialspace=True)
    else:
        raw_df = pd.read_csv(DATA_PATH, skipinitialspace=True)

    raw_df= objectify(raw_df)
        
    train_df, temp_df = train_test_split(raw_df, test_size=val_split, shuffle=True, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, shuffle=True, random_state=42)
    print(f"Split: {len(train_df)} Train | {len(val_df)} Val | {len(test_df)} Test")
    
    continuous_cols = raw_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = raw_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    print(f"numerical cols: {len(continuous_cols)}|categorical cols: {len(categorical_cols)}")
    
    preprocessor = TabularDataPreprocessor(
        continuous_cols=continuous_cols, 
        categorical_cols=categorical_cols,
        continuous_scaler="quantile",  
        clip_outliers=False,        
        impute_missing=True
    )
    
    train_matrix = preprocessor.fit_transform(train_df)
    val_matrix = preprocessor.transform(val_df)
    test_matrix = preprocessor.transform(test_df)  
    
    train_loader = DataLoader(TensorDataset(torch.tensor(train_matrix, dtype=torch.float32)), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, persistent_workers=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_matrix, dtype=torch.float32)), batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_matrix, dtype=torch.float32)), batch_size=batch_size * 2, shuffle=False)
    
    cardinalities = preprocessor.cardinalities
    input_dim = train_matrix.shape[1]
    print("Total Features:", input_dim, len(preprocessor.continuous_cols))
    continuous_dim = len(continuous_cols)
    
    latent_mode = config["vae"]["latent"]["mode"]
    
    if latent_mode == "adaptive":
        heuristic = config["vae"]["latent"].get("heuristic", "google").lower()
        
        if heuristic == "fh":
            latent_dim, s = preprocessor.getheuristic_fast()
        else: 
            latent_dim, s = preprocessor.getheuristic_google()
            
        lower = 2 ** math.floor(math.log2(latent_dim))
        upper = 2 ** math.ceil(math.log2(latent_dim))
        latent_dim = lower if (latent_dim - lower) < (upper - latent_dim) else upper
        print(f"Adaptive Mode Active: {latent_dim} (Heuristic: {s})")
        
        la = [4 * latent_dim, 2 * latent_dim, latent_dim]
        decoder_la = la[::-1]
    else:
        s = "Custom"
        latent_dim = config["vae"]["latent"]["latent_dim"]
        la = config["vae"]["latent"]["encoder_dims"]
        decoder_la = config["vae"]["latent"]["decoder_dims"]

    model = MixedTabularVAE(input_dim, continuous_dim, cardinalities, la, latent_dim).to(device)
    
    criterion = TabVAELoss(continuous_dim, cardinalities, mmd_policy=raw_mmd_dict, total_epochs=epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # ---MLflow Tracking & Execution ---
    mlflow.set_tracking_uri(f"file://{os.path.abspath('./mlruns')}")
    mlflow.set_experiment("Mixed_Tabular_VAE_Diffusion")
    
    with mlflow.start_run(run_name=f"VAE_latent{latent_dim}_epochs{epochs}") as run:
        
        run_id = run.info.run_id 
        
        mlflow.log_params({
            # ...
        })
        skip_vae= True
        # Execute Training
        if skip_vae==False:
            model, history = train_vae(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, epochs=epochs, device=device)
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

        model.load_state_dict(torch.load(weight_path))
        model.eval()
        print("\nGenerating VAE-only samples...")

        with torch.no_grad():

            # Option 1 (true VAE prior)
            z_vae = torch.randn(
                len(test_df),
                latent_dim,
                device=device
            )

            synthetic_vae_raw = (
                model.decoder(z_vae)
                .cpu()
                .numpy()
            )

        postprocessor = TabularDataPostprocessor(preprocessor)

        synthetic_vae_df = postprocessor.inverse_transform(
            synthetic_vae_raw
        )

        vae_metrics = evaluate_generator_performance(
            real_df=test_df,
            synthetic_df=synthetic_vae_df,
            k=5
        )

        print("\n===== PURE VAE =====")
        print(f"Shape Error     : {vae_metrics['shape_error_pct']:.2f}")
        print(f"Trend Error     : {vae_metrics['trend_error_pct']:.2f}")
        print(f"Alpha Precision : {vae_metrics['alpha_precision_pct']:.2f}")
        print(f"Beta Recall     : {vae_metrics['beta_recall_pct']:.2f}")
        
        for p in model.parameters():
            p.requires_grad = False
            
        train_tensor = torch.tensor(
            train_matrix,
            dtype=torch.float32,
            device=device
        )
        with torch.no_grad():
            mu, logvar = model.encoder(train_tensor)
            latent_dataset = model.reparameterize(mu, logvar)

        latent_loader = DataLoader(
            TensorDataset(latent_dataset),
            batch_size=batch_size,
            shuffle=True
        )
        
        # ==========================================
        # 3. PARSE DIFFUSION CONFIGS (UPDATED FOR CONTINUOUS TIME)
        # ==========================================
        diffusion_epochs = config["diffusion"]["epochs"]
        diffusion_lr = config["diffusion"]["learning_rate"]
        num_diffusion_steps = config["diffusion"]["num_diffusion_steps"]
        diffusion_hidden_dims = config["diffusion"]["architecture"]["hidden_dims"]
        time_embedding_dim = config["diffusion"]["time_conditioning"]["embedding_dim"]
        
        # Parse new boundaries for the continuous power-mean schedule
        noise_schedule = config["diffusion"]["noise_schedule"]
        sigma_min = noise_schedule.get("sigma_min", 0.002)
        sigma_max = noise_schedule.get("sigma_max", 80.0)
        schedule = "continuous_power_mean"

        diffusion_model = DiffusionMLP(
            latent_dim=latent_dim,
            hidden_dims=diffusion_hidden_dims,
            time_embedding_dim=time_embedding_dim
        ).to(device)

        # Initialize the updated continuous scheduler
        noise_scheduler = DiffusionScheduler(
            beta_start=sigma_min,
            beta_end=sigma_max,
            device=device
        )

        diffusion_optimizer = torch.optim.AdamW(
            diffusion_model.parameters(),
            lr=diffusion_lr
        )

        mlflow.log_param("diffusion_hidden_dims", str(diffusion_hidden_dims))
        mlflow.log_param("time_embedding_dim", time_embedding_dim)
        mlflow.log_params(diffusion_model.get_config())
        mlflow.log_params(noise_scheduler.get_config())
        mlflow.log_param("num_diffusion_steps", num_diffusion_steps)

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

        sampler = DiffusionSampler(
            diffusion_model=diffusion_model,
            scheduler=noise_scheduler,
            device=device,
        )
        
        # Pass num_steps dynamically to the continuous stochastic sampler
        latent_samples = sampler.sample(
            num_samples=gen_size,
            latent_dim=latent_dim,
            num_steps=num_diffusion_steps
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
        
        output_data_path = "data/synthetic_data_export.csv"
        synthetic_df_export.to_csv(output_data_path, index=False)

        print("Generating evaluation plots...")
        plot_dir = f"data/plots/{run_id}"
        os.makedirs(plot_dir, exist_ok=True)
        
        # A. Numerical Distributions (KDE)
        for col in continuous_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.kdeplot(test_df[col], fill=True, label="Real", color="#3b82f6", ax=ax, alpha=0.5)
            sns.kdeplot(synthetic_df_export[col], fill=True, label="Synthetic", color="#f59e0b", ax=ax, alpha=0.5)
            ax.set_title(f"Numerical Distribution: {col}")
            ax.legend()
            fig.savefig(os.path.join(plot_dir, f"dist_num_{col}.png"), bbox_inches="tight")
            plt.close(fig)
            
        # B. Categorical Distributions (Bar Charts)
        for col in categorical_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            real_counts = test_df[col].value_counts(normalize=True).rename("Real")
            fake_counts = synthetic_df_export[col].value_counts(normalize=True).rename("Synthetic")
            comp_df = pd.concat([real_counts, fake_counts], axis=1).fillna(0)
            comp_df.plot(kind="bar", ax=ax, color=["#3b82f6", "#f59e0b"], alpha=0.8)
            ax.set_title(f"Categorical Distribution: {col}")
            ax.set_ylabel("Proportion")
            plt.xticks(rotation=45, ha='right')
            fig.savefig(os.path.join(plot_dir, f"dist_cat_{col}.png"), bbox_inches="tight")
            plt.close(fig)
            
        # C. Numerical Correlation Heatmap
        if len(continuous_cols) > 1:
            fig, axes = plt.subplots(1, 3, figsize=(24, 6))
            corr_real = test_df[continuous_cols].corr()
            corr_fake = synthetic_df_export[continuous_cols].corr()
            corr_diff = corr_real - corr_fake
            
            sns.heatmap(corr_real, ax=axes[0], cmap="coolwarm", vmin=-1, vmax=1, annot=False)
            axes[0].set_title("Real Data Correlation (Numerical)")
            sns.heatmap(corr_fake, ax=axes[1], cmap="coolwarm", vmin=-1, vmax=1, annot=False)
            axes[1].set_title("Synthetic Data Correlation (Numerical)")
            sns.heatmap(corr_diff, ax=axes[2], cmap="RdBu", vmin=-1, vmax=1, annot=False)
            axes[2].set_title("Difference (Real - Fake)")
            
            fig.savefig(os.path.join(plot_dir, "heatmap_numerical.png"), bbox_inches="tight")
            plt.close(fig)
            
        # D. Categorical Correlation Heatmap (Cramer's V)
        if len(categorical_cols) > 1:
            fig, axes = plt.subplots(1, 3, figsize=(24, 6))
            cat_corr_real = compute_categorical_corr(test_df, categorical_cols)
            cat_corr_fake = compute_categorical_corr(synthetic_df_export, categorical_cols)
            cat_corr_diff = cat_corr_real - cat_corr_fake
            
            sns.heatmap(cat_corr_real, ax=axes[0], cmap="viridis", vmin=0, vmax=1, annot=False)
            axes[0].set_title("Real Data Cramer's V (Categorical)")
            sns.heatmap(cat_corr_fake, ax=axes[1], cmap="viridis", vmin=0, vmax=1, annot=False)
            axes[1].set_title("Synthetic Data Cramer's V (Categorical)")
            sns.heatmap(cat_corr_diff, ax=axes[2], cmap="RdBu", vmin=-1, vmax=1, annot=False)
            axes[2].set_title("Difference (Real - Fake)")
            
            fig.savefig(os.path.join(plot_dir, "heatmap_categorical.png"), bbox_inches="tight")
            plt.close(fig)

        # Bulk log everything as artifacts explicitly into "evaluation_plots" path
        mlflow.log_artifacts(plot_dir, artifact_path="evaluation_plots")
        print(f"All visualizations saved and logged to MLflow under run: {run_id}")

        # --- Final Metric Evaluation & Data Export ---
        output_data_path = "data/synthetic_data_export.csv"
        synthetic_df_export.to_csv(output_data_path, index=False)
        gen_metrics = evaluate_generator_performance(real_df=test_df, synthetic_df=synthetic_df_export, k=5)
        
        mlflow.log_metrics({
            "test_shape_error": gen_metrics["shape_error_pct"],
            "test_trend_error": gen_metrics["trend_error_pct"],
            "test_alpha_precision": gen_metrics["alpha_precision_pct"],
            "test_beta_recall": gen_metrics["beta_recall_pct"]
        })
            
        mlflow.log_artifact(output_data_path, artifact_path="synthetic_data")
        mlflow.pytorch.log_model(model, "vae_model")
        
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
        print("Exporting metrics to Google Sheets...")
        
        try:
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
            cred_path = os.path.join(os.path.dirname(__file__), 'google credentials.json')
            creds = Credentials.from_service_account_file(cred_path, scopes=scopes)
            client = gspread.authorize(creds)

            sheet = client.open("VAE-MMD-Diff").sheet1
            run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            run_name = f"VAE_MMD_Diff_latent{latent_dim}_epochs{epochs}"
            num_cont = continuous_dim
            num_cat = len(cardinalities)
            latent_dim_h = str(latent_dim) + s
            encoder_dims_str = str(la)
            decoder_dims_str = str(decoder_la)
            diff_hidden_dims_str = str(diffusion_hidden_dims)

            row_data = [
                run_timestamp,
                run_name,
                name,
                num_cont,
                num_cat,
                epochs,
                batch_size,
                encoder_dims_str,
                decoder_dims_str,
                str(raw_mmd_dict), # Logs the exact stringified dictionary to Sheets
                latent_dim_h,
                lr,
                diff_hidden_dims_str,
                time_embedding_dim,
                num_diffusion_steps,
                diffusion_epochs,
                diffusion_lr,
                sigma_min,     # Logs sigma boundaries to Google sheets instead of discrete betas
                sigma_max,
                schedule,
                round(gen_metrics['shape_error_pct'], 3),
                round(gen_metrics['trend_error_pct'], 3),
                round(gen_metrics['alpha_precision_pct'], 3),
                round(gen_metrics['beta_recall_pct'], 3),
            ]

            sheet.insert_row(row_data, index=2, value_input_option='USER_ENTERED')
            print("Successfully logged run to Google Sheets.")
            
        except Exception as e:
            print(f"Failed to log to Google Sheets. Error: {e}")
