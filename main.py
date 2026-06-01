import os
import torch
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mlflow.pytorch
from torch.utils.data import DataLoader, TensorDataset
from src.models.vae import MixedTabularVAE
from src.data_processing.data_preprocessing import TabularDataPreprocessor
from src.data_processing.data_postprocessing import TabularDataPostprocessor
from src.models.vae import MixedTabularVAE
from src.utils.loss import TabVAELoss
from src.Evaluations.eval_metrics import evaluate_generator_performance

def train_vae(data_path: str, epochs: int=500, batch_size: int=256, latent_dim: int= 64, mmd_weight: float=1000.0, lr: float=1e-3, val_split: float=0.2):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training engine initialised on device: {device}")
    print(f"Raw Tabular data loaded from {data_path}")
    raw_df = pd.read_csv(data_path)
    train_df, temp_df= train_test_split(raw_df, test_size= val_split, shuffle= True, random_state=42)
    val_df, test_df= train_test_split(temp_df, test_size= 0.50, shuffle= True, random_state=42)

    print(f"Split: {len(train_df)} Train | {len(val_df)} Val | {len(test_df)} Test")
    continuous_cols=raw_df.select_dtypes(include=['float64','int64']).columns.tolist()
    categorical_cols=raw_df.select_dtypes(include=['object','category','bool']).columns.tolist()
    preprocessor = TabularDataPreprocessor(
        continuous_cols=continuous_cols, 
        categorical_cols=categorical_cols,
        continuous_scaler="standard",  
        clip_outliers=False,        
        impute_missing=True
    )
    train_matrix= preprocessor.fit_transform(train_df)
    val_matrix = preprocessor.transform(val_df)
    test_matrix= preprocessor.transform(test_df)  

    cardinalities= preprocessor.cardinalities
    input_dim = train_matrix.shape[1]
    continuous_dim=len(continuous_cols)
    train_loader= DataLoader(TensorDataset(torch.tensor(train_matrix, dtype= torch.float32)), batch_size=batch_size, shuffle= True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_matrix, dtype=torch.float32)), batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_matrix, dtype=torch.float32)), batch_size=batch_size * 2, shuffle=False)

    mlflow.set_experiment("Mixed_Tabular_VAE_Synthesis")
    history= {"train_loss": [], "val_loss": [], "train_recon" : [], "val_recon" : [], "train_mmd" :[], "val_mmd" :[], "tau": []}
    weight_path = "best_vae_weights.pt"
    with mlflow.start_run(run_name=f"VAE_latent{latent_dim}_epochs{epochs}"):
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "latent_dim": latent_dim,
            "train_size": len(train_df), "val_size": len(val_df), "test_size": len(test_df)})
        model = MixedTabularVAE(input_dim, continuous_dim, cardinalities, [256, 128, 64], latent_dim).to(device)
        criterion=TabVAELoss(continuous_dim,cardinalities,mmd_weight)
        optimizer= torch.optim.AdamW(model.parameters(),lr=lr, weight_decay=1e-4)
        best_val_loss=float('inf')
        if True:
            print("Starting Optimization:") # helps identify the starting of the loop for debugging
            for epoch in range(1,epochs+1):
                model.train()
                train_loss, train_recon,mmd_loss= 0.0, 0.0, 0.0
                current_tau= max(0.1, 1.0-(epoch/epochs)*0.9) # baseline tau=0.1
                model.decoder.tau= current_tau
                history["tau"].append(current_tau)
                

                for (batch_x,) in train_loader:
                    batch_x= batch_x.to(device)
                    recon_x,z= model.forward(batch_x)
                    loss,recon_loss,train_mmd=criterion(recon_x,batch_x,z)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss+=loss.item()
                    train_recon+=recon_loss.item()
                    mmd_loss+=train_mmd.item()

                avg_train_loss =train_loss/len(train_loader)
                avg_train_recon= train_recon/len(train_loader)
                avg_train_mmd=mmd_loss/len(train_loader)

                # Validation

                model.eval()
                val_loss, val_recon,val_mmd=0.0, 0.0, 0.0
                with torch.no_grad():
                    for (batch_x,) in val_loader:
                        batch_x = batch_x.to(device)
                        recon_x,z = model.forward(batch_x)
                        loss, recon_loss, mmd_loss=criterion(recon_x,batch_x,z)
                        val_loss+=loss.item()
                        val_recon+=recon_loss.item()
                        val_mmd+=mmd_loss.item()
                avg_val_loss = val_loss/len(val_loader)
                avg_val_recon = val_recon/len(val_loader)
                avg_val_mmd = val_mmd/len(val_loader)
                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(avg_val_loss)
                history["train_recon"].append(avg_train_recon)
                history["val_recon"].append(avg_val_recon)
                history["train_mmd"].append(avg_train_mmd)
                history["val_mmd"].append(avg_val_mmd)

                if avg_val_loss<best_val_loss:
                    best_val_loss=avg_val_loss
                    torch.save(model.state_dict(),weight_path)

                mlflow.log_metrics({"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step=epoch)
                if epoch % 10 == 0 or epoch == 1:
                    print(f"Epoch [{epoch:03d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | MMD Loss: {avg_train_mmd}")

            print("Loss Curves:")
            fig = plt.figure(figsize=(18, 5))
            plt.subplot(1, 3, 1)
            plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Total Loss", color='blue')
            plt.plot(range(1, epochs + 1), history["val_loss"], label="Validation Total Loss", color='orange')
            plt.title("Total Objective Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.subplot(1, 3, 2)
            plt.plot(range(1, epochs + 1), history["train_recon"], label="Train Recon Loss", color='green')
            plt.plot(range(1, epochs + 1), history["val_recon"], label="Validation Recon Loss", color='red')
            plt.title("Reconstruction Accuracy")
            plt.xlabel("Epochs")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

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
            
            plt.grid(True, linestyle='--', alpha=0.7)

            os.makedirs("data/plots", exist_ok=True)
            plot_path = "data/plots/loss_curves.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close(fig)
            mlflow.log_artifact(plot_path, artifact_path="plots")

        print("Evaluation against unseen test dataset:")
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        model.eval()
        gen_size=len(test_df)
        synthetic_raw = model.generate(gen_size,device=device).cpu().numpy()
        postprocessor=TabularDataPostprocessor(preprocessor)
        synthetic_df_export = postprocessor.inverse_transform(synthetic_raw)
        output_path = "data/plots/loss_curves.png"
        synthetic_df_export.to_csv(output_path, index=False)
        gen_metrics = evaluate_generator_performance(real_df=test_df,synthetic_df=synthetic_df_export, k=5)
        
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
            
        mlflow.log_artifact(output_path, artifact_path="synthetic_data")
        mlflow.pytorch.log_model(model, "vae_model")
        print(f"Pipeline complete.")

        return model


if __name__ == "__main__":
    target_data = "/Users/arnabchakraborti/tabular/tabular_generation_project/data/adult/adult.data" 
    if os.path.exists(target_data):
        train_vae(data_path=target_data, epochs=100)
    else:
        print(f"Error: Could not find '{target_data}'.")




























        
        
