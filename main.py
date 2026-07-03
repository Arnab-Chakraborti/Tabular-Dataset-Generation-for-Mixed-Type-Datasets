import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from src.models.tabSyn import TabSynVAE
from src.models.tabSyn_diffusion import TabSynDenoisingMLP
from src.utils.loss import TabSynLossEngine, compute_diffusion_loss
from src.data_processing.data_preprocessing import TabularDataPreprocessor
from src.data_processing.data_postprocessing import TabularDataPostprocessor
from src.Evaluations.eval_metrics import evaluate_generator_performance


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'adult', 'adult.data')


# -------------------------
# Dataset
# -------------------------
class TabularDataset(Dataset):
    def __init__(self, data_matrix, num_dim, cat_dims):
        self.data = torch.tensor(data_matrix, dtype=torch.float32)
        self.num_dim = num_dim
        self.cat_dims = cat_dims

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        x_num = row[:self.num_dim]

        ohe_list = []
        labels = []
        start = self.num_dim

        for dim in self.cat_dims:
            ohe = row[start:start + dim]
            ohe_list.append(ohe)
            labels.append(torch.argmax(ohe))
            start += dim

        return x_num, ohe_list, torch.tensor(labels, dtype=torch.long)


def collate_fn(batch):
    x_nums = torch.stack([b[0] for b in batch])

    x_cat_ohes = []
    if len(batch[0][1]) > 0:
        for i in range(len(batch[0][1])):
            x_cat_ohes.append(torch.stack([b[1][i] for b in batch]))

    x_labels = torch.stack([b[2] for b in batch])
    return x_nums, x_cat_ohes, x_labels


# -------------------------
# Helpers
# -------------------------
def auto_detect_columns(df):
    cont = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return cont, cat


# -------------------------
# Main
# -------------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    torch.backends.cudnn.benchmark = True

    scaler = torch.amp.GradScaler()

    # -------------------------
    # Data
    # -------------------------
    df = pd.read_csv(DATA_PATH)
    cont_cols, cat_cols = auto_detect_columns(df)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    preprocessor = TabularDataPreprocessor(
        continuous_cols=cont_cols,
        categorical_cols=cat_cols
    )

    processed = preprocessor.fit_transform(train_df)

    num_numeric = len(preprocessor.continuous_cols)
    cat_cardinalities = preprocessor.cardinalities
    num_columns = num_numeric + len(cat_cardinalities)

    dataset = TabularDataset(processed, num_numeric, cat_cardinalities)

    train_loader = DataLoader(
        dataset,
        batch_size=1024,   # reduced from 4096 (important for stability)
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # -------------------------
    # VAE
    # -------------------------
    d_token = 4

    vae = TabSynVAE(
        num_numeric=num_numeric,
        cat_cardinalities=cat_cardinalities,
        d_token=d_token
    ).to(device)

    vae = torch.compile(vae)

    vae_optimizer = optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_engine = TabSynLossEngine(beta_max=0.01, beta_min=1e-5, lambda_decay=0.7, patience=5)

    # -------------------------
    # TRAIN VAE (AMP)
    # -------------------------
    vae_epochs = 200

    vae.train()
    for epoch in range(vae_epochs):
        total_loss = 0

        for x_num, x_cat_ohes, x_labels in train_loader:
            x_num = x_num.to(device, non_blocking=True)
            x_cat_ohes = [c.to(device, non_blocking=True) for c in x_cat_ohes]
            x_labels = x_labels.to(device, non_blocking=True)

            vae_optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                mu, logvar = vae.encode(x_num, None, x_cat_ohes)
                z = vae.reparameterize(mu, logvar)
                x_num_hat, x_cat_hat = vae.decode(z)

                loss, recon, kl = loss_engine.compute_vae_loss(
                    x_num, x_labels, x_num_hat, x_cat_hat, mu, logvar
                )

            scaler.scale(loss).backward()
            scaler.step(vae_optimizer)
            scaler.update()

            total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"VAE Epoch {epoch}: {total_loss/len(train_loader):.4f}")

    # -------------------------
    # Freeze VAE
    # -------------------------
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # -------------------------
    # PRECOMPUTE LATENT SPACE (CRITICAL SPEEDUP)
    # -------------------------
    print("Precomputing latent space...")

    Z_list = []

    with torch.no_grad():
        for x_num, x_cat_ohes, _ in train_loader:
            x_num = x_num.to(device)
            x_cat_ohes = [c.to(device) for c in x_cat_ohes]

            mu, logvar = vae.encode(x_num, None, x_cat_ohes)
            z = vae.reparameterize(mu, logvar)

            Z_list.append(z.flatten(start_dim=1))

    Z_dataset = torch.cat(Z_list, dim=0)

    # -------------------------
    # DIFFUSION MODEL
    # -------------------------
    diffusion_model = TabSynDenoisingMLP(
        num_columns=num_columns,
        d_token=d_token
    ).to(device)

    diffusion_model = torch.compile(diffusion_model)

    diff_optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-3, weight_decay=1e-4)

    diff_epochs = 500

    # -------------------------
    # DIFFUSION TRAINING (FAST)
    # -------------------------
    diffusion_model.train()

    for epoch in range(diff_epochs):
        perm = torch.randperm(Z_dataset.size(0), device=device)
        total_loss = 0

        for i in range(0, len(perm), 1024):
            idx = perm[i:i+1024]
            z_0 = Z_dataset[idx]

            diff_optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                loss = compute_diffusion_loss(diffusion_model, z_0, t_max=1.0)

            scaler.scale(loss).backward()
            scaler.step(diff_optimizer)
            scaler.update()

            total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"Diff Epoch {epoch}: {total_loss:.4f}")

    # -------------------------
    # SAMPLING
    # -------------------------
    diffusion_model.eval()

    gen_size = len(test_df)
    flat_dim = num_columns * d_token

    z_t = torch.randn(gen_size, flat_dim, device=device)

    steps = 20
    dt = 1.0 / steps

    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        for i in range(steps):
            t = torch.full((gen_size,), 1.0 - i * dt, device=device)
            eps = diffusion_model(z_t, t)
            z_t = z_t - eps * dt

        Z_clean = z_t.view(gen_size, num_columns, d_token)
        synth_num, synth_cat = vae.decode(Z_clean)

    # -------------------------
    # POSTPROCESS
    # -------------------------
    postprocessor = TabularDataPostprocessor(preprocessor)

    synth_num_np = synth_num.cpu().numpy() if synth_num is not None else np.empty((gen_size, 0))
    synth_cat_np = [x.cpu().numpy() for x in synth_cat]

    raw = np.hstack([synth_num_np] + synth_cat_np) if len(synth_cat_np) else synth_num_np

    synthetic_df = postprocessor.inverse_transform(raw)

    metrics = evaluate_generator_performance(test_df, synthetic_df, k=5)

    print("\nFinal Metrics:")
    for k, v in metrics.items():
        print(k, v)


if __name__ == "__main__":
    main()
