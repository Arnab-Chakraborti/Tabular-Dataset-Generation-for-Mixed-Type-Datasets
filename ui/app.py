import streamlit as st
import yaml
import ast
import os
import sys

# Define the path to your config file
CONFIG_PATH = "../configs/default_params.yaml"

st.set_page_config(page_title="Tabular Generation Config", layout="wide")

st.title("Tabular Model Hyperparameter UI")
st.markdown("Adjust the parameters below. Clicking the button will update your YAML configuration.")

DATASET_MAP = {
    "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "default": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "magic": "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip",
    "shoppers": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
    "beijing": "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip",
    "news": "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip",
    "diabetes": "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
}
AVAILABLE_DATASETS = list(DATASET_MAP.keys())
# Helper function to parse list inputs safely
def parse_list(list_str):
    try:
        return ast.literal_eval(list_str)
    except:
        return list_str

# ---------------------------------------------------------
# UI Layout: Organizing parameters into logical columns/tabs
# ---------------------------------------------------------
st.header("Dataset Selection")
selected_dataset = st.selectbox("Choose the dataset to train on:", options= AVAILABLE_DATASETS)
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Training & Core")
    epochs = st.number_input("VAE epochs", min_value=1, value=50, step=100)
    lr = st.number_input("VAE Learning Rate", min_value=0.0001, value=0.001, step=0.00005, format="%.5f")
    batch_size = st.number_input("batch_size", min_value=1, value=256, step=32)
    latent_dim = st.number_input("latent_dim", min_value=1, value=64, step=8)
    dropout = st.number_input("dropout", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    activation = st.selectbox("activation", options=["SiLU", "ReLU", "GELU", "LeakyReLU"])
    use_layernorm = st.checkbox("use_layernorm", value=True)

with col2:
    st.header("VAE + Diffusion Architecture")
    # For lists, we use text inputs and parse them back to Python lists
    mmd_weight = st.number_input("MMD Weight", min_value=0.0, value=500.0, step=10.0)
    encoder_dims = st.text_input("Encoder_dims", value="[512, 256, 128]")
    decoder_dims = st.text_input("Decoder_dims", value="[128, 256, 512]")
    diffusion_hidden_dims = st.text_input("diffusion_hidden_dims", value="[512, 512, 256]")
    time_embedding_dim = st.number_input("time_embedding_dim", min_value=1, value=128, step=32)

with col3:
    st.header("Diffusion & Loss")
    hidden_dims = st.text_input("hidden_dims", value="[512, 512, 256]")
    prediction_type = st.selectbox("prediction_type", options=["noise", "sample", "v_prediction"])
    num_diffusion_steps = st.number_input("num_diffusion_steps", min_value=1, value=100, step=10)
    diffusion_lr = st.number_input("Diffusion Learning Rate", min_value=0.0001, value=0.001, step=0.00005, format="%.5f")
    diffusion_epochs = st.number_input("diffusion Epochs" , min_value=100, value=200, step=100)

    
    st.subheader("Noise Schedule")
    beta_start = st.number_input("beta_start", format="%.5f", value=0.0001, step=0.0001)
    beta_end = st.number_input("beta_end", format="%.4f", value=0.02, step=0.001)
    schedule = st.selectbox("schedule", options=["linear", "cosine", "quadratic"])

st.divider()

if st.button("Save Configuration & Trigger Training", type="primary"):
    
    # Construct the configuration dictionary
    config_data = {
        "data": {
            "URL": DATASET_MAP[selected_dataset],
            "name": selected_dataset
        },
        
        "vae_training": {
            "MMD Weight": mmd_weight,
            "VAE epochs": epochs,
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "dropout": dropout,
            "lr": lr
        },
        "architecture": {
            "Encoder_dims": parse_list(encoder_dims),
            "Decoder_dims": parse_list(decoder_dims),
            "hidden_dims": parse_list(hidden_dims),
            "diffusion_hidden_dims": parse_list(diffusion_hidden_dims),
            "time_embedding_dim": time_embedding_dim,
            "activation": activation,
            "use_layernorm": use_layernorm,
        },
        "diffusion": {
            "prediction_type": prediction_type,
            "num_diffusion_steps": num_diffusion_steps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "schedule": schedule,
            "diffusion_epochs": diffusion_epochs,
            "diffusion_lr": diffusion_lr
        }
    }

    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)

    # Write the dictionary to the YAML file
    with open(CONFIG_PATH, 'w') as yaml_file:
        yaml.dump(config_data, yaml_file, default_flow_style=False, sort_keys=False)
    
    st.success(f"✅ Configuration successfully saved to `{CONFIG_PATH}`!")
    st.json(config_data) # Display the saved payload to the user for visual confirmation
    
    # ---------------------------------------------------------
    # OPTIONAL: Trigger the training script automatically
    # ---------------------------------------------------------
    import subprocess
    st.info("Starting training run...")
    subprocess.Popen(["python3", "../alt_main.py", "--config", CONFIG_PATH])
