import streamlit as st
import yaml
import ast
import os
import subprocess
from architecture import render_architecture
import html
import sys
import mlflow
from mlflow.tracking import MlflowClient
from PIL import Image
import tempfile
import pandas as pd

if "config_saved" not in st.session_state:
    st.session_state.config_saved = False
CONFIG_PATH = "../configs/default_params.yaml"

st.set_page_config(
    page_title="Tabular Generation Config",
    layout="wide"
)

st.title("Tabular Model Hyperparameter UI")
st.markdown(
    "Adjust the parameters below. Clicking the button will update your YAML configuration."
)

DATASET_MAP = {
    "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "default": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "magic": "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip",
    "shoppers": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
    "beijing": "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip",
    "news": "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip",
    "diabetes": "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
}

AVAILABLE_DATASETS = list(DATASET_MAP.keys()) + ["custom"]
def parse_list(value):
    try:
        return ast.literal_eval(value)
    except Exception:
        return value

@st.cache_data(ttl=10)
def fetch_mlflow_images(tracking_uri, run_id, path=""):
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        artifacts = client.list_artifacts(run_id, path)
    except Exception:
        return []
    
    image_paths = []
    for a in artifacts:
        if a.is_dir:
            image_paths.extend(fetch_mlflow_images(tracking_uri, run_id, a.path))
        elif a.path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(a.path)
    return image_paths

def render_image_grid(image_paths, client, run_id, title):
    if not image_paths:
        return
    st.markdown(f"#### {title}")
    cols = st.columns(2)
    for i, img_path in enumerate(image_paths):
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = client.download_artifacts(run_id, img_path, tmp_dir)
            img = Image.open(local_path)
            raw_name = img_path.split('/')[-1]
            clean_caption = raw_name.replace('.png', '').replace('_', ' ').title()
            
            with cols[i % 2]:
                st.image(img, caption=clean_caption, width="stretch")
                
st.header("Dataset Selection")

selected_dataset = st.selectbox(
    "Choose the dataset to train on:",
    AVAILABLE_DATASETS
)
if selected_dataset == "custom":
    uploaded_file = st.file_uploader("Upload your custom CSV dataset", type=["csv", "txt", "data"])
    if uploaded_file is not None:
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        custom_dir = os.path.join(ROOT_DIR, "data", "custom")
        os.makedirs(custom_dir, exist_ok=True)
        
        for f in os.listdir(custom_dir):
            os.remove(os.path.join(custom_dir, f))
            
        custom_file_path = os.path.join(custom_dir, uploaded_file.name)
        with open(custom_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success(f"Custom dataset '{uploaded_file.name}' saved and ready for training!")
       
st.divider()

st.header("Context Injection (Optional)")
st.markdown("Select columns. These will be used strictly to steer the Generation")

context_columns_list = []
if selected_dataset == "custom" and 'uploaded_file' in locals() and uploaded_file is not None:
    df_preview = pd.read_csv(custom_file_path, nrows=0) 
    context_columns_list = st.multiselect(
        "Select Conditional Context Columns:",
        options=df_preview.columns.tolist(),
        help="Recommended: Target classification labels (e.g., 'Class') or key demographics."
    )
else:
    context_cols_input = st.text_input(
        "Context Columns (Comma-separated)",
        placeholder="e.g., y, job, marital",
        help="Ensure these match the exact column headers in your dataset."
    )
    context_columns_list = [c.strip() for c in context_cols_input.split(",") if c.strip()]

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "VAE",
        "Diffusion",
        "Training",
        "Architecture",
        "Model Performance Analytics",
        "Generation and Inference",
    ]
)
with tab3:
    st.header("Training")
    epochs = st.number_input("VAE Epochs", min_value=1, value=1000, step=100)
    diffusion_epochs = st.number_input("Diffusion Epochs", min_value=1, value=2000, step=100)
    vae_lr = st.number_input("VAE Learning Rate", min_value=0.00001, value=0.001, step=0.0001, format="%.5f")
    diff_lr = st.number_input("Diffusion Learning Rate", min_value=0.00001, value=0.001, step=0.0001, format="%.5f")
    batch_size = st.number_input("Batch Size", min_value=1, value=256, step=32)
    
with tab1:
    st.header("VAE Setup")
    st.subheader("Latent Representation and Encoder/Decoder")
    latent_mode = st.radio("Latent Dimension Strategy", ["Adaptive", "Custom"], horizontal=True)

    if latent_mode == "Adaptive":
        latent_heuristic = st.selectbox("Heuristic", ["Google", "FastAI", "FH"])
        latent_dim = None
        encoder_dims = None
        decoder_dims = None
        st.info("Latent dimension and encoder/decoder sizes will be computed automatically.")
    else:
        latent_heuristic = None
        latent_dim = st.number_input("Latent Dimension", min_value=2, value=256, step=2)
        encoder_dims = st.text_input("Encoder Dimensions", "[2048,1024,512]")
        auto_decoder = st.checkbox("Auto-generate decoder", value=True)

        if auto_decoder:
            try:
                enc = parse_list(encoder_dims)
                if isinstance(enc, list):
                    decoder_dims = str(enc[::-1])
                else:
                    decoder_dims = "[512,1024,2048]"
            except Exception:
                decoder_dims = "[512,1024,2048]"
            st.text_input("Decoder Dimensions", value=decoder_dims, disabled=True)
        else:
            decoder_dims = st.text_input("Decoder Dimensions", "[512,1024,2048]")
            
    st.subheader("MMD Loss Strategy")
    mmd_mode = st.radio("MMD Strategy", ["Constant", "Linear", "Cosine", "Adaptive"], horizontal=True)
    mmd_config = {"mode": mmd_mode.lower()}
    
    if mmd_mode == "Constant":
        mmd_config["weight"] = st.number_input("MMD Weight", min_value=0.0, value=500.0, step=10.0)
    elif mmd_mode == "Linear":
        col_a, col_b = st.columns(2)
        with col_a:
            mmd_config["start_weight"] = st.number_input("Start Weight", value=0.0, step=10.0)
        with col_b:
            mmd_config["end_weight"] = st.number_input("End Weight", value=500.0, step=10.0)
        mmd_config["warmup_epochs"] = st.number_input("Warmup Epochs", min_value=1, value=250, step=50)
    elif mmd_mode == "Cosine":
        col_a, col_b = st.columns(2)
        with col_a:
            mmd_config["min_weight"] = st.number_input("Minimum Weight", value=0.0, step=5.0)
        with col_b:
            mmd_config["max_weight"] = st.number_input("Maximum Weight", value=500.0, step=10.0)
        mmd_config["period"] = st.number_input("Cosine Period (epochs)", min_value=1, value=1000, step=100)
    else:
        st.info("Automatically adjusts the MMD weight during training.")
        mmd_config["target_ratio"] = st.number_input("Target MMD/Reconstruction Ratio", min_value=0.01, value=0.20, step=0.01, format="%.2f")
        mmd_config["momentum"] = st.number_input("Momentum", min_value=0.0, max_value=0.999, value=0.95, step=0.01, format="%.3f")

    st.divider()
    dropout = st.number_input("Dropout", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    activation = st.selectbox("Activation", ["SiLU", "ReLU", "GELU", "LeakyReLU"])
    use_layernorm = st.checkbox("Use LayerNorm", value=True)
    st.divider()

with tab2:
    st.subheader("Diffusion Architecture")
    diffusion_architecture = st.selectbox("Architecture", ["Standard", "Residual", "TabDiff", "Custom"])
    num_diffusion_steps = st.number_input("Diffusion Timesteps", min_value=1, value=2000, step=100)

    if diffusion_architecture == "Standard":
        diffusion_hidden_dims = "[512,512,256]"
        st.text_input("Hidden Dimensions", value=diffusion_hidden_dims, disabled=True)
    elif diffusion_architecture == "Residual":
        diffusion_hidden_dims = "[512,512,512,512,256]"
        st.text_input("Hidden Dimensions", value=diffusion_hidden_dims, disabled=True)
    elif diffusion_architecture == "TabDiff":
        diffusion_hidden_dims = "[1024,1024,1024,512,512]"
        st.text_input("Hidden Dimensions", value=diffusion_hidden_dims, disabled=True)
    else:
        diffusion_hidden_dims = st.text_input("Hidden Dimensions", "[512,512,256]")

    diffusion_dropout = st.number_input(
        "Diffusion Dropout", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.05,
        help="Use 0.1 or 0.2 to prevent the MLP from memorizing large tabular datasets."
    )

    st.divider()
    st.subheader("Noise Schedule")
    noise_schedule = st.selectbox("Schedule", ["Linear", "Cosine", "Quadratic", "Sigmoid", "Learnable"])
    noise_config = {"type": noise_schedule.lower()}

    if noise_schedule != "Learnable":
        beta_start = st.number_input("Beta Start", min_value=0.0, value=0.0001, format="%.5f")
        beta_end = st.number_input("Beta End", min_value=0.0, value=0.0200, step=0.005, format="%.5f")
        noise_config["beta_start"] = beta_start
        noise_config["beta_end"] = beta_end
    else:
        st.info("Column-wise learnable schedule parameters are optimized jointly with the denoising network.")

    st.subheader("Time Conditioning")
    time_embedding_type = st.selectbox("Embedding Strategy", ["Sinusoidal", "Learned", "Fourier"])
    time_conditioning = st.selectbox("Conditioning Method", ["Addition", "FiLM", "Concatenation"])
    time_embedding_dim = st.number_input("Embedding Dimension", min_value=8, value=128, step=8)

with tab5:
    st.header("Model Performance Analytics")
    mlflow_uri = f"file://{os.path.abspath('../mlruns')}"
    
    @st.cache_resource
    def get_mlflow_client(uri):
        mlflow.set_tracking_uri(uri)
        return MlflowClient(tracking_uri=uri)
    client= get_mlflow_client(mlflow_uri)
    try:
        experiment = client.get_experiment_by_name("Mixed_Tabular_VAE_Diffusion")
    except Exception:
        experiment = None

    if experiment is None:
        st.warning("No MLflow experiment found. Run the pipeline at least once to generate the database!")
    else:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        if not runs:
            st.info("No runs logged yet.")
        else:
            run_options = {}
            for run in runs:
                start_time = run.info.start_time
                dt = pd.to_datetime(start_time, unit='ms').strftime('%Y-%m-%d %H:%M') if start_time else "Unknown Time"
                run_name = run.data.tags.get('mlflow.runName', 'Unnamed Run')
                short_id = run.info.run_id[:8]
                label = f"{dt} | {run_name} ({short_id})"
                run_options[label] = run

            selected_run_label = st.selectbox("Select an Experiment Run:", list(run_options.keys()))
            selected_run = run_options[selected_run_label]
            run_id = selected_run.info.run_id
            st.divider()
            
            col_params, col_metrics = st.columns(2)
            with col_params:
                st.subheader("Hyperparameters")
                params = selected_run.data.params
                if params:
                    param_df = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
                    st.dataframe(param_df, width="stretch", hide_index=True, height=250)
            
            with col_metrics:
                st.subheader("Final Metrics")
                metrics = selected_run.data.metrics
                if metrics:
                    metric_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
                    st.dataframe(metric_df, width="stretch", hide_index=True, height=250)

            st.divider()
            st.subheader("Experiment Visualizations")
            with st.spinner("Fetching plots from MLflow artifacts..."):
                all_images = fetch_mlflow_images(mlflow_uri, run_id)

            if all_images:
                loss_plots = [img for img in all_images if "loss" in img.lower()]
                heatmap_plots = [img for img in all_images if "heatmap" in img.lower()]
                num_dist_plots = [img for img in all_images if "dist_num" in img.lower()]
                cat_dist_plots = [img for img in all_images if "dist_cat" in img.lower()]
                
                if loss_plots:
                    with st.expander(" Training & Loss Curves", expanded=True):
                        render_image_grid(loss_plots, client, run_id, "Loss Curves")
                if heatmap_plots:
                    with st.expander(" Correlation Heatmaps", expanded=True):
                        render_image_grid(heatmap_plots, client, run_id, "Real vs Fake Correlations")
                if num_dist_plots:
                    with st.expander(" Numerical Distributions", expanded=False):
                        render_image_grid(num_dist_plots, client, run_id, "Numerical Density (KDE)")
                if cat_dist_plots:
                    with st.expander("Categorical Distributions", expanded=False):
                        render_image_grid(cat_dist_plots, client, run_id, "Categorical Bar Charts")

with tab4:
    st.header("Model Architecture")
    dataset_input_dims = {
        "adult": 104, "default": 30, "magic": 10, "shoppers": 18, 
        "beijing": 15, "news": 58, "diabetes": 45
    }
    current_input_dim = dataset_input_dims.get(selected_dataset, 128)
    
    architecture_config = {
        "dataset": selected_dataset,
        "input_dim": current_input_dim,
        "vae_params": {
            "activation": activation,
            "layernorm": use_layernorm,
            "dropout": dropout,
            "epochs": epochs,
            "learning_rate": vae_lr,
            "batch_size": batch_size
        },
        "latent": {
            "mode": latent_mode.lower(),
            "heuristic": latent_heuristic.lower() if latent_heuristic else None,
            "latent_dim": latent_dim,
            "encoder_dims": parse_list(encoder_dims) if encoder_dims else None,
            "decoder_dims": parse_list(decoder_dims) if decoder_dims else None
        },
        "mmd": {"strategy": mmd_mode.lower()},
        "diffusion": {
            "architecture": diffusion_architecture.lower(),
            "hidden_dims": parse_list(diffusion_hidden_dims),
            "num_diffusion_steps": num_diffusion_steps,
            "noise_schedule": noise_config,
            "time_conditioning": {
                "embedding": time_embedding_type.lower(),
                "method": time_conditioning.lower(),
                "embedding_dim": time_embedding_dim
            }
        }
    }

    dataset_name = selected_dataset
    dataset_url = "local" if selected_dataset == "custom" else DATASET_MAP[selected_dataset]

    # This config dictionary is what actually gets dumped to YAML
    config = {
        "dataset": {"name": selected_dataset, "url": dataset_url},
        "vae": {
            "epochs": epochs,
            "learning_rate": vae_lr,
            "batch_size": batch_size,
            "dropout": dropout,
            "activation": activation,
            "layernorm": use_layernorm,
            "latent": architecture_config["latent"],
            "mmd": mmd_config
        },
        "diffusion": {
            "epochs": diffusion_epochs,
            "learning_rate": diff_lr,
            "num_diffusion_steps": num_diffusion_steps,
            "context_injection": {"context_columns": context_columns_list},
            "architecture": {
                "type": diffusion_architecture.lower(),
                "hidden_dims": parse_list(diffusion_hidden_dims),
                "dropout": diffusion_dropout
            },
            "time_conditioning": {
                "embedding": time_embedding_type.lower(),
                "method": time_conditioning.lower(),
                "embedding_dim": time_embedding_dim
            },
            "noise_schedule": noise_config
        }
    }
    
    html = render_architecture(architecture_config)
    st.components.v1.html(html, height=700, scrolling=True)

# ----------------------------------------------------------------------
# NEW: UPDATED TAB 6 WITH MULTI-COLUMN DYNAMIC BLUEPRINT LOGIC
# ----------------------------------------------------------------------
with tab6: 
    st.header("Synthetic Data Generation")
    
    # 1. Pull the context columns dynamically from the active UI list
    context_cols = context_columns_list
    
    if not context_cols:
        st.info("Unconditional Generation: No context columns selected. The model will output random latent profiles.")
        total_rows = st.number_input("Total rows to generate:", min_value=100, value=1000, step=100)
        
        if st.button(" Generate Unconditional Data", type="primary"):
            st.success("Feature ready to be linked to unconditional inference backend!")
            
    else:
        st.success(f" Conditional Generation Active. Steering anchors: **{', '.join(context_cols)}**")
        st.markdown("### Define Your Specific Context Blueprint")
        st.markdown("Specify the exact values you want to lock in for this generation batch (e.g., Year: 2025, Month: 3)")
        
        blueprint_values = {}
        
        # Format the UI so inputs sit side-by-side cleanly
        input_columns = st.columns(min(len(context_cols), 4))
        
        # Dynamically create an input field for EVERY context column chosen
        for i, col in enumerate(context_cols):
            with input_columns[i % 4]:
                raw_val = st.text_input(f"Value for '{col}':", key=f"bp_{col}", placeholder=f"Enter {col}...")
                blueprint_values[col] = raw_val
                
        total_rows = st.number_input(
            "Total rows to generate:", 
            min_value=1, 
            max_value=100000, 
            value=10, 
            step=1, 
            key="total_rows"
        )
        
        st.divider()
        if st.button("🚀 Generate Steered Data", type="primary"):
            with st.spinner("Building Blueprint and Steering Diffusion Model..."):
                clean_dict = {}
                for col_name, user_input in blueprint_values.items():
                    try:
                        parsed_val = ast.literal_eval(user_input)
                        clean_dict[col_name] = [parsed_val] * total_rows
                    except (ValueError, SyntaxError):
                        clean_dict[col_name] = [user_input] * total_rows
                
                blueprint_df = pd.DataFrame(clean_dict)
                
                # Resolve absolute path
                ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                custom_dir = os.path.join(ROOT_DIR, "data", "custom")
                os.makedirs(custom_dir, exist_ok=True)
                blueprint_path = os.path.join(custom_dir, "ui_blueprint.csv")
                
                blueprint_df.to_csv(blueprint_path, index=False)
                
                # --- ADD THIS LINE TO DEBUG ---
                print(f"DEBUG STREAMLIT: Wrote blueprint file to -> {blueprint_path}")
                
                st.success(f"Blueprint successfully created with {total_rows} rows! Saved to {blueprint_path}.")
                st.info("The backend Inference script can now load this blueprint, pass it to the ContextEncoder, and generate the exact data requested.")
                
st.markdown("### Master Training Controls")
left, center, right = st.columns([3, 2, 3])

with center:
    save_config = st.button("💾 Save Configuration YAML", width="stretch")

if save_config:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=4)
    st.session_state.config_saved = True
    st.success(f"Configuration saved successfully to `{CONFIG_PATH}`.")

if st.session_state.config_saved:
    with center:
        run_model = st.button("🔥 Run Training Pipeline", width="stretch", type="primary")

    if run_model:
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        custom_dir = os.path.join(ROOT_DIR, "data", "custom")
        os.makedirs(custom_dir, exist_ok=True)
        blueprint_path = os.path.join(custom_dir, "ui_blueprint.csv")
        if context_columns_list:
            try:
                clean_dict = {}
                # Pulls the exact integer the user typed into the UI box
                target_rows = int(st.session_state.get("total_rows", 10))
                
                for col in context_columns_list:
                    user_input = st.session_state.get(f"bp_{col}", "")
                    try:
                        parsed_val = ast.literal_eval(user_input)
                        clean_dict[col] = [parsed_val] * target_rows
                    except (ValueError, SyntaxError):
                        clean_dict[col] = [user_input] * target_rows
                
                blueprint_df = pd.DataFrame(clean_dict)
                blueprint_df.to_csv(blueprint_path, index=False)
                print(f"SUCCESS: Blueprint created at {blueprint_path} with exactly {target_rows} rows.")
            except Exception as e:
                st.error(f"Failed to build blueprint from UI inputs: {e}")

        # --- RUN PIPELINE ---
        with st.spinner("Running full backend training process..."):
            try:
                subprocess.run([sys.executable, "alt_main.py"], cwd="..", check=True)
                st.success("Pipeline executed and trained successfully!")
            except subprocess.CalledProcessError as e:
                st.error(f"Script crashed during execution: {e}")
            except FileNotFoundError as e:
                st.error(f"Could not find the script to run: {e}")
