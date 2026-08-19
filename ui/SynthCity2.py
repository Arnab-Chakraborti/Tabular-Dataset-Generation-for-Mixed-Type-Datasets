import streamlit as st
import yaml
import ast
import os
import subprocess
import sys
import tempfile
import pandas as pd
import mlflow
import requests
import zipfile
import io
from mlflow.tracking import MlflowClient
from PIL import Image

try:
    from architecture import render_architecture
except Exception:
    def render_architecture(cfg):
        return "<div style='color:#8fa3c8;padding:24px;'>Architecture preview module not found.</div>"

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="SynthCity 2.0 | Synthetic Data Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG_PATH = "../configs/default_params.yaml"
# Ensure we resolve to the right absolute project directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR if os.path.exists(os.path.join(SCRIPT_DIR, "configs")) else os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# ============================================================================
# CONSTANTS & META
# ============================================================================
DATASET_MAP = {
    "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "default": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "magic": "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip",
    "shoppers": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
    "beijing": "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip",
    "news": "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip",
    "diabetes": "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
}

DATASET_META = {
    "adult": {"label": "Adult Census Income", "desc": "Demographic attributes and income bracket classification."},
    "default": {"label": "Credit Card Default", "desc": "Financial risk indicators for credit default prediction."},
    "magic": {"label": "MAGIC Gamma Telescope", "desc": "High-energy gamma particle detection simulation data."},
    "shoppers": {"label": "Online Shoppers Intent", "desc": "E-commerce browsing behavior and purchase-intent signals."},
    "beijing": {"label": "Beijing PM2.5", "desc": "Air quality and meteorological time-series measurements."},
    "news": {"label": "Online News Popularity", "desc": "Content and engagement metrics for online news articles."},
    "diabetes": {"label": "Diabetes 130-US Hospitals", "desc": "Clinical records from diabetic patient hospital encounters."},
}

AVAILABLE_DATASETS = list(DATASET_MAP.keys())

INPUT_DIM_MAP = {
    "adult": 104, "default": 30, "magic": 10, "shoppers": 18,
    "beijing": 15, "news": 58, "diabetes": 45,
}

EXPERIMENT_NAME = "Mixed_Tabular_VAE_Diffusion"

# ============================================================================
# SESSION STATE
# ============================================================================
_DEFAULTS = {
    "config_saved": False,
    "nav": "New Training Run",
    "selected_dataset": None,
    "custom_file_path": None,
    "custom_file_name": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ============================================================================
# STYLES (Professional & Modern Dark Theme)
# ============================================================================
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .stApp {
        background-color: #060913;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #0a0f1c;
        border-right: 1px solid #1e293b;
    }
    
    /* Modern Glassmorphism Banner */
    .sc-banner {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.8) 0%, rgba(30, 58, 138, 0.8) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    .sc-banner h1 {
        color: #ffffff;
        font-size: 28px;
        margin: 0 0 8px 0;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .sc-banner p {
        color: #bfdbfe;
        margin: 0;
        font-size: 15px;
    }
    
    /* Smooth Container Hover Effects */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #0f172a;
        border: 1px solid #1e293b !important;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #3b82f6 !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.12);
        transform: translateY(-2px);
    }
    
    /* Typography & Badges */
    .sc-section-title {
        color: #f8fafc;
        font-size: 18px;
        font-weight: 600;
        margin: 10px 0 16px 0;
    }
    .sc-card-title {
        color: #f1f5f9;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .sc-card-desc {
        color: #94a3b8;
        font-size: 13px;
        line-height: 1.45;
        min-height: 55px;
    }
    .sc-card-meta {
        color: #64748b;
        font-size: 11px;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }
    .sc-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .sc-badge-selected {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .sc-badge-complete {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .sc-badge-running {
        background-color: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Stats Box in Sidebar */
    .sc-stat-box {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 10px;
        text-align: center;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }
    .sc-stat-num {
        color: #f8fafc;
        font-size: 24px;
        font-weight: 800;
    }
    .sc-stat-label {
        color: #64748b;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-top: 4px;
    }
    
    /* Sleek Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #334155;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #3b82f6;
        color: #3b82f6;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        border: none;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3);
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 15px rgba(37, 99, 235, 0.4);
    }
    
    /* Divider */
    .sc-divider-label {
        display: flex;
        align-items: center;
        text-align: center;
        color: #475569;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        margin: 35px 0;
    }
    .sc-divider-label::before, .sc-divider-label::after {
        content: "";
        flex: 1;
        border-bottom: 1px solid #1e293b;
    }
    .sc-divider-label::before { margin-right: 20px; }
    .sc-divider-label::after { margin-left: 20px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# DYNAMIC DATASET INTROSPECTION SCRIPT
# ============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def get_dataset_schema(dataset_key):
    """
    Dynamically loads the number of columns and column names by reading just the 
    header of the dataset from the local cache or the UCI zip URL.
    """
    if dataset_key == "custom":
        return []
    url = DATASET_MAP.get(dataset_key)
    if not url:
        return []
        
    try:
        # 1. Prefer local cached data first
        local_dir = os.path.join(ROOT_DIR, "data", dataset_key)
        if os.path.exists(local_dir):
            valid_exts = ['.csv', '.data', '.txt']
            for f in os.listdir(local_dir):
                if any(f.lower().endswith(ext) for ext in valid_exts):
                    df = pd.read_csv(os.path.join(local_dir, f), nrows=0, skipinitialspace=True)
                    return df.columns.tolist()
                    
        # 2. Fetch directly from remote zip if not found locally
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                valid_files = [info for info in z.infolist() if any(info.filename.lower().endswith(ext) for ext in ['.csv', '.data', '.txt'])]
                if not valid_files: 
                    return []
                largest_file = max(valid_files, key=lambda i: i.file_size)
                with z.open(largest_file.filename) as f:
                    df = pd.read_csv(f, nrows=0, skipinitialspace=True)
                    return df.columns.tolist()
    except Exception:
        pass
    return []

# ============================================================================
# HELPERS
# ============================================================================
def parse_list(value):
    try:
        return ast.literal_eval(value)
    except Exception:
        return value

@st.cache_data(ttl=30)
def get_sidebar_run_count(tracking_uri, exp_name):
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        exp = client.get_experiment_by_name(exp_name)
        if exp:
            return len(client.search_runs(experiment_ids=[exp.experiment_id]))
    except Exception:
        pass
    return 0

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
        elif a.path.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(a.path)
    return image_paths

@st.cache_data(show_spinner=False, ttl=300)
def fetch_image_bytes(tracking_uri, run_id, artifact_path):
    client = MlflowClient(tracking_uri=tracking_uri)
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = client.download_artifacts(run_id, artifact_path, tmp_dir)
        with open(local_path, "rb") as f:
            return f.read()

@st.cache_data(ttl=10)
def fetch_mlflow_all_artifacts(tracking_uri, run_id, path=""):
    """Recursively list every artifact path logged for a run."""
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        artifacts = client.list_artifacts(run_id, path)
    except Exception:
        return []
    all_paths = []
    for a in artifacts:
        if a.is_dir:
            all_paths.extend(fetch_mlflow_all_artifacts(tracking_uri, run_id, a.path))
        else:
            all_paths.append(a.path)
    return all_paths


def render_image_grid(image_paths, tracking_uri, run_id, title=None):
    if not image_paths:
        return
    if title:
        st.markdown(f"**{title}**")
    cols = st.columns(2)
    for i, img_path in enumerate(image_paths):
        img_bytes = fetch_image_bytes(tracking_uri, run_id, img_path)
        raw_name = img_path.split("/")[-1]
        clean_caption = raw_name.replace(".png", "").replace(".jpg", "").replace("_", " ").title()
        with cols[i % 2]:
            st.image(img_bytes, caption=clean_caption, width="stretch")


def render_keyword_artifact_section(client, run_id, tracking_uri, keyword, empty_message):
    all_paths = fetch_mlflow_all_artifacts(tracking_uri, run_id)
    matches = [p for p in all_paths if keyword.lower() in p.lower()]

    if not matches:
        st.caption(empty_message)
        return

    image_matches = [p for p in matches if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    html_matches = [p for p in matches if p.lower().endswith((".html", ".htm"))]
    other_matches = [p for p in matches if p not in image_matches and p not in html_matches]

    if image_matches:
        render_image_grid(image_matches, tracking_uri, run_id)

    for h in html_matches:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = client.download_artifacts(run_id, h, tmp_dir)
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                html_body = f.read()
            st.components.v1.html(html_body, height=600, scrolling=True)

    for o in other_matches:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = client.download_artifacts(run_id, o, tmp_dir)
            with open(local_path, "rb") as f:
                st.download_button(
                    label=f"📥 Download {os.path.basename(o)}",
                    data=f.read(),
                    file_name=os.path.basename(o),
                    key=f"dl_{keyword}_{run_id}_{o}",
                )

@st.cache_resource
def get_mlflow_client(uri):
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)

def save_custom_dataset(uploaded_file):
    custom_dir = os.path.join(ROOT_DIR, "data", "custom")
    os.makedirs(custom_dir, exist_ok=True)
    for f in os.listdir(custom_dir):
        os.remove(os.path.join(custom_dir, f))
    custom_file_path = os.path.join(custom_dir, uploaded_file.name)
    with open(custom_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return custom_file_path

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:6px 0 24px 0;">
            <div style="color:#ffffff;font-size:22px;font-weight:800;letter-spacing:-0.5px;">SynthCity <span style='color:#3b82f6;'>2.0</span></div>
            <div style="color:#94a3b8;font-size:12px;line-height:1.5;margin-top:6px;">
                A unified platform for tabular generative models,
                synthetic data quality evaluation, and comprehensive latent diagnostics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav = st.radio(
        "Navigation",
        ["New Training Run", "Previous Runs"],
        index=0 if st.session_state.nav == "New Training Run" else 1,
        label_visibility="collapsed",
    )
    st.session_state.nav = nav

    st.divider()

    mlflow_uri = f"file://{os.path.join(ROOT_DIR, 'mlruns')}"
    _run_count = get_sidebar_run_count(mlflow_uri, EXPERIMENT_NAME)

    stat_a, stat_b = st.columns(2)
    with stat_a:
        st.markdown(
            f"""<div class="sc-stat-box"><div class="sc-stat-num">{len(AVAILABLE_DATASETS)}</div>
            <div class="sc-stat-label">Presets</div></div>""",
            unsafe_allow_html=True,
        )
    with stat_b:
        st.markdown(
            f"""<div class="sc-stat-box"><div class="sc-stat-num">{_run_count}</div>
            <div class="sc-stat-label">Logged Runs</div></div>""",
            unsafe_allow_html=True,
        )

# ============================================================================
# PAGE: NEW TRAINING RUN
# ============================================================================
def render_new_run_page():
    st.markdown(
        """
        <div class="sc-banner">
            <h1>Initialize Training Pipeline</h1>
            <p>Select an architecture and dataset to configure your VAE + Latent Diffusion engine.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sc-section-title">Model Architecture</div>', unsafe_allow_html=True)
    col_model, _ = st.columns([1, 3])
    with col_model:
        st.selectbox("Select Generative Framework", ["Latent-Diffusion 2.2", "GAN 2.1", "Vanilla VAE 1.3", "InfoVAE-MMD 1.5"])

    st.markdown('<div class="sc-section-title" style="margin-top:20px;">Dataset Selection</div>', unsafe_allow_html=True)

    grid_cols = st.columns(4)
    for i, key in enumerate(AVAILABLE_DATASETS):
        meta = DATASET_META[key]
        # Dynamically load columns to show accurate schema counts on cards
        cols_schema = get_dataset_schema(key)
        col_count = len(cols_schema) if cols_schema else "Unknown"
        
        with grid_cols[i % 4]:
            with st.container(border=True):
                st.markdown(f'<div class="sc-card-title">{meta["label"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="sc-card-desc">{meta["desc"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="sc-card-meta">{col_count} attributes</div>', unsafe_allow_html=True)
                is_selected = st.session_state.selected_dataset == key
                if is_selected:
                    st.markdown('<div style="text-align:center;margin-top:12px;"><span class="sc-badge sc-badge-selected">Selected</span></div>', unsafe_allow_html=True)
                else:
                    if st.button("Select", key=f"select_{key}", width="stretch"):
                        st.session_state.selected_dataset = key
                        st.session_state.custom_file_path = None
                        st.session_state.custom_file_name = None
                        st.rerun()

    st.markdown('<div class="sc-divider-label">OR UPLOAD CUSTOM</div>', unsafe_allow_html=True)

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Ingest local CSV dataset",
            type=["csv", "txt", "data"],
            label_visibility="visible",
        )
        if uploaded_file is not None:
            custom_file_path = save_custom_dataset(uploaded_file)
            st.session_state.selected_dataset = "custom"
            st.session_state.custom_file_path = custom_file_path
            st.session_state.custom_file_name = uploaded_file.name
            st.success(f"'{uploaded_file.name}' staged successfully.")
        elif st.session_state.selected_dataset == "custom" and st.session_state.custom_file_name:
            st.info(f"Using staged file: {st.session_state.custom_file_name}")

    if not st.session_state.selected_dataset:
        return

    st.divider()
    render_configuration_section()


def render_configuration_section():
    selected_dataset = st.session_state.selected_dataset
    custom_file_path = st.session_state.custom_file_path

    label = DATASET_META.get(selected_dataset, {}).get("label", "Custom Dataset")
    st.markdown(f'<div class="sc-section-title">Context Injection &middot; {label}</div>', unsafe_allow_html=True)
    st.caption("Select columns to be used as strict conditional anchors to steer the diffusion generative process.")

    context_columns_list = []
    if selected_dataset == "custom" and custom_file_path:
        try:
            df_preview = pd.read_csv(custom_file_path, nrows=0)
            context_columns_list = st.multiselect(
                "Select Conditional Context Columns:",
                options=df_preview.columns.tolist(),
                help="Target classification labels or key demographic constraints.",
            )
        except Exception as e:
            st.warning(f"Could not parse uploaded file: {e}")
    else:
        with st.spinner("Analyzing dataset schema..."):
            schema_cols = get_dataset_schema(selected_dataset)
        
        if schema_cols:
            context_columns_list = st.multiselect(
                f"Select Conditional Context Columns (Available: {len(schema_cols)}):",
                options=schema_cols,
                help="These columns will be routed through the ContextEncoder.",
            )
        else:
            ctx_cols_input = st.text_input("Context Columns (Comma-separated)", placeholder="e.g., job, marital")
            context_columns_list = [c.strip() for c in ctx_cols_input.split(",") if c.strip()]

    st.divider()

    st.markdown('<div class="sc-section-title">Hyperparameter Workbench</div>', unsafe_allow_html=True)
    tab_training, tab_vae, tab_diffusion = st.tabs(["Training Options", "InfoVAE", "Latent Diffusion"])

    # ---------------- TRAINING ----------------
    with tab_training:
        st.subheader("Optimizer Settings")
        epochs = st.number_input("VAE Epochs", min_value=1, value=1000, step=100)
        diffusion_epochs = st.number_input("Diffusion Epochs", min_value=1, value=2000, step=100)
        vae_lr = st.number_input("VAE Learning Rate", min_value=0.00001, value=0.001, step=0.0001, format="%.5f")
        diff_lr = st.number_input("Diffusion Learning Rate", min_value=0.00001, value=0.001, step=0.0001, format="%.5f")
        batch_size = st.number_input("Batch Size", min_value=1, value=256, step=32)

    # ---------------- VAE ----------------
    with tab_vae:
        st.subheader("Manifold Architecture")
        latent_mode = st.radio("Latent Dimension Strategy", ["Adaptive", "Custom"], horizontal=True)

        if latent_mode == "Adaptive":
            latent_heuristic = st.selectbox("Heuristic Model", ["Google", "FastAI", "FH"])
            latent_dim = None
            encoder_dims = None
            decoder_dims = None
            st.info("Latent geometry will be inferred dynamically from the input tabular space.")
        else:
            latent_heuristic = None
            latent_dim = st.number_input("Latent Dimension", min_value=2, value=256, step=2)
            encoder_dims = st.text_input("Encoder Dimensions", "[2048, 1024, 512]")
            if st.checkbox("Auto-reflect decoder", value=True):
                try:
                    enc = parse_list(encoder_dims)
                    decoder_dims = str(enc[::-1]) if isinstance(enc, list) else "[512, 1024, 2048]"
                except Exception:
                    decoder_dims = "[512, 1024, 2048]"
                st.text_input("Decoder Dimensions", value=decoder_dims, disabled=True)
            else:
                decoder_dims = st.text_input("Decoder Dimensions", "[512, 1024, 2048]")

        st.markdown("**MMD Kernel Distance**")
        mmd_mode = st.radio("MMD Schedule", ["Constant", "Linear", "Cosine", "Adaptive"], horizontal=True)
        mmd_config = {"mode": mmd_mode.lower()}

        if mmd_mode == "Constant":
            mmd_config["weight"] = st.number_input("MMD Base Weight", min_value=0.0, value=500.0, step=10.0)
        elif mmd_mode == "Linear":
            c1, c2 = st.columns(2)
            mmd_config["start_weight"] = c1.number_input("Start Weight", value=0.0, step=10.0)
            mmd_config["end_weight"] = c2.number_input("End Weight", value=500.0, step=10.0)
            mmd_config["warmup_epochs"] = st.number_input("Warmup Epochs", min_value=1, value=250, step=50)
        elif mmd_mode == "Cosine":
            c1, c2 = st.columns(2)
            mmd_config["min_weight"] = c1.number_input("Min Weight", value=0.0, step=5.0)
            mmd_config["max_weight"] = c2.number_input("Max Weight", value=500.0, step=10.0)
            mmd_config["period"] = st.number_input("Cosine Period (epochs)", min_value=1, value=1000, step=100)
        else:
            mmd_config["target_ratio"] = st.number_input("Target Reconstruct Ratio", min_value=0.01, value=0.20, step=0.01, format="%.2f")
            mmd_config["momentum"] = st.number_input("Momentum Term", min_value=0.0, max_value=0.999, value=0.95, step=0.01, format="%.3f")

        st.divider()
        c3, c4 = st.columns(2)
        activation = c3.selectbox("Activation Function", ["SiLU", "ReLU", "GELU", "LeakyReLU"])
        dropout = c4.number_input("Dropout Rate", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        use_layernorm = st.checkbox("Enable LayerNorm", value=True)

    # ---------------- DIFFUSION ----------------
    with tab_diffusion:
        st.subheader("Denoising MLP")
        diffusion_architecture = st.selectbox("Network Topology", ["Standard", "Residual", "TabDiff", "Custom"])
        num_diffusion_steps = st.number_input("Markov Timesteps ($T$)", min_value=1, value=2000, step=100)

        if diffusion_architecture == "Standard": diffusion_hidden_dims = "[512, 512, 256]"
        elif diffusion_architecture == "Residual": diffusion_hidden_dims = "[512, 512, 512, 512, 256]"
        elif diffusion_architecture == "TabDiff": diffusion_hidden_dims = "[1024, 1024, 1024, 512, 512]"
        else: diffusion_hidden_dims = st.text_input("Hidden Layers", "[512, 512, 256]")

        if diffusion_architecture != "Custom":
            st.text_input("Hidden Layers (Locked)", value=diffusion_hidden_dims, disabled=True)
            
        diffusion_dropout = st.number_input("Denoising Dropout", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

        st.markdown("**Noise Beta Schedule**")
        noise_schedule = st.selectbox("Variance Scheduler", ["Linear", "Cosine", "Quadratic", "Sigmoid", "Learnable"])
        noise_config = {"type": noise_schedule.lower()}

        if noise_schedule != "Learnable":
            c5, c6 = st.columns(2)
            noise_config["beta_start"] = c5.number_input("$\\beta_{start}$", min_value=0.0, value=0.0001, format="%.5f")
            noise_config["beta_end"] = c6.number_input("$\\beta_{end}$", min_value=0.0, value=0.0200, step=0.005, format="%.5f")
        else:
            st.info("Schedule parameters will be optimized directly via gradient descent.")

        st.markdown("**Conditional Injection**")
        c7, c8 = st.columns(2)
        time_embedding_type = c7.selectbox("Temporal Embedding", ["Sinusoidal", "Learned", "Fourier"])
        time_conditioning = c8.selectbox("Modulation Approach", ["Addition", "FiLM", "Concatenation"])
        time_embedding_dim = st.number_input("Temporal Projection Dim", min_value=8, value=128, step=8)

    st.divider()
    
    # ---------------- GENERATION BLUEPRINT ----------------
    st.markdown('<div class="sc-section-title">Data Generation Blueprint</div>', unsafe_allow_html=True)
    if not context_columns_list:
        st.info("Unconditional regime: The diffusion model will sample randomly from the learned prior.")
        total_rows = st.number_input("Number of samples to synthesize:", min_value=100, value=1000, step=100)
    else:
        st.markdown("Set specific constraint values to steer generation across the synthesized batch.")
        blueprint_values = {}
        input_columns = st.columns(min(len(context_columns_list), 4))
        for i, col in enumerate(context_columns_list):
            with input_columns[i % 4]:
                blueprint_values[col] = st.text_input(f"Target '{col}':", key=f"bp_{col}")

        total_rows = st.number_input("Samples to synthesize:", min_value=1, max_value=100000, value=10, step=1)

    # ---------------- SYSTEM EXECUTION ----------------
    st.markdown('<div class="sc-section-title" style="margin-top:40px;">Execution Engine</div>', unsafe_allow_html=True)
    left, center, right = st.columns([1, 2, 1])

    with center:
        run_model = st.button("Initialize Generative Pipeline", width="stretch", type="primary")

    if run_model:
        dataset_url = "local" if selected_dataset == "custom" else DATASET_MAP[selected_dataset]
        
        # Build Config Dict
        config = {
            "dataset": {"name": selected_dataset, "url": dataset_url},
            "vae": {
                "epochs": epochs,
                "learning_rate": vae_lr,
                "batch_size": batch_size,
                "dropout": dropout,
                "activation": activation,
                "layernorm": use_layernorm,
                "latent": {
                    "mode": latent_mode.lower(),
                    "heuristic": latent_heuristic.lower() if latent_heuristic else None,
                    "latent_dim": latent_dim,
                    "encoder_dims": parse_list(encoder_dims) if encoder_dims else None,
                    "decoder_dims": parse_list(decoder_dims) if decoder_dims else None,
                },
                "mmd": mmd_config,
            },
            "diffusion": {
                "epochs": diffusion_epochs,
                "learning_rate": diff_lr,
                "num_diffusion_steps": num_diffusion_steps,
                "context_injection": {"context_columns": context_columns_list},
                "architecture": {
                    "type": diffusion_architecture.lower(),
                    "hidden_dims": parse_list(diffusion_hidden_dims),
                    "dropout": diffusion_dropout,
                },
                "time_conditioning": {
                    "embedding": time_embedding_type.lower(),
                    "method": time_conditioning.lower(),
                    "embedding_dim": time_embedding_dim,
                },
                "noise_schedule": noise_config,
            },
        }

        # 1. Save Config
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=4)
        
        # 2. Build UI Blueprint CSV
        custom_dir = os.path.join(ROOT_DIR, "data", "custom")
        os.makedirs(custom_dir, exist_ok=True)
        blueprint_path = os.path.join(custom_dir, "ui_blueprint.csv")

        if context_columns_list:
            try:
                clean_dict = {}
                for col in context_columns_list:
                    user_input = st.session_state.get(f"bp_{col}", "")
                    try:
                        parsed_val = ast.literal_eval(user_input)
                        clean_dict[col] = [parsed_val] * int(total_rows)
                    except (ValueError, SyntaxError):
                        clean_dict[col] = [user_input] * int(total_rows)
                pd.DataFrame(clean_dict).to_csv(blueprint_path, index=False)
            except Exception as e:
                st.error(f"Failed to build generation blueprint: {e}")
        else:
            # Clear it if unconditional
            if os.path.exists(blueprint_path):
                os.remove(blueprint_path)

        # 3. Fire Subprocess
        with st.spinner("Compiling graphs and tracking to MLflow..."):
            try:
                subprocess.run([sys.executable, "alt_main.py"], cwd=ROOT_DIR, check=True)
                st.success("Training executed successfully! Results available in Previous Runs.")
            except subprocess.CalledProcessError as e:
                st.error(f"Execution Engine Crashed: {e}")
            except FileNotFoundError as e:
                st.error(f"Backend script missing: {e}")


# ============================================================================
# PAGE: PREVIOUS RUNS (MLFLOW HISTORICAL LEDGER)
# ============================================================================
def render_previous_runs_page():
    st.markdown(
        """
        <div class="sc-banner">
            <h1>Experiment Ledger</h1>
            <p>Access historical synthetic models, download generated datasets, and review latent diagnostics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mlflow_uri = f"file://{os.path.join(ROOT_DIR, 'mlruns')}"
    client = get_mlflow_client(mlflow_uri)

    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    except Exception:
        experiment = None

    if experiment is None:
        st.warning("No tracking database found. Execute a training run to initialize MLFlow.")
        return

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    if not runs:
        st.info("No experiment logs found.")
        return

    runs = sorted(runs, key=lambda r: r.info.start_time or 0, reverse=True)

    for run in runs:
        run_id = run.info.run_id
        start_time = run.info.start_time
        dt = pd.to_datetime(start_time, unit="ms").strftime("%d %b %Y, %H:%M") if start_time else "Unknown"
        run_name = run.data.tags.get("mlflow.runName", "Pipeline Run")
        dataset_name = run.data.params.get("dataset.name", run.data.params.get("dataset", "Unknown"))
        status = run.info.status
        badge_class = "sc-badge-complete" if status == "FINISHED" else "sc-badge-running"

        header_cols = st.columns([3, 2, 2, 2, 1.4])
        with header_cols[0]:
            st.markdown(f'<div class="sc-run-title">{run_name}</div><div class="sc-run-sub">{run_id[:8]}</div>', unsafe_allow_html=True)
        with header_cols[1]:
            st.markdown(f'<div class="sc-run-sub">Target Distribution</div><div class="sc-run-title">{dataset_name}</div>', unsafe_allow_html=True)
        with header_cols[2]:
            st.markdown(f'<div class="sc-run-sub">Executed On</div><div class="sc-run-title">{dt}</div>', unsafe_allow_html=True)
        with header_cols[3]:
            st.markdown(f'<span class="sc-badge {badge_class}">{status.title()}</span>', unsafe_allow_html=True)
        with header_cols[4]:
            expand = st.toggle("View Artifacts", key=f"toggle_{run_id}")

        if expand:
            with st.container(border=True):
                detail_view = st.radio(
                    "Inspect Analytics:",
                    ["Generated Data", "Model Parameters", "Generative Accuracy", "Latent Dynamics", "Visualizations", "Terminal Report"],
                    horizontal=True,
                    key=f"view_{run_id}"
                )

                # -------- 1. Generated Data Download --------
                if detail_view == "Generated Data":
                    st.markdown("**Synthetic Data Export**")
                    all_paths = fetch_mlflow_all_artifacts(mlflow_uri, run_id)
                    csv_paths = [p for p in all_paths if p.endswith(".csv")]
                    
                    if csv_paths:
                        for cp in csv_paths:
                            with tempfile.TemporaryDirectory() as tmp_dir:
                                local_path = client.download_artifacts(run_id, cp, tmp_dir)
                                with open(local_path, "rb") as f:
                                    st.download_button(
                                        label=f"📥 Download {os.path.basename(cp)}",
                                        data=f.read(),
                                        file_name=os.path.basename(cp),
                                        mime="text/csv",
                                        key=f"dl_csv_{run_id}_{cp}"
                                    )
                                try:
                                    df_prev = pd.read_csv(local_path, nrows=100)
                                    st.dataframe(df_prev, use_container_width=True, height=300)
                                    st.caption(f"Previewing first 100 rows of synthetic output.")
                                except Exception:
                                    pass
                    else:
                        st.info("No synthetic output files (`.csv`) found in the artifact store for this run.")

                # -------- 2. Hyperparameters --------
                elif detail_view == "Model Parameters":
                    st.markdown("**Hyperparameter Configurations**")
                    params = run.data.params
                    if params:
                        param_df = pd.DataFrame(list(params.items()), columns=["Parameter Key", "Configured Value"])
                        st.dataframe(param_df, width="stretch", hide_index=True, height=350)
                    else:
                        st.caption("No parameters logged.")

                # -------- 3. Generative Accuracy (Final Test Metrics) --------
                elif detail_view == "Generative Accuracy":
                    st.markdown("**Holdout Set Evaluation Metrics**")
                    metrics = run.data.metrics
                    if metrics:
                        final_metrics = {k: v for k, v in metrics.items() if k.startswith("test_")}
                        if final_metrics:
                            metric_df = pd.DataFrame(list(final_metrics.items()), columns=["Evaluation Metric", "Score"])
                            st.dataframe(metric_df, width="stretch", hide_index=True, height=280)
                        else:
                            st.caption("No final test accuracy metrics found in logs.")
                    else:
                        st.caption("No metrics logged.")

                # -------- 4. Latent Space (Internal VAE metrics) --------
                elif detail_view == "Latent Dynamics":
                    st.markdown("**Latent Manifold & Representation Metrics**")
                    metrics = run.data.metrics
                    if metrics:
                        latent_metrics = {k: v for k, v in metrics.items() if not k.startswith("test_")}
                        if latent_metrics:
                            latent_df = pd.DataFrame(list(latent_metrics.items()), columns=["Diagnostic Metric", "Computed Value"])
                            st.dataframe(latent_df, use_container_width=True, hide_index=True, height=350)
                        else:
                            st.caption("No internal latent space diagnostics found.")
                    else:
                        st.caption("No metrics logged.")

                # -------- 5. Visualizations --------
                elif detail_view == "Visualizations":
                    with st.spinner("Fetching plots from artifact store..."):
                        all_images = fetch_mlflow_images(mlflow_uri, run_id)

                    if not all_images:
                        st.info("No visualization plots generated for this run.")
                    else:
                        loss_plots = [img for img in all_images if "loss" in img.lower()]
                        heatmap_plots = [img for img in all_images if "heatmap" in img.lower()]
                        num_dist_plots = [img for img in all_images if "dist_num" in img.lower()]
                        cat_dist_plots = [img for img in all_images if "dist_cat" in img.lower()]

                        if loss_plots:
                            with st.expander("Convergence & Loss Trajectories", expanded=True):
                                render_image_grid(loss_plots, mlflow_uri, run_id)
                        if heatmap_plots:
                            with st.expander("Feature Correlation Alignment", expanded=True):
                                render_image_grid(heatmap_plots, mlflow_uri, run_id)
                        if num_dist_plots:
                            with st.expander("Marginal Distributions (Numerical)", expanded=False):
                                render_image_grid(num_dist_plots, mlflow_uri, run_id)
                        if cat_dist_plots:
                            with st.expander("Categorical Distributions", expanded=False):
                                render_image_grid(cat_dist_plots, mlflow_uri, run_id)

                # -------- 6. Report (report.py) --------
                elif detail_view == "Terminal Report":
                    st.caption("High-level system diagnostics generated by `report.py`.")
                    render_keyword_artifact_section(
                        client, run_id, mlflow_uri,
                        keyword="report",
                        empty_message="No text reports found.",
                    )
        st.divider()


# ============================================================================
# ROUTING
# ============================================================================
if st.session_state.nav == "New Training Run":
    render_new_run_page()
else:
    render_previous_runs_page()
