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
import psutil
import shutil
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
    page_title="SynthLab 2.0 | Synthetic Data Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG_PATH = "../configs/default_params.yaml"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR if os.path.exists(os.path.join(SCRIPT_DIR, "configs")) else os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# ============================================================================
# YAML BLANK TEMPLATE (For Help Section)
# ============================================================================
YAML_TEMPLATE = """dataset:
  name: default
  url: local
vae:
  epochs: 1000
  learning_rate: 0.001
  batch_size: 256
  dropout: 0.0
  activation: LeakyReLU
  layernorm: true
  latent:
    mode: adaptive
    heuristic: null
    latent_dim: 256
    encoder_dims: [2048, 1024, 512]
    decoder_dims: [512, 1024, 2048]
  mmd:
    mode: constant
    weight: 500.0
diffusion:
  epochs: 2000
  learning_rate: 0.001
  num_diffusion_steps: 2000
  context_injection:
    context_columns: []
  architecture:
    type: standard
    hidden_dims: [512, 512, 256]
    dropout: 0.1
  time_conditioning:
    embedding: sinusoidal
    method: film
    embedding_dim: 128
  noise_schedule:
    type: linear
    beta_start: 0.0001
    beta_end: 0.02
"""

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
    "loaded_config": {},
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
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 32px;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .sc-banner h1 {
        color: #f8fafc;
        font-size: 26px;
        margin: 0 0 6px 0;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .sc-banner p {
        color: #94a3b8;
        margin: 0;
        font-size: 14px;
    }
    
    /* Smooth Container Hover Effects */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #0f172a;
        border: 1px solid #1e293b !important;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #3b82f6 !important;
    }
    
    /* Typography & Badges */
    .sc-section-title {
        color: #f8fafc;
        font-size: 18px;
        font-weight: 600;
        margin: 20px 0 12px 0;
        letter-spacing: 0.5px;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 8px;
    }
    .sc-card-title {
        color: #f1f5f9;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .sc-card-desc {
        color: #94a3b8;
        font-size: 12.5px;
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
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .sc-badge-selected {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .sc-badge-complete {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .sc-badge-running {
        background-color: rgba(59, 130, 246, 0.1);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Stats Box in Sidebar */
    .sc-stat-box {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px 10px;
        text-align: center;
    }
    .sc-stat-num {
        color: #f8fafc;
        font-size: 22px;
        font-weight: 700;
    }
    .sc-stat-label {
        color: #64748b;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-top: 4px;
    }

    /* ==================================================
       SIDEBAR NAVIGATION (Pill/Tab replacement)
       ================================================== */
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background-color: transparent;
        border: none;
        color: #94a3b8;
        justify-content: flex-start;
        padding: 10px 16px;
        font-weight: 500;
        font-size: 14.5px;
        transition: all 0.2s ease;
        box-shadow: none;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background-color: rgba(30, 41, 59, 0.8);
        color: #f8fafc;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: #1e293b;
        border: 1px solid #1e293b;
        border-left: 4px solid #3b82f6;
        color: #ffffff;
        justify-content: flex-start;
        padding: 10px 12px; /* adjust for left border */
        font-weight: 600;
        font-size: 14.5px;
        box-shadow: none;
        border-radius: 4px;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background-color: #1e293b;
        border-color: #1e293b;
        border-left: 4px solid #3b82f6;
    }

    /* Main Area Matte Buttons */
    div.stMain .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        border: 1px solid #334155;
        background-color: #0f172a;
        color: #e2e8f0;
        transition: all 0.2s ease;
        height: 42px;
    }
    div.stMain .stButton > button:hover {
        border-color: #475569;
        background-color: #1e293b;
        color: #f8fafc;
    }
    
    div.stMain .stButton > button[kind="primary"] {
        background-color: #1e40af; 
        border: 1px solid #1e3a8a;
        color: #ffffff;
        letter-spacing: 0.5px;
    }
    div.stMain .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8;
        border-color: #1e40af;
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        border-radius: 6px;
        font-weight: 500;
        border: 1px solid #334155;
        background-color: #0f172a;
        color: #e2e8f0;
        width: 100%;
        height: 42px;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        border-color: #475569;
        background-color: #1e293b;
        color: #f8fafc;
    }
    
    .sc-divider-label {
        display: flex;
        align-items: center;
        text-align: center;
        color: #475569;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        margin: 30px 0;
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
# DYNAMIC DATASET INTROSPECTION
# ============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_dataset_preview(dataset_key):
    if dataset_key == "custom":
        custom_file = st.session_state.custom_file_path
        if custom_file and os.path.exists(custom_file):
            return pd.read_csv(custom_file, nrows=500, skipinitialspace=True)
        return None

    url = DATASET_MAP.get(dataset_key)
    if not url: return None
    
    try:
        local_dir = os.path.join(ROOT_DIR, "data", dataset_key)
        if os.path.exists(local_dir):
            valid_exts = ['.csv', '.data', '.txt']
            for f in os.listdir(local_dir):
                if any(f.lower().endswith(ext) for ext in valid_exts):
                    return pd.read_csv(os.path.join(local_dir, f), nrows=500, skipinitialspace=True)
                    
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                valid_files = [info for info in z.infolist() if any(info.filename.lower().endswith(ext) for ext in ['.csv', '.data', '.txt'])]
                if valid_files:
                    largest_file = max(valid_files, key=lambda i: i.file_size)
                    with z.open(largest_file.filename) as f:
                        return pd.read_csv(f, nrows=500, skipinitialspace=True)
    except Exception:
        pass
    return None

def get_dataset_schema(dataset_key):
    df = fetch_dataset_preview(dataset_key)
    return df.columns.tolist() if df is not None else []

def get_cfg(keys, default):
    d = st.session_state.loaded_config
    try:
        for k in keys: d = d[k]
        return d if d is not None else default
    except (KeyError, TypeError): return default

def parse_list(value):
    try: return ast.literal_eval(value)
    except Exception: return value

# ============================================================================
# MLFLOW HELPERS
# ============================================================================
@st.cache_data(ttl=30)
def get_sidebar_run_count(tracking_uri, exp_name):
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        exp = client.get_experiment_by_name(exp_name)
        if exp: return len(client.search_runs(experiment_ids=[exp.experiment_id]))
    except Exception: pass
    return 0

@st.cache_data(ttl=10)
def fetch_mlflow_images(tracking_uri, run_id, path=""):
    client = MlflowClient(tracking_uri=tracking_uri)
    try: artifacts = client.list_artifacts(run_id, path)
    except Exception: return []
    image_paths = []
    for a in artifacts:
        if a.is_dir: image_paths.extend(fetch_mlflow_images(tracking_uri, run_id, a.path))
        elif a.path.lower().endswith((".png", ".jpg", ".jpeg")): image_paths.append(a.path)
    return image_paths

@st.cache_data(show_spinner=False, ttl=300)
def fetch_image_bytes(tracking_uri, run_id, artifact_path):
    client = MlflowClient(tracking_uri=tracking_uri)
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = client.download_artifacts(run_id, artifact_path, tmp_dir)
        with open(local_path, "rb") as f: return f.read()

@st.cache_data(ttl=10)
def fetch_mlflow_all_artifacts(tracking_uri, run_id, path=""):
    client = MlflowClient(tracking_uri=tracking_uri)
    try: artifacts = client.list_artifacts(run_id, path)
    except Exception: return []
    all_paths = []
    for a in artifacts:
        if a.is_dir: all_paths.extend(fetch_mlflow_all_artifacts(tracking_uri, run_id, a.path))
        else: all_paths.append(a.path)
    return all_paths

def render_image_grid(image_paths, tracking_uri, run_id, title=None):
    if not image_paths: return
    if title: st.markdown(f"**{title}**")
    cols = st.columns(2)
    for i, img_path in enumerate(image_paths):
        img_bytes = fetch_image_bytes(tracking_uri, run_id, img_path)
        raw_name = img_path.split("/")[-1]
        clean_caption = raw_name.replace(".png", "").replace(".jpg", "").replace("_", " ").title()
        with cols[i % 2]: st.image(img_bytes, caption=clean_caption, width="stretch")

def render_keyword_artifact_section(client, run_id, tracking_uri, keyword, empty_message):
    all_paths = fetch_mlflow_all_artifacts(tracking_uri, run_id)
    matches = [p for p in all_paths if keyword.lower() in p.lower()]
    if not matches:
        st.caption(empty_message)
        return

    image_matches = [p for p in matches if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    html_matches = [p for p in matches if p.lower().endswith((".html", ".htm"))]
    other_matches = [p for p in matches if p not in image_matches and p not in html_matches]

    if image_matches: render_image_grid(image_matches, tracking_uri, run_id)

    for h in html_matches:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = client.download_artifacts(run_id, h, tmp_dir)
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)

    for o in other_matches:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = client.download_artifacts(run_id, o, tmp_dir)
            with open(local_path, "rb") as f:
                st.download_button(label=f"📥 Download {os.path.basename(o)}", data=f.read(), file_name=os.path.basename(o), key=f"dl_{keyword}_{run_id}_{o}")

@st.cache_resource
def get_mlflow_client(uri):
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)

def save_custom_dataset(uploaded_file):
    custom_dir = os.path.join(ROOT_DIR, "data", "custom")
    os.makedirs(custom_dir, exist_ok=True)
    for f in os.listdir(custom_dir): os.remove(os.path.join(custom_dir, f))
    custom_file_path = os.path.join(custom_dir, uploaded_file.name)
    with open(custom_file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    return custom_file_path

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:0 0 20px 0;">
            <div style="color:#ffffff;font-size:22px;font-weight:800;letter-spacing:-0.5px;">SynthLab <span style='color:#3b82f6;'>2.0</span></div>
            <div style="color:#94a3b8;font-size:12px;line-height:1.4;margin-top:4px;">
                Enterprise platform for tabular generative modeling, utility evaluation, and auditing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<span style='font-size:11px;font-weight:700;color:#64748b;letter-spacing:1px;'>NAVIGATION</span>", unsafe_allow_html=True)
    st.write("")
    
    # Sleek, custom-styled navigation buttons
    pages = ["New Training Run", "Previous Runs", "Auditing", "Documentation"]
    for page in pages:
        if st.button(
            label=page, 
            key=f"nav_{page}", 
            use_container_width=True, 
            type="primary" if st.session_state.nav == page else "secondary"
        ):
            st.session_state.nav = page
            st.rerun()

    st.divider()

    mlflow_uri = f"file://{os.path.join(ROOT_DIR, 'mlruns')}"
    _run_count = get_sidebar_run_count(mlflow_uri, EXPERIMENT_NAME)

    stat_a, stat_b = st.columns(2)
    with stat_a:
        st.markdown(f'<div class="sc-stat-box"><div class="sc-stat-num">{len(AVAILABLE_DATASETS)}</div><div class="sc-stat-label">Presets</div></div>', unsafe_allow_html=True)
    with stat_b:
        st.markdown(f'<div class="sc-stat-box"><div class="sc-stat-num">{_run_count}</div><div class="sc-stat-label">Logged Runs</div></div>', unsafe_allow_html=True)
        
    st.divider()
    
    st.markdown("<span style='font-size:11px;font-weight:700;color:#64748b;letter-spacing:1px;'>SYSTEM SETTINGS</span>", unsafe_allow_html=True)
    st.write("")
    st.selectbox("Hardware Accelerator", ["Auto-detect", "CPU", "CUDA (GPU)", "MPS (Apple Silicon)"], label_visibility="collapsed")

    st.markdown("<div style='margin-top:20px;'><span style='font-size:11px;font-weight:700;color:#64748b;letter-spacing:1px;'>RESOURCE MONITORS</span></div>", unsafe_allow_html=True)
    @st.fragment(run_every="2s")
    def render_resource_monitors():
        vm = psutil.virtual_memory()
        st.caption("Host RAM Allocation")
        st.progress(vm.percent / 100.0, text=f"{vm.used / (1024**3):.1f} GB / {vm.total / (1024**3):.1f} GB")

        total, used, free = shutil.disk_usage("/")
        st.caption("Root Disk Storage")
        st.progress(used / total, text=f"{used / (1024**3):.1f} GB / {total / (1024**3):.1f} GB")

    render_resource_monitors()

# ============================================================================
# PAGE: NEW TRAINING RUN
# ============================================================================
def render_new_run_page():
    st.markdown(
        """
        <div class="sc-banner">
            <h1>Initialize Training Pipeline</h1>
            <p>Configure and deploy high-fidelity tabular generative engines.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ds_tick = " <span style='color:#10b981;'>✅</span>" if st.session_state.selected_dataset else ""
    st.markdown(f'<div class="sc-section-title">Model Architecture & Dataset Selection{ds_tick}</div>', unsafe_allow_html=True)
    
    col_model, _ = st.columns([1, 2])
    with col_model:
        model_options = {
            "Latent-Diffusion 2.2": "Latent-Diffusion 2.2 (SOTA Tabular Synthesis)",
            "GAN 2.1": "GAN 2.1 (Adversarial Baseline)",
            "Vanilla VAE 1.3": "Vanilla VAE 1.3 (Fast Probabilistic Autoencoder)",
            "InfoVAE-MMD 1.5": "InfoVAE-MMD 1.5 (MMD Regularized Space)"
        }
        st.selectbox("Select Generative Framework", list(model_options.keys()), format_func=lambda x: model_options[x], label_visibility="collapsed")

    st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)

    grid_cols = st.columns(4)
    for i, key in enumerate(AVAILABLE_DATASETS):
        meta = DATASET_META[key]
        cols_schema = get_dataset_schema(key)
        col_count = len(cols_schema) if cols_schema else "Unknown"
        
        with grid_cols[i % 4]:
            with st.container(border=True):
                st.markdown(f'<div class="sc-card-title">{meta["label"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="sc-card-desc">{meta["desc"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="sc-card-meta">{col_count} attributes</div>', unsafe_allow_html=True)
                if st.session_state.selected_dataset == key:
                    st.markdown('<div style="text-align:center;margin-top:12px;"><span class="sc-badge sc-badge-selected">Selected</span></div>', unsafe_allow_html=True)
                else:
                    if st.button("Select", key=f"select_{key}", width="stretch"):
                        st.session_state.selected_dataset = key
                        st.session_state.custom_file_path = None
                        st.session_state.custom_file_name = None
                        st.rerun()

    st.markdown('<div class="sc-divider-label">OR UPLOAD CUSTOM DATASET</div>', unsafe_allow_html=True)

    with st.container(border=True):
        uploaded_file = st.file_uploader("Ingest local CSV dataset", type=["csv", "txt", "data"], label_visibility="collapsed")
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

    # ---------------- PRE-FLIGHT EDA PROFILER ----------------
    with st.expander("📊 Pre-flight Data Profiler (EDA)"):
        with st.spinner("Analyzing dataset distribution..."):
            df_preview = fetch_dataset_preview(st.session_state.selected_dataset)
        
        if df_preview is not None and not df_preview.empty:
            st.markdown("**Data Snapshot (Top rows)**")
            st.dataframe(df_preview.head(10), use_container_width=True)
            c_desc, c_info = st.columns(2)
            with c_desc:
                st.markdown("**Numerical Distribution Summary**")
                st.dataframe(df_preview.describe(), use_container_width=True)
            with c_info:
                st.markdown("**Feature Data Types & Missing Values**")
                info_df = pd.DataFrame({"Data Type": df_preview.dtypes.astype(str), "Missing Values": df_preview.isna().sum(), "Missing %": (df_preview.isna().sum() / len(df_preview) * 100).round(2)})
                st.dataframe(info_df, use_container_width=True)
        else:
            st.info("EDA preview not available for this dataset format.")
            
    render_configuration_section()


def render_configuration_section():
    selected_dataset = st.session_state.selected_dataset
    label = DATASET_META.get(selected_dataset, {}).get("label", "Custom Dataset")
    
    loaded_ctx = get_cfg(("diffusion", "context_injection", "context_columns"), [])
    ctx_tick = " <span style='color:#10b981;'>✅</span>" if loaded_ctx or 'ui_ctx' in st.session_state else ""
    
    st.markdown(f'<div class="sc-section-title">Contextual Injection Constraints &middot; {label}{ctx_tick}</div>', unsafe_allow_html=True)
    st.caption("Select columns to act as strict conditioning anchors during the diffusion synthesis process.")

    context_columns_list = []
    schema_cols = get_dataset_schema(selected_dataset)
    
    if schema_cols:
        context_columns_list = st.multiselect(f"Select Anchor Columns (Available: {len(schema_cols)}):", options=schema_cols, default=[c for c in loaded_ctx if c in schema_cols], key="ui_ctx")
    else:
        ctx_cols_input = st.text_input("Context Columns (Comma-separated)", placeholder="e.g., job, marital", key="ui_ctx")
        context_columns_list = [c.strip() for c in ctx_cols_input.split(",") if c.strip()]

    # ---------------- CONFIG IMPORT & WORKBENCH ----------------
    wb_tick = " <span style='color:#10b981;'>✅</span>" if st.session_state.config_saved or st.session_state.loaded_config else ""
    
    st.markdown(f'<div class="sc-section-title" style="margin-bottom:8px;">Hyperparameter Workbench{wb_tick}</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;font-size:14px;margin-bottom:16px;'>Upload a YAML configuration to auto-fill the environment, OR manually tune the engines below.</p>", unsafe_allow_html=True)
    
    c_help, c_upload = st.columns([1, 4], vertical_alignment="bottom")
    with c_help:
        with st.popover("ℹ️ YAML Format Guide", use_container_width=True):
            st.markdown("Upload a `.yaml` file to auto-populate the workbench. Required structure:")
            st.code(YAML_TEMPLATE, language="yaml")
            st.download_button("Download Blank Template", YAML_TEMPLATE, file_name="default_template.yaml", use_container_width=True)
            
    with c_upload:
        uploaded_yaml = st.file_uploader("Load Config File", type=["yaml", "yml"], label_visibility="collapsed")
        if uploaded_yaml:
            try:
                st.session_state.loaded_config = yaml.safe_load(uploaded_yaml)
                st.success("Configuration Loaded Successfully.")
            except Exception as e:
                st.error("Invalid YAML file format.")

    st.write("")

    # --- CLEAN SWAP LOGIC ---
    if st.session_state.loaded_config:
        st.info("✅ Workbench settings locked via imported YAML configuration.")
        if st.button("Clear Configuration (Manual Override)", icon="🗑️"):
            st.session_state.loaded_config = {}
            st.rerun()
            
        epochs = int(get_cfg(("vae", "epochs"), 1000))
        diffusion_epochs = int(get_cfg(("diffusion", "epochs"), 2000))
        vae_lr = float(get_cfg(("vae", "learning_rate"), 0.001))
        diff_lr = float(get_cfg(("diffusion", "learning_rate"), 0.001))
        batch_size = int(get_cfg(("vae", "batch_size"), 256))
        latent_mode = str(get_cfg(("vae", "latent", "mode"), "adaptive")).capitalize()
        latent_heuristic = None
        latent_dim = int(get_cfg(("vae", "latent", "latent_dim"), 256))
        encoder_dims = str(get_cfg(("vae", "latent", "encoder_dims"), [2048, 1024, 512]))
        decoder_dims = str(get_cfg(("vae", "latent", "decoder_dims"), [512, 1024, 2048]))
        mmd_mode = str(get_cfg(("vae", "mmd", "mode"), "constant")).capitalize()
        mmd_config = st.session_state.loaded_config.get("vae", {}).get("mmd", {"mode": mmd_mode.lower(), "weight": 500.0})
        activation = get_cfg(("vae", "activation"), "LeakyReLU")
        dropout = float(get_cfg(("vae", "dropout"), 0.0))
        use_layernorm = bool(get_cfg(("vae", "layernorm"), True))
        diffusion_architecture = str(get_cfg(("diffusion", "architecture", "type"), "standard")).capitalize()
        num_diffusion_steps = int(get_cfg(("diffusion", "num_diffusion_steps"), 2000))
        diffusion_hidden_dims = str(get_cfg(("diffusion", "architecture", "hidden_dims"), "[512, 512, 256]"))
        diffusion_dropout = float(get_cfg(("diffusion", "architecture", "dropout"), 0.1))
        noise_schedule = str(get_cfg(("diffusion", "noise_schedule", "type"), "linear")).capitalize()
        noise_config = st.session_state.loaded_config.get("diffusion", {}).get("noise_schedule", {"type": noise_schedule.lower(), "beta_start": 0.0001, "beta_end": 0.02})
        time_embedding_type = str(get_cfg(("diffusion", "time_conditioning", "embedding"), "sinusoidal")).capitalize()
        time_conditioning = str(get_cfg(("diffusion", "time_conditioning", "method"), "film")).capitalize()
        if time_conditioning == "Film": time_conditioning = "FiLM"
        time_embedding_dim = int(get_cfg(("diffusion", "time_conditioning", "embedding_dim"), 128))
            
    else:
        tab_training, tab_vae, tab_diffusion = st.tabs(["Optimization Parameters", "Manifold (InfoVAE)", "Latent Diffusion Network"])

        with tab_training:
            epochs = st.number_input("VAE Epochs", min_value=1, value=1000, step=100)
            diffusion_epochs = st.number_input("Diffusion Epochs", min_value=1, value=2000, step=100)
            vae_lr = st.number_input("VAE Learning Rate", min_value=0.00001, value=0.001, step=0.0001, format="%.5f")
            diff_lr = st.number_input("Diffusion Learning Rate", min_value=0.00001, value=0.001, step=0.0001, format="%.5f")
            batch_size = st.number_input("Batch Size", min_value=1, value=256, step=32)

        with tab_vae:
            latent_mode = st.radio("Latent Dimension Strategy", ["Adaptive", "Custom"], horizontal=True)
            if latent_mode == "Adaptive":
                latent_heuristic = st.selectbox("Heuristic Model", ["Google", "FastAI", "FH"])
                latent_dim, encoder_dims, decoder_dims = None, None, None
                st.caption("Latent geometry will be inferred dynamically from the input tabular space.")
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
            if mmd_mode == "Constant": mmd_config["weight"] = st.number_input("MMD Base Weight", value=500.0, step=10.0)
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
                mmd_config["target_ratio"] = st.number_input("Target Reconstruct Ratio", min_value=0.01, value=0.20, step=0.01)
                mmd_config["momentum"] = st.number_input("Momentum Term", min_value=0.0, max_value=0.999, value=0.95, step=0.01)

            st.divider()
            c3, c4 = st.columns(2)
            activation = c3.selectbox("Activation Function", ["SiLU", "ReLU", "GELU", "LeakyReLU"], index=3)
            dropout = c4.number_input("Dropout Rate", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
            use_layernorm = st.checkbox("Enable LayerNorm", value=True)

        with tab_diffusion:
            diffusion_architecture = st.selectbox("Network Topology", ["Standard", "Residual", "TabDiff", "Custom"])
            num_diffusion_steps = st.number_input("Markov Timesteps ($T$)", min_value=1, value=2000, step=100)

            if diffusion_architecture == "Standard": diffusion_hidden_dims = "[512, 512, 256]"
            elif diffusion_architecture == "Residual": diffusion_hidden_dims = "[512, 512, 512, 512, 256]"
            elif diffusion_architecture == "TabDiff": diffusion_hidden_dims = "[1024, 1024, 1024, 512, 512]"
            else: diffusion_hidden_dims = st.text_input("Hidden Layers", "[512, 512, 256]")

            if diffusion_architecture != "Custom": st.text_input("Hidden Layers (Locked)", value=diffusion_hidden_dims, disabled=True)
                
            diffusion_dropout = st.number_input("Denoising Dropout", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

            st.markdown("**Noise Beta Schedule**")
            noise_schedule = st.selectbox("Variance Scheduler", ["Linear", "Cosine", "Quadratic", "Sigmoid", "Learnable"])
            noise_config = {"type": noise_schedule.lower()}
            if noise_schedule != "Learnable":
                c5, c6 = st.columns(2)
                noise_config["beta_start"] = c5.number_input("$\\beta_{start}$", min_value=0.0, value=0.0001, format="%.5f")
                noise_config["beta_end"] = c6.number_input("$\\beta_{end}$", min_value=0.0, value=0.0200, step=0.005, format="%.5f")
            else:
                st.caption("Schedule parameters will be optimized directly via gradient descent.")

            st.markdown("**Temporal Conditioning**")
            c7, c8 = st.columns(2)
            time_embedding_type = c7.selectbox("Temporal Embedding", ["Sinusoidal", "Learned", "Fourier"])
            time_conditioning = c8.selectbox("Modulation Approach", ["Addition", "FiLM", "Concatenation"], index=1)
            time_embedding_dim = st.number_input("Temporal Projection Dim", min_value=8, value=128, step=8)

    # ---------------- GENERATION BLUEPRINT ----------------
    st.markdown('<div class="sc-section-title" style="margin-top:32px;">Data Generation Blueprint</div>', unsafe_allow_html=True)
    if not context_columns_list:
        st.info("Unconditional Regime: The diffusion model will sample dynamically from the learned prior without constraints.")
        total_rows = st.number_input("Volume of samples to synthesize:", min_value=100, value=1000, step=100)
    else:
        st.markdown("<div style='color:#94a3b8;font-size:14px;margin-bottom:12px;'>Declare structural values to steer generation across the synthesized batch.</div>", unsafe_allow_html=True)
        blueprint_values = {}
        input_columns = st.columns(min(len(context_columns_list), 4))
        for i, col in enumerate(context_columns_list):
            with input_columns[i % 4]:
                blueprint_values[col] = st.text_input(f"Target '{col}':", key=f"bp_{col}")
        total_rows = st.number_input("Volume of samples to synthesize:", min_value=1, max_value=100000, value=10, step=1)

    # ---------------- SYSTEM EXECUTION ----------------
    st.markdown('<div class="sc-section-title" style="margin-top:40px;">Execution Engine</div>', unsafe_allow_html=True)
    
    dataset_url = "local" if selected_dataset == "custom" else DATASET_MAP[selected_dataset]
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

    # Perfectly balanced [1,1] execution columns
    c_run, c_export = st.columns(2)

    with c_run:
        run_model = st.button("Initialize Generative Pipeline", use_container_width=True, type="primary")
        
    with c_export:
        yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False, indent=4)
        st.download_button("Export Configuration Setup", data=yaml_str, file_name=f"synthlab_{selected_dataset}_cfg.yaml", mime="text/yaml", use_container_width=True)

    if run_model:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            f.write(yaml_str)
        st.session_state.config_saved = True
        
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
            if os.path.exists(blueprint_path): os.remove(blueprint_path)

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

    try: experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    except Exception: experiment = None

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

                if detail_view == "Generated Data":
                    all_paths = fetch_mlflow_all_artifacts(mlflow_uri, run_id)
                    csv_paths = [p for p in all_paths if p.endswith(".csv")]
                    if csv_paths:
                        for cp in csv_paths:
                            with tempfile.TemporaryDirectory() as tmp_dir:
                                local_path = client.download_artifacts(run_id, cp, tmp_dir)
                                with open(local_path, "rb") as f:
                                    st.download_button(label=f"📥 Download {os.path.basename(cp)}", data=f.read(), file_name=os.path.basename(cp), mime="text/csv", key=f"dl_csv_{run_id}_{cp}")
                                try:
                                    st.dataframe(pd.read_csv(local_path, nrows=100), use_container_width=True, height=300)
                                except Exception: pass
                    else: st.info("No synthetic output files (`.csv`) found.")

                elif detail_view == "Model Parameters":
                    if run.data.params: st.dataframe(pd.DataFrame(list(run.data.params.items()), columns=["Parameter Key", "Configured Value"]), width="stretch", hide_index=True, height=350)
                    else: st.caption("No parameters logged.")

                elif detail_view == "Generative Accuracy":
                    if run.data.metrics:
                        final_metrics = {k: v for k, v in run.data.metrics.items() if k.startswith("test_")}
                        if final_metrics: st.dataframe(pd.DataFrame(list(final_metrics.items()), columns=["Evaluation Metric", "Score"]), width="stretch", hide_index=True, height=280)
                        else: st.caption("No final test accuracy metrics found in logs.")
                    else: st.caption("No metrics logged.")

                elif detail_view == "Latent Dynamics":
                    if run.data.metrics:
                        latent_metrics = {k: v for k, v in run.data.metrics.items() if not k.startswith("test_")}
                        if latent_metrics: st.dataframe(pd.DataFrame(list(latent_metrics.items()), columns=["Diagnostic Metric", "Computed Value"]), use_container_width=True, hide_index=True, height=350)
                        else: st.caption("No internal latent space diagnostics found.")
                    else: st.caption("No metrics logged.")

                elif detail_view == "Visualizations":
                    with st.spinner("Fetching plots from artifact store..."):
                        all_images = fetch_mlflow_images(mlflow_uri, run_id)
                    if not all_images: st.info("No visualization plots generated for this run.")
                    else:
                        loss_plots = [img for img in all_images if "loss" in img.lower()]
                        heatmap_plots = [img for img in all_images if "heatmap" in img.lower()]
                        num_dist_plots = [img for img in all_images if "dist_num" in img.lower()]
                        cat_dist_plots = [img for img in all_images if "dist_cat" in img.lower()]

                        if loss_plots:
                            with st.expander("Convergence & Loss Trajectories", expanded=True): render_image_grid(loss_plots, mlflow_uri, run_id)
                        if heatmap_plots:
                            with st.expander("Feature Correlation Alignment", expanded=True): render_image_grid(heatmap_plots, mlflow_uri, run_id)
                        if num_dist_plots:
                            with st.expander("Marginal Distributions (Numerical)", expanded=False): render_image_grid(num_dist_plots, mlflow_uri, run_id)
                        if cat_dist_plots:
                            with st.expander("Categorical Distributions", expanded=False): render_image_grid(cat_dist_plots, mlflow_uri, run_id)

                elif detail_view == "Terminal Report":
                    render_keyword_artifact_section(client, run_id, mlflow_uri, keyword="report", empty_message="No text reports found.")
        st.divider()

# ============================================================================
# PAGE: AUDITING
# ============================================================================
def render_auditing_page():
    st.markdown(
        """
        <div class="sc-banner">
            <h1>Auditing & Utility Evaluation</h1>
            <p>Assess privacy, utility, and statistical fidelity of synthetic datasets against their real-world counterparts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown('<div class="sc-section-title">Configure Evaluation Target</div>', unsafe_allow_html=True)
    
    c_real, c_synth = st.columns(2)
    with c_real:
        st.markdown("**1. Source Data (Real)**")
        st.selectbox("Reference Dataset", AVAILABLE_DATASETS + ["Custom Upload"], key="audit_real")
    with c_synth:
        st.markdown("**2. Generated Data (Synthetic)**")
        st.selectbox("Select MLFlow Run", ["Latest Run"] + [f"Run {i}" for i in range(1, 6)], key="audit_synth", help="Orchestrates data from eval_metrics.py")
        
    st.write("")
    if st.button("Execute Statistical Audit", use_container_width=True, type="primary"):
        with st.spinner("Orchestrating eval_metrics via test.py..."):
            try:
                subprocess.run([sys.executable, "test.py"], cwd=ROOT_DIR, check=True)
                st.success("Audit Complete! Report saved to destination.")
            except subprocess.CalledProcessError as e:
                st.error(f"Audit Script Crashed: {e}")
            except FileNotFoundError as e:
                st.error("Could not find test.py in the root directory.")

# ============================================================================
# PAGE: DOCUMENTATION (KNOWLEDGE BASE)
# ============================================================================
def render_documentation_page():
    st.markdown(
        """
        <div class="sc-banner">
            <h1>Platform Knowledge Base</h1>
            <p>Reference guides for generative architectures, statistical evaluation metrics, and latent diagnostics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    with st.expander("🧠 Generative Architectures Overview", expanded=True):
        st.markdown("""
        **1. Latent-Diffusion 2.2**
        The state-of-the-art framework. It compresses tabular data into a continuous latent space using a VAE, and then iteratively adds and removes Gaussian noise using a Markov chain process. It handles highly dimensional data with exceptional fidelity.
        
        **2. GAN 2.1 (Generative Adversarial Network)**
        A baseline adversarial setup. Two networks (a Generator and a Discriminator) are trained in a min-max game. While powerful, GANs are notoriously prone to *mode collapse* (failing to represent the full diversity of the real dataset).
        
        **3. Vanilla VAE 1.3 (Variational Autoencoder)**
        A standard probabilistic autoencoder. It learns to map inputs to a smooth normal distribution (using Kullback-Leibler divergence). It is incredibly fast but often produces blurry or overly-smoothed synthetic distributions.
        
        **4. InfoVAE-MMD 1.5**
        An upgrade over the Vanilla VAE. Instead of relying purely on KL divergence, it uses Maximum Mean Discrepancy (MMD) as a kernel-based distance metric to ensure the *aggregate* generated data perfectly matches the *prior* distribution. This prevents the mode collapse common in standard VAEs.
        """)
        
    with st.expander("📊 Evaluation Metrics Glossary (Utility & Fidelity)", expanded=True):
        st.markdown("""
        SynthLab 2.0 uses a rigorous suite of metrics to audit synthetic data:
        
        *   **Shape Error:** Measures the difference in the marginal distributions (1D histograms) between real and synthetic features. A lower score means the synthetic data perfectly mimics the column-by-column statistics of the real data.
        *   **Trend Error:** Evaluates feature-to-feature correlations (2D interactions). If Feature A goes up when Feature B goes down in the real data, Trend Error checks if the synthetic data captured that exact relationship.
        *   **Alpha-Precision:** Represents the *Fidelity* or accuracy of the synthetic data. It measures whether the synthetic records logically fall within the actual support/boundaries of the real-world dataset. High precision means no "impossible" records were generated.
        *   **Beta-Recall:** Represents the *Diversity* or coverage of the synthetic data. High recall guarantees that the synthetic data covers all the edge cases and minorities present in the real dataset, rather than just copying the average profile.
        *   **Normalized MSE:** Measures the Mean Squared Error for mathematical reconstruction, scaled down to allow fair comparisons across wildly different datasets.
        """)

    with st.expander("🧬 Understanding the Latent Diagnostic Report", expanded=True):
        st.markdown("""
        When training VAE/Diffusion pipelines, SynthLab generates a `Latent Dynamics` report. This is essentially an MRI of the model's brain.
        
        **What to look for:**
        *   **Active Dimensions:** Out of the hundreds of latent dimensions you allocated, how many is the model *actually* using? If you allocated 256 dimensions but the Active Dimensions count is 5, you have severe posterior collapse.
        *   **Latent MMD Loss:** The mathematical distance between the model's output distribution and the target distribution. You want this number to steadily approach zero.
        *   **Reconstruction Loss:** Evaluates how well the Decoder can rebuild the original input from the compressed latent vector. 
        """)

# ============================================================================
# ROUTING
# ============================================================================
if st.session_state.nav == "New Training Run":
    render_new_run_page()
elif st.session_state.nav == "Previous Runs":
    render_previous_runs_page()
elif st.session_state.nav == "Auditing":
    render_auditing_page()
elif st.session_state.nav == "Documentation":
    render_documentation_page()
