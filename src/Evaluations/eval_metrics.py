import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def compute_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Computes Bias-Corrected Cramér's V for two categorical columns.
    """
    confusion_matrix = pd.crosstab(x, y).values
    n = confusion_matrix.sum()
    if n == 0:
        return 0.0
        
    r, k = confusion_matrix.shape
    if r <= 1 or k <= 1:
        return 0.0

    # Calculate standard Chi-Square statistic
    # Using a manual calculation to maintain stability with small/empty categories
    expected = (confusion_matrix.sum(axis=1, keepdims=True) * confusion_matrix.sum(axis=0, keepdims=True)) / n
    # Avoid division by zero for unpopulated combinations
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((confusion_matrix - expected) ** 2 / expected)

    phi2 = chi2 / n
    
    # Apply bias correction formulas
    phi2_corrected = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corrected = r - ((r - 1) ** 2) / (n - 1)
    k_corrected = k - ((k - 1) ** 2) / (n - 1)
    
    denominator = min(k_corrected - 1, r_corrected - 1)
    if denominator <= 0:
        return 0.0
        
    return float(np.sqrt(phi2_corrected / denominator))


def evaluate_generator_performance(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, k: int = 5) -> dict:
    """
    Parameters:
    -----------
    real_df : pd.DataFrame
        The original mixed-type baseline dataset.
    synthetic_df : pd.DataFrame
        The generated mixed-type synthetic dataset.
    k : int, default=5
        The neighborhood constraint used to evaluate high-dimensional manifold radii.
        
    Returns:
    --------
    dict
        A summary tracking dictionary matching the reference guidelines.
    """
    # Create copies to safely process without altering original DataFrames
    df_r = real_df.copy()
    df_s = synthetic_df.copy()
    
    all_cols = df_r.columns.tolist()
    M = len(all_cols)
    
    # Segregate columns based on type
    num_cols = df_r.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [col for col in all_cols if col not in num_cols]

    # ---------------------------------------------------------
    # 1. SHAPE ERROR EVALUATION (Equation 4)
    # ---------------------------------------------------------
    shape_distances = []
    
    # Numerical Columns: Two-Sample Kolmogorov-Smirnov Distance
    for col in num_cols:
        d_ks, _ = ks_2samp(df_r[col].dropna(), df_s[col].dropna())
        shape_distances.append(d_ks)
        
    # Categorical Columns: Total Variation Distance (TVD)
    for col in cat_cols:
        # Compute relative empirical frequencies for both spaces
        p_r = df_r[col].value_counts(normalize=True)
        p_s = df_s[col].value_counts(normalize=True)
        
        # Align indexes to ensure identical structural dimensions
        all_categories = p_r.index.union(p_s.index)
        p_r = p_r.reindex(all_categories, fillvalue=0.0).values
        p_s = p_s.reindex(all_categories, fillvalue=0.0).values
        
        # Equation 3 calculation
        tvd = 0.5 * np.sum(np.abs(p_r - p_s))
        shape_distances.append(tvd)
        
    shape_error_pct = (np.sum(shape_distances) / M) * 100.0

    # ---------------------------------------------------------
    # 2. TREND ERROR EVALUATION 
    # ---------------------------------------------------------
    # Initialize clean, parallel relationship matrix blocks
    A_real = np.zeros((M, M))
    A_synth = np.zeros((M, M))
    
    # Compute association parameters across all feature pairs
    for u in range(M):
        for v in range(M):
            if u == v:
                A_real[u, v] = 1.0
                A_synth[u, v] = 1.0
                continue
                
            col_u, col_v = all_cols[u], all_cols[v]
            
            # Scenario A: Numerical-to-Numerical (Pearson)
            if col_u in num_cols and col_v in num_cols:
                # Use fillna to protect linear coefficients against isolated nan records
                r_val = df_r[[col_u, col_v]].corr(method='pearson').iloc[0, 1]
                s_val = df_s[[col_u, col_v]].corr(method='pearson').iloc[0, 1]
                A_real[u, v] = 0.0 if np.isnan(r_val) else r_val
                A_synth[u, v] = 0.0 if np.isnan(s_val) else s_val
                
            # Scenario B: Categorical-to-Categorical (Cramér's V)
            elif col_u in cat_cols and col_v in cat_cols:
                A_real[u, v] = compute_cramers_v(df_r[col_u], df_r[col_v])
                A_synth[u, v] = compute_cramers_v(df_s[col_u], df_s[col_v])
                
            # Scenario C: Mixed Interactions (Absolute Spearman's Rank)
            else:
                # Ensure the numerical item maps to index 0, categorical to index 1
                r_num_data = df_r[col_u] if col_u in num_cols else df_r[col_v]
                r_cat_data = df_r[col_v] if col_v in cat_cols else df_r[col_u]
                
                s_num_data = df_s[col_u] if col_u in num_cols else df_s[col_v]
                s_cat_data = df_s[col_v] if col_v in cat_cols else df_s[col_u]
                
                # Convert categorical sequences temporarily to codes to execute rank tracking
                r_rho, _ = spearmanr(r_num_data, pd.Series(r_cat_data).astype('category').cat.codes)
                s_rho, _ = spearmanr(s_num_data, pd.Series(s_cat_data).astype('category').cat.codes)
                
                A_real[u, v] = 0.0 if np.isnan(r_rho) else abs(r_rho)
                A_synth[u, v] = 0.0 if np.isnan(s_rho) else abs(s_rho)
                
    # Extract Upper Triangles to drop identities and symmetric duplicates (Equation 7)
    tri_u, tri_v = np.triu_indices(M, k=1)
    trend_error_pct = np.mean(np.abs(A_real[tri_u, tri_v] - A_synth[tri_u, tri_v])) * 100.0

    # ---------------------------------------------------------
    # 3. HIGH-DIMENSIONAL GEOMETRY (α-Precision & β-Recall)
    # ---------------------------------------------------------
    # Step A: Enforce identical categorical categories alignment using One-Hot encoding mapping
    # This constructs the standardized space R^D
    combined_df = pd.concat([df_r, df_s], axis=0, keys=['real', 'synth'])
    if len(cat_cols) > 0:
        combined_df = pd.get_dummies(combined_df, columns=cat_cols, drop_first=False)
        
    # Re-extract clean vector chunks
    X_real_raw = combined_df.xs('real').fillna(0.0).values.astype(float)
    X_synth_raw = combined_df.xs('synth').fillna(0.0).values.astype(float)
    
    # Scale space vectors uniformly using a Standard Scaler fitted exclusively to the real data space
    scaler = StandardScaler().fit(X_real_raw)
    X_r_space = scaler.transform(X_real_raw)
    X_s_space = scaler.transform(X_synth_raw)
    
    # Verify neighbor indexing parameters safely against matrix bounds
    k_val = min(k, len(X_r_space) - 1, len(X_s_space) - 1)
    
    # Compute local support boundaries on Real Manifold
    nn_real = NearestNeighbors(n_neighbors=k_val + 1, metric='euclidean').fit(X_r_space)
    distances_real, _ = nn_real.kneighbors(X_r_space)
    radii_real = distances_real[:, -1]

    nn_synth = NearestNeighbors(n_neighbors=k_val + 1, metric='euclidean').fit(X_s_space)
    distances_synth, _ = nn_synth.kneighbors(X_s_space)
    radii_synth = distances_synth[:, -1]
    
    # Calculate α-Precision
    dist_s_to_r, indices_closest_real = nn_real.kneighbors(X_s_space, n_neighbors=1)
    closest_real_radii = radii_real[indices_closest_real.squeeze()]
    # Force arrays to match sizing context if squeeze reduces dimensions to scalar
    if np.isscalar(closest_real_radii):
        closest_real_radii = np.array([closest_real_radii])
    alpha_precision_pct = np.mean(dist_s_to_r.squeeze() <= closest_real_radii) * 100.0
    
    # Calculate β-Recall 
    dist_r_to_s, indices_closest_synth = nn_synth.kneighbors(X_r_space, n_neighbors=1)
    closest_synth_radii = radii_synth[indices_closest_synth.squeeze()]
    if np.isscalar(closest_synth_radii):
        closest_synth_radii = np.array([closest_synth_radii])
    beta_recall_pct = np.mean(dist_r_to_s.squeeze() <= closest_synth_radii) * 100.0


    return {
        "shape_error_pct": float(shape_error_pct),      
        "trend_error_pct": float(trend_error_pct),     
        "alpha_precision_pct": float(alpha_precision_pct),  
        "beta_recall_pct": float(beta_recall_pct)
    }    
