import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler, 
    QuantileTransformer
)

class TabularDataPreprocessor:
    def __init__(
        self, 
        continuous_cols: list, 
        categorical_cols: list,
        continuous_scaler: str = "standard",   # "standard", "minmax", "robust", "quantile_normal", "quantile_uniform"
        categorical_encoding: str = "one_hot", # "one_hot", "ordinal"
        clip_outliers: bool = False,            # Applies Winsorization (1st-99th percentile capping)
        impute_missing: bool = True            # Automatically handles NaNs
    ):
        """
        A highly configurable preprocessing engine for mixed-type tabular datasets,
        specifically optimized for probabilistic generative models like VAEs.
        """
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.continuous_scaler = continuous_scaler
        self.categorical_encoding = categorical_encoding
        self.clip_outliers = clip_outliers
        self.impute_missing = impute_missing
        
        # 1. Initialize Continuous Scaler
        if self.continuous_scaler == "standard":
            self.scaler = StandardScaler()
        elif self.continuous_scaler == "robust":
            # Uses median and IQR; heavily resistant to severe outliers
            self.scaler = RobustScaler()
        elif self.continuous_scaler == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.continuous_scaler == "quantile_normal":
            # Forces arbitrary distributions into a smooth Gaussian bell curve
            self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        elif self.continuous_scaler == "quantile_uniform":
            self.scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
        else:
            self.scaler = None

        # Metadata tracking for inverse transformations and neural network dimension sizing
        self.clipping_bounds = {}
        self.imputation_values = {}
        self.categories_per_col = {}
        self.cardinalities = []

    def fit(self, df: pd.DataFrame):
        """Learns statistical parameters, clipping boundaries, and categorical structures."""
        
        # --- Continuous Features ---
        if self.continuous_cols:
            df_cont = df[self.continuous_cols].copy()
            
            # Impute by mean
            if self.impute_missing:
                for col in self.continuous_cols:
                    self.imputation_values[col] = df_cont[col].mean()
                    df_cont[col] = df_cont[col].fillna(self.imputation_values[col])
            else:
                df_cont = df_cont.fillna(0.0)

            # Learn clipping bounds (1st and 99th percentiles)
            if self.clip_outliers:
                for col in self.continuous_cols:
                    lower = df_cont[col].quantile(0.01)
                    upper = df_cont[col].quantile(0.99)
                    self.clipping_bounds[col] = (lower, upper)
                    df_cont[col] = df_cont[col].clip(lower, upper)
                    
            if self.scaler is not None:
                self.scaler.fit(df_cont)

        # --- Categorical Features ---
        self.categories_per_col = {}
        self.cardinalities = []
        
        for col in self.categorical_cols:
            col_data = df[col].copy()
            
            # Impute by mode
            if self.impute_missing:
                mode_val = col_data.mode()[0] if not col_data.mode().empty else "Missing"
                self.imputation_values[col] = mode_val
                col_data = col_data.fillna(mode_val)
            
            # Track unique categories sequentially
            unique_cats = sorted(col_data.dropna().unique().tolist())
            self.categories_per_col[col] = unique_cats
            self.cardinalities.append(len(unique_cats))
            
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms a DataFrame into a combined numerical matrix."""
        processed_blocks = []
        
        # --- Continuous Features ---
        if self.continuous_cols:
            df_cont = df[self.continuous_cols].copy()
            
            if self.impute_missing:
                for col in self.continuous_cols:
                    df_cont[col] = df_cont[col].fillna(self.imputation_values[col])
            else:
                df_cont = df_cont.fillna(0.0)
                
            if self.clip_outliers:
                for col in self.continuous_cols:
                    lower, upper = self.clipping_bounds[col]
                    df_cont[col] = df_cont[col].clip(lower, upper)
                    
            if self.scaler is not None:
                scaled_cont = self.scaler.transform(df_cont)
            else:
                scaled_cont = df_cont.values
                
            processed_blocks.append(scaled_cont)

        # --- Categorical Features ---
        for col in self.categorical_cols:
            col_data = df[col].copy()
            
            if self.impute_missing:
                col_data = col_data.fillna(self.imputation_values[col])
            else:
                fallback = self.categories_per_col[col][0]
                col_data = col_data.fillna(fallback)
                
            vals = col_data.values
            num_cats = len(self.categories_per_col[col])
            
            if self.categorical_encoding == "one_hot":
                # Create hard binary matrices (best for VAE categorical cross-entropy loss)
                one_hot = np.zeros((len(df), num_cats))
                for idx, val in enumerate(vals):
                    cat_idx = self.categories_per_col[col].index(val) if val in self.categories_per_col[col] else 0
                    one_hot[idx, cat_idx] = 1.0
                processed_blocks.append(one_hot)
                
            elif self.categorical_encoding == "ordinal":
                # Create a single column of integer codes
                codes = np.array([
                    self.categories_per_col[col].index(val) if val in self.categories_per_col[col] else 0
                    for val in vals
                ], dtype=float).reshape(-1, 1)
                processed_blocks.append(codes)
                
        return np.hstack(processed_blocks) if processed_blocks else np.empty((len(df), 0))

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
