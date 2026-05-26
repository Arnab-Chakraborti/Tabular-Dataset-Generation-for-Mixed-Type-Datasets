import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer


class TabularDataPreprocessor:
    def __init__(
        self, 
        continuous_cols: list, 
        categorical_cols: list,
        scaling_method: str = "minmax",       # Options: "minmax", "standard", "quantile", None
        clip_outliers: bool = True,           # Toggle Quantile Clipping (Winsorization)
        lower_quantile: float = 0.01,         # Lower outlier limit boundary
        upper_quantile: float = 0.99,         # Upper outlier limit boundary
        categorical_format: str = "one_hot"   # Options: "one_hot", "ordinal"
    ):
        """
        A comprehensive, modular preprocessor engine designed to handle mixed-type 
        tabular datasets for deep generative networks.
        """
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.scaling_method = scaling_method
        self.clip_outliers = clip_outliers
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.categorical_format = categorical_format
        
        # Dictionary to lock clipping thresholds determined on the training data
        self.clipping_bounds = {}
        
        # Map structural dictionary elements for categorical decoding tracks
        self.categories_per_col = {}
        self.cardinalities = []
        
        # Configure the chosen continuous scaler engine
        if self.scaling_method == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.scaling_method == "quantile":
            # Maps arbitrary shapes directly to a uniform distribution between -1 and 1
            self.scaler = QuantileTransformer(n_quantiles=1000, output_distribution='uniform')
        else:
            self.scaler = None

    def fit(self, df: pd.DataFrame):
        """
        Learns statistical parameters, outlier clipping limits, and categorical class structures.
        """
        # 1. Process continuous data distributions
        if self.continuous_cols:
            df_cont = df[self.continuous_cols].fillna(0.0).copy()
            
            # Learn and execute clipping limits if enabled
            if self.clip_outliers:
                for col in self.continuous_cols:
                    lower_bound = df_cont[col].quantile(self.lower_quantile)
                    upper_bound = df_cont[col].quantile(self.upper_quantile)
                    self.clipping_bounds[col] = (lower_bound, upper_bound)
                    df_cont[col] = df_cont[col].clip(lower_bound, upper_bound)
            
            # Fit the configured scaling algorithm
            if self.scaler is not None:
                self.scaler.fit(df_cont)

        # 2. Map categorical features and cache structural sizing
        self.categories_per_col = {}
        self.cardinalities = []
        for col in self.categorical_cols:
            unique_cats = sorted(df[col].dropna().unique().tolist())
            self.categories_per_col[col] = unique_cats
            self.cardinalities.append(len(unique_cats))
            
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms an incoming mixed-type DataFrame into a single numerical matrix layout 
        based on your designated continuous and categorical scaling options.
        """
        processed_blocks = []
        
        # 1. Continuous transformation pipeline
        if self.continuous_cols:
            df_cont = df[self.continuous_cols].fillna(0.0).copy()
            
            if self.clip_outliers:
                for col in self.continuous_cols:
                    lower_bound, upper_bound = self.clipping_bounds[col]
                    df_cont[col] = df_cont[col].clip(lower_bound, upper_bound)
                    
            if self.scaler is not None:
                scaled_continuous = self.scaler.transform(df_cont)
                # If using QuantileTransformer, scale from [0, 1] to [-1, 1] manually
                if self.scaling_method == "quantile":
                    scaled_continuous = (scaled_continuous * 2.0) - 1.0
            else:
                scaled_continuous = df_cont.values
                
            processed_blocks.append(scaled_continuous)

        # 2. Categorical processing pipeline
        for col in self.categorical_cols:
            fallback_val = self.categories_per_col[col][0]
            col_data = df[col].fillna(fallback_val).values
            
            if self.categorical_format == "one_hot":
                num_categories = len(self.categories_per_col[col])
                one_hot = np.zeros((len(df), num_categories))
                for idx, val in enumerate(col_data):
                    cat_idx = self.categories_per_col[col].index(val) if val in self.categories_per_col[col] else 0
                    one_hot[idx, cat_idx] = 1.0
                processed_blocks.append(one_hot)
                
            elif self.categorical_format == "ordinal":
                cat_codes = np.array([
                    self.categories_per_col[col].index(val) if val in self.categories_per_col[col] else 0
                    for val in col_data
                ], dtype=float).reshape(-1, 1)
                processed_blocks.append(cat_codes)
                
        return np.hstack(processed_blocks) if processed_blocks else np.empty((len(df), 0))

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
