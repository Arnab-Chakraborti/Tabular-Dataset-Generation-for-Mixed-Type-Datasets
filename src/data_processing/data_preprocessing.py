import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler, 
    QuantileTransformer
)

def objectify(df):
    df = df.copy()
    for col in df.columns:
        r = df[col].nunique()
        N = len(df)
        if (r / N) < 0.2:
            # Cast directly to object instead of category to avoid restriction locks
            df[col] = df[col].astype(object)
        if (r / N) >= 0.2:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    return df


class TabularDataPreprocessor:
    def __init__(
        self, 
        continuous_cols: list, 
        categorical_cols: list,
        continuous_scaler: str = "standard",   
        categorical_encoding: str = "one_hot", 
        clip_outliers: bool = False,            
        impute_missing: bool = True            
    ):
        """
        A highly configurable preprocessing engine for mixed-type tabular datasets,
        specifically optimized for probabilistic generative models like VAEs.
        """
        self.continuous_cols = continuous_cols.copy()
        self.categorical_cols = categorical_cols.copy()
        self.continuous_scaler = continuous_scaler
        self.categorical_encoding = categorical_encoding
        self.clip_outliers = clip_outliers
        self.impute_missing = impute_missing
        
        # 1. Initialize Continuous Scaler
        if self.continuous_scaler == "standard":
            self.scaler = StandardScaler()
        elif self.continuous_scaler == "robust":
            self.scaler = RobustScaler()
        elif self.continuous_scaler == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.continuous_scaler == "quantile_normal":
            self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        elif self.continuous_scaler == "quantile_uniform":
            self.scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
        else:
            self.scaler = None

        self.clipping_bounds = {}
        self.imputation_values = {}
        self.categories_per_col = {}
        self.cardinalities = []
        
        # Track which continuous columns dynamically generated missing indicators
        self.missing_indicators = []

    def fit(self, df: pd.DataFrame):
        """Learns statistical parameters, clipping boundaries, and categorical structures."""
        
        # --- Continuous Features ---
        if self.continuous_cols:
            df_cont = df[self.continuous_cols].copy()
            
            if self.impute_missing:
                # 1. Bulk-create missing indicators for true NaNs
                new_indicator_cols = {}
                for col in list(self.continuous_cols):
                    is_missing = df_cont[col].isna()
                    if is_missing.any():
                        indicator_col = f'{col}_is_missing'
                        self.missing_indicators.append(indicator_col)
                        new_indicator_cols[indicator_col] = is_missing.astype(float)
                        if indicator_col not in self.continuous_cols:
                            self.continuous_cols.append(indicator_col)
                            
                # Concatenate all new columns at once to prevent memory fragmentation
                if new_indicator_cols:
                    indicators_df = pd.DataFrame(new_indicator_cols, index=df_cont.index)
                    df_cont = pd.concat([df_cont, indicators_df], axis=1)
                            
                # 2. Impute the original continuous columns with their mean
                for col in self.continuous_cols:
                    if col not in self.missing_indicators:
                        mean_val = df_cont[col].mean()
                        if pd.isna(mean_val):
                            mean_val = 0.0
                        self.imputation_values[col] = df_cont[col].mean()
                        df_cont[col] = df_cont[col].fillna(self.imputation_values[col])
            else:
                df_cont = df_cont.fillna(0.0)

            if self.clip_outliers:
                for col in self.continuous_cols:
                    if col not in self.missing_indicators:  
                        lower = df_cont[col].quantile(0.01)
                        upper = df_cont[col].quantile(0.99)
                        self.clipping_bounds[col] = (lower, upper)
                        df_cont[col] = df_cont[col].clip(lower, upper)
                    
            if self.scaler is not None:
                # ---> FIX 1: Only fit the scaler on actual continuous columns <---
                base_cols = [c for c in self.continuous_cols if c not in self.missing_indicators]
                self.scaler.fit(df_cont[base_cols])

        # --- Categorical Features ---
        self.categories_per_col = {}
        self.cardinalities = []
        
        for col in self.categorical_cols:
            col_data = df[col].copy()
            
            if self.impute_missing:
                col_data = col_data.replace(r'^\s*$', '[miss]', regex=True)
                self.imputation_values[col] = '[miss]'
                col_data = col_data.fillna('[miss]')
            
            unique_cats = sorted(col_data.dropna().unique().tolist(), key=str)
            self.categories_per_col[col] = unique_cats
            self.cardinalities.append(len(unique_cats))
            
        return self

    def getheuristic_fast(self):
        x = len(self.continuous_cols)
        for card in self.cardinalities:
            x += min(50, math.floor(card / 2))
        return x, "FH"

    def getheuristic_google(self):
        x = len(self.continuous_cols)
        for card in self.cardinalities:
            x += math.ceil(card ** (1/4))
        return x, "GH"

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms a DataFrame into a combined numerical matrix."""
        processed_blocks = []
        
        # --- Continuous Features ---
        if self.continuous_cols:
            base_cols = [c for c in self.continuous_cols if c not in self.missing_indicators]
            df_cont = df[base_cols].copy()
            
            if self.impute_missing:
                # 1. Bulk-recreate the exact indicator columns learned during fit()
                new_indicator_cols = {}
                for indicator_col in self.missing_indicators:
                    orig_col = indicator_col.replace('_is_missing', '')
                    if orig_col in df_cont.columns:
                        new_indicator_cols[indicator_col] = df_cont[orig_col].isna().astype(float)
                    else:
                        new_indicator_cols[indicator_col] = 0.0  # Safety fallback
                        
                if new_indicator_cols:
                    indicators_df = pd.DataFrame(new_indicator_cols, index=df_cont.index)
                    df_cont = pd.concat([df_cont, indicators_df], axis=1)
                        
                # 2. Impute base columns using learned means
                for col in base_cols:
                    df_cont[col] = df_cont[col].fillna(self.imputation_values[col])
            else:
                df_cont = df_cont.fillna(0.0)
                
            # Guarantee column order perfectly matches self.continuous_cols before passing to scaler
            df_cont = df_cont[self.continuous_cols]
                
            if self.clip_outliers:
                for col in base_cols:
                    lower, upper = self.clipping_bounds[col]
                    df_cont[col] = df_cont[col].clip(lower, upper)
                    
            if self.scaler is not None:
                base_cols = [c for c in self.continuous_cols if c not in self.missing_indicators]
                df_cont[base_cols] = self.scaler.transform(df_cont[base_cols])
                scaled_cont = df_cont.values
            else:
                scaled_cont = df_cont.values
                
            processed_blocks.append(scaled_cont)

        # --- Categorical Features ---
        for col in self.categorical_cols:
            col_data = df[col].copy()
            
            if self.impute_missing:
                col_data = col_data.replace(r'^\s*$', '[miss]', regex=True)
                col_data = col_data.fillna(self.imputation_values[col])
            else:
                fallback = self.categories_per_col[col][0]
                col_data = col_data.fillna(fallback)
                
            vals = col_data.values
            num_cats = len(self.categories_per_col[col])
            
            if self.categorical_encoding == "one_hot":
                one_hot = np.zeros((len(df), num_cats))
                for idx, val in enumerate(vals):
                    cat_idx = self.categories_per_col[col].index(val) if val in self.categories_per_col[col] else 0
                    one_hot[idx, cat_idx] = 1.0
                processed_blocks.append(one_hot)
                
            elif self.categorical_encoding == "ordinal":
                codes = np.array([
                    self.categories_per_col[col].index(val) if val in self.categories_per_col[col] else 0
                    for val in vals
                ], dtype=float).reshape(-1, 1)
                processed_blocks.append(codes)
                
        return np.hstack(processed_blocks) if processed_blocks else np.empty((len(df), 0))

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
    
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder 

class ContextEncoder:
    def __init__(self):
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = StandardScaler()
        
        self.cat_cols = []
        self.cyclical_cols = []
        self.continuous_cols = []
        
        self.output_dim = 0
        
        # Hardcode the true mathematical periods of temporal data to prevent collisions
        self.cyclical_periods = {
            'month': 12,
            'hour': 24,
            'minute': 60,
            'second': 60,
            'season': 4
        }
        
        # Tracking variables for automatic time engineering
        self.engineered_time = False
        self.year_col_name = None
        self.month_col_name = None

    def _preprocess_temporal(self, df):
        df = df.copy()
        cols_lower = {c.lower(): c for c in df.columns}
        
        if 'year' in cols_lower and 'month' in cols_lower:
            self.engineered_time = True
            self.year_col_name = cols_lower['year']
            self.month_col_name = cols_lower['month']
            
            # 1. Calculate continuous fractional time
            df['fractional_year'] = df[self.year_col_name] + (df[self.month_col_name] - 1) / 12.0
            
            # 2. Drop the original year column to prevent redundant collinear features
            df = df.drop(columns=[self.year_col_name])
            
        return df

    def fit(self, context_df):
        # 1. Automatically engineer temporal features before routing
        context_df = self._preprocess_temporal(context_df)
        
        # 2. Route columns
        for col in context_df.columns:
            col_lower = col.lower()
            is_cyclical = any(k in col_lower for k in self.cyclical_periods.keys())
            
            if pd.api.types.is_numeric_dtype(context_df[col]):
                if is_cyclical:
                    self.cyclical_cols.append(col)
                else:
                    self.continuous_cols.append(col)
            else:
                self.cat_cols.append(col)

        cat_dim = 0
        if len(self.cat_cols) > 0:
            self.cat_encoder.fit(context_df[self.cat_cols])
            cat_dim = sum([len(cats) for cats in self.cat_encoder.categories_])

        cont_dim = 0
        if len(self.continuous_cols) > 0:
            self.scaler.fit(context_df[self.continuous_cols])
            cont_dim = len(self.continuous_cols)
            
        cyc_dim = len(self.cyclical_cols) * 2 

        self.output_dim = cat_dim + cont_dim + cyc_dim
        
        print(f"Context Encoder Fitted. Output Dimension: {self.output_dim}")
        if self.engineered_time:
            print(f"Auto-engineered 'fractional_year' replacing '{self.year_col_name}'.")

    def transform(self, context_df):
        context_df = self._preprocess_temporal(context_df)
        tensors = []
        if len(self.cat_cols) > 0:
            cat_matrix = self.cat_encoder.transform(context_df[self.cat_cols])
            tensors.append(torch.tensor(cat_matrix, dtype=torch.float32))
        if len(self.continuous_cols) > 0:
            cont_matrix = self.scaler.transform(context_df[self.continuous_cols])
            tensors.append(torch.tensor(cont_matrix, dtype=torch.float32))
            
        if len(self.cyclical_cols) > 0:
            cyc_list = []
            for col in self.cyclical_cols:
                val = context_df[col].values
                
                # Identify the correct period based on the column name
                col_lower = col.lower()
                period = next((p for k, p in self.cyclical_periods.items() if k in col_lower), None)
                
                if period is None:
                    period = context_df[col].max() + 1 
                
                # Map using the true period length
                sin_val = np.sin(2 * np.pi * val / period)
                cos_val = np.cos(2 * np.pi * val / period)
                
                cyc_list.append(sin_val.reshape(-1, 1))
                cyc_list.append(cos_val.reshape(-1, 1))
                
            cyc_matrix = np.hstack(cyc_list)
            tensors.append(torch.tensor(cyc_matrix, dtype=torch.float32))
            
        final_tensor = torch.cat(tensors, dim=1)
        return final_tensor




