import numpy as np
import pandas as pd
from src.data_processing.data_preprocessing import TabularDataPreprocessor

class TabularDataPostprocessor:
    def __init__(self, preprocessor: TabularDataPreprocessor):
        self.preprocessor = preprocessor

    def restore_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverts generated missing indicators and '[miss]' tags back to natural empty states,
        ensuring the final synthetic dataset matches the original schema perfectly.
        """
        df = df.copy()
        
        # 1. Restore continuous NaNs using the predicted indicators
        if hasattr(self.preprocessor, 'missing_indicators'):
            for indicator_col in self.preprocessor.missing_indicators:
                if indicator_col in df.columns:
                    orig_col = indicator_col.replace('_is_missing', '')
                    
                    if orig_col in df.columns:
                        # If the generative model predicted this should be missing (threshold > 0.5)
                        # Overwrite the continuous value with a true NaN
                        df.loc[df[indicator_col] >= 0.5, orig_col] = np.nan
                        
        # 2. Hard-delete ALL impute missing indicator columns via string matching
        cols_to_drop = [c for c in df.columns if str(c).endswith('_is_missing')]
        df = df.drop(columns=cols_to_drop, errors='ignore')
                    
        # 3. Restore categorical empty cells
        for col in self.preprocessor.categorical_cols:
            if col in df.columns:
                # Replace the hardcoded string with true pandas missing values
                df[col] = df[col].replace('[miss]', np.nan)
                
        return df

    def inverse_transform(self, processed_matrix: np.ndarray) -> pd.DataFrame:
        reconstructed_data = {}
        continuous_dim = len(self.preprocessor.continuous_cols)
        
        if self.preprocessor.continuous_cols:
            scaled_cont = processed_matrix[:, :continuous_dim]

            if self.preprocessor.scaler is not None:
                # ---> FIX: Inverse transform only the base columns <---
                base_cols = [c for c in self.preprocessor.continuous_cols if c not in self.preprocessor.missing_indicators]
                base_indices = [self.preprocessor.continuous_cols.index(c) for c in base_cols]
                
                orig_cont = scaled_cont.copy()
                orig_cont[:, base_indices] = self.preprocessor.scaler.inverse_transform(scaled_cont[:, base_indices])
            else:
                orig_cont = scaled_cont

            for idx, col in enumerate(self.preprocessor.continuous_cols):
                reconstructed_data[col] = orig_cont[:, idx]

        current_idx = continuous_dim

        for col in self.preprocessor.categorical_cols:
            categories = self.preprocessor.categories_per_col[col]
            num_cats = len(categories)

            if self.preprocessor.categorical_encoding == "one_hot":
                one_hot_block = processed_matrix[:, current_idx : current_idx+num_cats]
                cat_indices = np.argmax(one_hot_block, axis=1)
                reconstructed_data[col] = [categories[i] for i in cat_indices]
                current_idx += num_cats
            # elif for ordinal encodings placeholder
            
        df_reconstructed = pd.DataFrame(reconstructed_data)
        
        # Order the columns based on the neural network's output schema
        final_column_order = self.preprocessor.continuous_cols + self.preprocessor.categorical_cols
        df_reconstructed = df_reconstructed[final_column_order]
        
        # Scrub the missing indicators and tags before handing the dataframe back
        return self.restore_missing_values(df_reconstructed)
