import numpy as np
import pandas as pd
from src.data_processing.data_preprocessing import TabularDataPreprocessor
class TabularDataPostprocessor:
    def __init__(self, preprocessor: TabularDataPreprocessor):
        self.preprocessor= preprocessor

    def inverse_transform(self, processed_matrix: np.ndarray) -> pd.DataFrame:
        reconstructed_data={}
        continuous_dim= len(self.preprocessor.continuous_cols)
        if self.preprocessor.continuous_cols:
            scaled_cont= processed_matrix[:, :continuous_dim]

            if self.preprocessor.scaler is not None:
                orig_cont= self.preprocessor.scaler.inverse_transform(scaled_cont)
            else:
                orig_cont= scaled_cont

            for idx,col in enumerate(self.preprocessor.continuous_cols):
                reconstructed_data[col]= orig_cont[:, idx]

        current_idx= continuous_dim

        for col in self.preprocessor.categorical_cols:
            categories=self.preprocessor.categories_per_col[col]
            num_cats= len(categories)

            if self.preprocessor.categorical_encoding == "one_hot":
                one_hot_block = processed_matrix[:, current_idx : current_idx+num_cats]
                cat_indices = np.argmax(one_hot_block, axis=1)
                reconstructed_data[col]=[categories[i] for i in cat_indices]
                current_idx+=num_cats
            '''elif for ordinal encodings placeholder'''
        df_reconstructed = pd.DataFrame(reconstructed_data)
        final_column_order=self.preprocessor.continuous_cols + self.preprocessor.categorical_cols
        return df_reconstructed[final_column_order]      
