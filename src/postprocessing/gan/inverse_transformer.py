"""
Inverse Transformer for GAN generated tabular data.

Converts the transformed GAN output back into the original tabular
representation using the fitted DataTransformer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class InverseTransformer:
    """
    Convert transformed GAN output back to the original tabular format.

    Parameters
    ----------
    transformer
        A fitted DataTransformer instance.
    """

    def __init__(self, transformer):

        self.transformer = transformer

    @classmethod
    def from_transformer(cls, transformer):

        return cls(transformer)

    # -----------------------------------------------------------
    def inverse_transform_continuous(
        self,
        column_name: str,
        transformed_column: np.ndarray,
    ) -> np.ndarray:

        bgm = self.transformer.bgm_models[column_name]

        info = self.transformer.metadata[column_name]

        valid_components = info["valid_components"]

        alpha = transformed_column[:, 0]

        beta = transformed_column[:, 1:]

        component_index = np.argmax(beta, axis=1)

        selected_components = valid_components[component_index]

        means = bgm.means_.flatten()

        stds = np.sqrt(bgm.covariances_).flatten()

        recovered = (
            alpha * 4 * stds[selected_components]
            + means[selected_components]
        )

        return recovered

    # -----------------------------------------------------------
    def inverse_transform_categorical(
        self,
        column_name: str,
        transformed_column: np.ndarray,
    ) -> np.ndarray:

        encoder = self.transformer.encoders[column_name]

        recovered = encoder.inverse_transform(
            transformed_column
        )

        return recovered.flatten()

    # -----------------------------------------------------------
    def inverse_transform(
        self,
        transformed_data: np.ndarray,
    ) -> pd.DataFrame:

        recovered = {}

        start = 0

        for column in self.transformer.column_order:

            info = self.transformer.metadata[column]

            dim = info["dimension"]

            current = transformed_data[:, start:start + dim]

            if info["type"] == "continuous":

                recovered[column] = self.inverse_transform_continuous(
                    column,
                    current,
                )

            else:

                recovered[column] = self.inverse_transform_categorical(
                    column,
                    current,
                )

            start += dim

        return pd.DataFrame(recovered)