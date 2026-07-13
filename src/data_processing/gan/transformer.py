"""
Data Transformer for GAN-based synthetic tabular data generation.

This module is responsible for converting raw mixed-type tabular data
into the numerical representation required by the GAN and converting
generated samples back into the original tabular format.

Continuous Columns
------------------
- Bayesian Gaussian Mixture Model (BGMM)
- Alpha-Beta representation

Categorical Columns
-------------------
- One-Hot Encoding

The transformer also stores the metadata required for inverse
transformation.
"""

import numpy as np
import pandas as pd

from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import OneHotEncoder

from src.configs.gan import GANConfig


class DataTransformer:
    """
    Transforms mixed-type tabular data into the representation
    required by the GAN.

    Continuous columns:
        Bayesian Gaussian Mixture Model (BGMM)
        -> alpha + beta representation

    Categorical columns:
        One-Hot Encoding
    """

    def __init__(self, config: GANConfig):
        self.config = config

        self.bgm_models = {}
        self.encoders = {}
        self.metadata = {}

        self.continuous_columns = []
        self.categorical_columns = []
        self.column_order = []

    # -----------------------------------------------------------
    def fit(self, df: pd.DataFrame):
        """
        Fit the transformer on a dataframe.
        """

        self.continuous_columns = []
        self.categorical_columns = []

        self.bgm_models = {}
        self.encoders = {}
        self.metadata = {}

        self.column_order = list(df.columns)

        for column in df.columns:

            if pd.api.types.is_numeric_dtype(df[column]):

                self.continuous_columns.append(column)

                X = df[column].values.reshape(-1, 1).astype(float)

                bgm = BayesianGaussianMixture(
                    n_components=self.config.max_components,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=self.config.weight_concentration_prior,
                    max_iter=self.config.bgmm_max_iter,
                    n_init=1,
                    random_state=self.config.random_state,
                )

                bgm.fit(X)

                self.bgm_models[column] = bgm

                valid_components = np.where(
                    bgm.weights_ > self.config.weight_threshold
                )[0]

                if len(valid_components) == 0:
                    valid_components = np.array([np.argmax(bgm.weights_)])

                self.metadata[column] = {
                    "type": "continuous",
                    "valid_components": valid_components,
                    "num_valid": len(valid_components),
                    "dimension": 1 + len(valid_components),
                }

            else:

                self.categorical_columns.append(column)

                encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore",
                )

                encoder.fit(df[[column]])

                self.encoders[column] = encoder

                self.metadata[column] = {
                    "type": "categorical",
                    "dimension": len(encoder.categories_[0]),
                    "categories": encoder.categories_[0],
                }

        return self

    # -----------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame):
        """
        Fit the transformer and transform the dataframe.
        """
        self.fit(df)
        return self.transform(df)

    # -----------------------------------------------------------
    def transform_continuous(
        self,
        column_name: str,
        column: pd.Series,
    ):

        info = self.metadata[column_name]
        bgm = self.bgm_models[column_name]
        valid_components = info["valid_components"]

        X = column.values.reshape(-1, 1).astype(float)

        probabilities = bgm.predict_proba(X)[:, valid_components]

        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-8

        probabilities = probabilities / row_sums

        modes = np.argmax(probabilities, axis=1)

        means = bgm.means_.flatten()[valid_components]
        stds = np.sqrt(bgm.covariances_).flatten()[valid_components]

        alpha = (X.flatten() - means[modes]) / (4 * stds[modes] + 1e-8)

        alpha = np.clip(alpha, -0.99, 0.99)

        beta = np.zeros((len(column), len(valid_components)))

        beta[np.arange(len(column)), modes] = 1

        return np.concatenate(
            [alpha.reshape(-1, 1), beta],
            axis=1,
        )

    # -----------------------------------------------------------
    def transform_categorical(
        self,
        column_name: str,
        column: pd.Series,
    ):

        encoder = self.encoders[column_name]

        return encoder.transform(column.to_frame())

    # -----------------------------------------------------------
    def transform(
        self,
        df: pd.DataFrame,
    ):

        transformed_columns = []

        for column in self.column_order:

            if column in self.continuous_columns:

                transformed = self.transform_continuous(
                    column,
                    df[column],
                )

            else:

                transformed = self.transform_categorical(
                    column,
                    df[column],
                )

            transformed_columns.append(transformed)

        return np.concatenate(
            transformed_columns,
            axis=1,
        )

    # -----------------------------------------------------------
    def get_activation_layout(self):
        """
        Returns the activation layout used by the Generator.
        """

        layout = []

        idx = 0

        for column in self.column_order:

            info = self.metadata[column]

            dim = info["dimension"]

            layout.append(
                {
                    "type": info["type"],
                    "start": idx,
                    "end": idx + dim,
                }
            )

            idx += dim

        return layout

    def get_inverse_transformer(self):
        """
        Return an InverseTransformer associated with this fitted transformer.
        """
    
        from src.postprocessing.gan.inverse_transformer import (
            InverseTransformer,
        )
    
        return InverseTransformer(self)