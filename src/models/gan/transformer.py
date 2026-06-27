import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder

class DataTransformer:

    def __init__(self,
                 n_components=5,
                 random_state=42):

        self.n_components = n_components
        self.random_state = random_state

        # Stores GMMs
        self.gmm_models = {}

        # Stores OneHotEncoders
        self.encoders = {}

        # Metadata for each column
        self.metadata = {}

        self.continuous_columns = []
        self.categorical_columns = []

    ####################################################################
    # FIT
    ####################################################################

    def fit(self, df):

        self.continuous_columns = []
        self.categorical_columns = []

        self.gmm_models = {}
        self.encoders = {}
        self.metadata = {}

        for column in df.columns:

            # -----------------------------
            # Continuous Column
            # -----------------------------
            if pd.api.types.is_numeric_dtype(df[column]):

                self.continuous_columns.append(column)

                X = df[column].values.reshape(-1, 1)

                gmm = GaussianMixture(
                    n_components=self.n_components,
                    random_state=self.random_state
                )

                gmm.fit(X)

                self.gmm_models[column] = gmm

                self.metadata[column] = {
                    "type": "continuous",
                    "dimension": 1 + self.n_components
                }

            # -----------------------------
            # Categorical Column
            # -----------------------------
            else:

                self.categorical_columns.append(column)

                encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore"
                )

                encoder.fit(df[[column]])

                self.encoders[column] = encoder

                self.metadata[column] = {
                    "type": "categorical",
                    "dimension": len(encoder.categories_[0]),
                    "categories": encoder.categories_[0]
                }
    ####################################################################
    # Transform Continuous Column
    ####################################################################

    def transform_continuous(self,
                             column_name,
                             column):

        gmm = self.gmm_models[column_name]

        X = column.values.reshape(-1, 1)

        probabilities = gmm.predict_proba(X)

        modes = np.argmax(probabilities, axis=1)

        means = gmm.means_.flatten()

        stds = np.sqrt(gmm.covariances_).flatten()

        alpha = (X.flatten() - means[modes]) / (4 * stds[modes])

        alpha = np.clip(alpha,
                        -0.99,
                        0.99)

        beta = np.zeros(
            (
                len(column),
                self.n_components
            )
        )

        beta[
            np.arange(len(column)),
            modes
        ] = 1

        transformed = np.concatenate(
            [
                alpha.reshape(-1, 1),
                beta
            ],
            axis=1
        )

        return transformed

    ####################################################################
    # Transform Categorical Column
    ####################################################################

    def transform_categorical(self,
                              column_name,
                              column):

        encoder = self.encoders[column_name]

        transformed = encoder.transform(
            column.to_frame()
        )

        return transformed

    ####################################################################
    # Transform Whole Dataset
    ####################################################################

    def transform(self, df):

        transformed_columns = []

        for column in df.columns:

            if column in self.continuous_columns:

                transformed = self.transform_continuous(
                    column,
                    df[column]
                )

            else:

                transformed = self.transform_categorical(
                    column,
                    df[column]
                )

            transformed_columns.append(transformed)

        transformed_data = np.concatenate(
            transformed_columns,
            axis=1
        )

        return transformed_data        
    

    ####################################################################
    # INVERSE TRANSFORM
    ####################################################################

    def inverse_transform_continuous(
        self,
        column_name,
        transformed_column
    ):

        gmm = self.gmm_models[column_name]

        alpha = transformed_column[:, 0]

        beta = transformed_column[:, 1:]

        modes = np.argmax(beta, axis=1)

        means = gmm.means_.flatten()

        stds = np.sqrt(gmm.covariances_).flatten()

        values = alpha * 4 * stds[modes] + means[modes]

        return values

    def inverse_transform_categorical(
        self,
        column_name,
        transformed_column
    ):

        encoder = self.encoders[column_name]

        indices = np.argmax(transformed_column, axis=1)

        categories = encoder.categories_[0]

        values = categories[indices]

        return values

    def inverse_transform_dataset(
        self,
        transformed_data
    ):

        reconstructed = {}

        current_index = 0

        for column in self.metadata:

            info = self.metadata[column]

            dim = info["dimension"]

            column_data = transformed_data[
                :,
                current_index:current_index + dim
            ]

            if info["type"] == "continuous":

                values = self.inverse_transform_continuous(
                    column,
                    column_data
                )

            else:

                values = self.inverse_transform_categorical(
                    column,
                    column_data
                )

            reconstructed[column] = values

            current_index += dim

        return pd.DataFrame(reconstructed)