"""
Conditional Sampler for CTGAN-style conditional training.

This module implements conditional vector sampling and
training-by-sampling for categorical columns.

During training:
    - Conditions are sampled using log-frequency probabilities.

During generation:
    - Conditions are sampled using the true data distribution.
"""

from __future__ import annotations

import numpy as np

from src.preprocessing.gan.transformer import DataTransformer


class ConditionalSampler:
    """
    CTGAN-style conditional sampler.

    Parameters
    ----------
    transformed_data : np.ndarray
        Transformed training data.

    transformer : DataTransformer
        Fitted data transformer.
    """

    def __init__(
        self,
        transformed_data: np.ndarray,
        transformer: DataTransformer,
    ):

        self.transformer = transformer

        cat_names = [
            column
            for column in transformer.column_order
            if column in transformer.categorical_columns
        ]

        cat_segments = [
            segment
            for segment in transformer.get_activation_layout()
            if segment["type"] == "categorical"
        ]

        self.discrete_columns = []

        cond_start = 0

        for name, segment in zip(cat_names, cat_segments):

            dim = segment["end"] - segment["start"]

            self.discrete_columns.append(
                {
                    "name": name,
                    "data_start": segment["start"],
                    "data_end": segment["end"],
                    "cond_start": cond_start,
                    "cond_end": cond_start + dim,
                    "dim": dim,
                }
            )

            cond_start += dim

        self.n_discrete_columns = len(self.discrete_columns)
        self.cond_dim = cond_start

        # Probability distributions
        self.log_freq = []
        self.true_freq = []

        # Training-by-sampling indices
        self.row_idx_by_cat = []

        for info in self.discrete_columns:

            one_hot = transformed_data[
                :,
                info["data_start"] : info["data_end"],
            ]

            counts = one_hot.sum(axis=0)

            # Log-frequency distribution (training)
            log_prob = np.log(counts + 1)
            log_prob /= log_prob.sum()

            self.log_freq.append(log_prob)

            # True distribution (generation)
            true_prob = counts / counts.sum()

            self.true_freq.append(true_prob)

            category_index = np.argmax(one_hot, axis=1)

            self.row_idx_by_cat.append(
                {
                    category: np.where(category_index == category)[0]
                    for category in range(info["dim"])
                }
            )

    # -------------------------------------------------------------
    # Training Conditional Vector
    # -------------------------------------------------------------

    def sample_train_condvec(
        self,
        batch_size: int,
    ):
        """
        Sample conditional vectors using log-frequency probabilities.
        """

        if self.n_discrete_columns == 0:
            return None, None, None

        cond = np.zeros(
            (batch_size, self.cond_dim),
            dtype=np.float32,
        )

        column_choice = np.random.randint(
            0,
            self.n_discrete_columns,
            size=batch_size,
        )

        category_choice = np.empty(
            batch_size,
            dtype=np.int64,
        )

        for i in range(batch_size):

            column = column_choice[i]

            probabilities = self.log_freq[column]

            category = np.random.choice(
                len(probabilities),
                p=probabilities,
            )

            category_choice[i] = category

            cond[
                i,
                self.discrete_columns[column]["cond_start"] + category,
            ] = 1.0

        return cond, column_choice, category_choice

    # -------------------------------------------------------------
    # Training-by-Sampling
    # -------------------------------------------------------------

    def sample_real_rows(
        self,
        column_choice,
        category_choice,
        transformed_data,
    ):
        """
        Sample real rows matching the selected conditions.
        """

        indices = np.empty(
            len(column_choice),
            dtype=np.int64,
        )

        for i, (column, category) in enumerate(
            zip(column_choice, category_choice)
        ):

            candidates = self.row_idx_by_cat[column][category]

            if len(candidates) > 0:
                indices[i] = np.random.choice(candidates)
            else:
                indices[i] = np.random.randint(
                    transformed_data.shape[0]
                )

        return transformed_data[indices]

    # -------------------------------------------------------------
    # Generation Conditional Vector
    # -------------------------------------------------------------

    def sample_generation_condvec(
        self,
        batch_size: int,
    ):
        """
        Sample conditional vectors using the true data distribution.
        """

        if self.n_discrete_columns == 0:
            return None

        cond = np.zeros(
            (batch_size, self.cond_dim),
            dtype=np.float32,
        )

        column_choice = np.random.randint(
            0,
            self.n_discrete_columns,
            size=batch_size,
        )

        for i in range(batch_size):

            column = column_choice[i]

            probabilities = self.true_freq[column]

            category = np.random.choice(
                len(probabilities),
                p=probabilities,
            )

            cond[
                i,
                self.discrete_columns[column]["cond_start"] + category,
            ] = 1.0

        return cond