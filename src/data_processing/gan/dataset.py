"""
PyTorch Dataset for transformed tabular data.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """
    Dataset wrapper for transformed tabular data.

    Parameters
    ----------
    data : np.ndarray
        Transformed tabular data.
    """

    def __init__(self, data: np.ndarray):
        self.data = torch.tensor(
            data,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]