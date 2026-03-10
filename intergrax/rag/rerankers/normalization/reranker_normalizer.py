# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List
import numpy as np


class RerankerNormalizer:
    """
    Score normalization utilities for reranker providers.
    """

    @staticmethod
    def minmax(values: List[float]) -> List[float]:

        if not values:
            return []

        arr = np.asarray(values, dtype=np.float32)

        mn = float(arr.min())
        mx = float(arr.max())

        if abs(mx - mn) < 1e-12:
            return [0.5] * len(values)

        norm = (arr - mn) / (mx - mn)

        return norm.tolist()

    @staticmethod
    def zscore(values: List[float]) -> List[float]:

        if not values:
            return []

        arr = np.asarray(values, dtype=np.float32)

        mean = float(arr.mean())
        std = float(arr.std())

        if std < 1e-12:
            return [0.0] * len(values)

        z = (arr - mean) / std

        # rescale to 0-1
        zmin = float(z.min())
        zmax = float(z.max())

        if abs(zmax - zmin) < 1e-12:
            return [0.5] * len(values)

        norm = (z - zmin) / (zmax - zmin)

        return norm.tolist()

    @staticmethod
    def softmax(values: List[float]) -> List[float]:

        if not values:
            return []

        arr = np.asarray(values, dtype=np.float32)

        e = np.exp(arr - arr.max())

        s = e / e.sum()

        return s.tolist()