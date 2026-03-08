# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


class EmbeddingProvider(ABC):
    """
    Tier-0 contract for embedding providers.

    Implementations must convert a batch of texts into embedding vectors.
    """

    @abstractmethod
    def provider_name(self) -> str:
        """
        Returns provider identifier (e.g. 'hf', 'openai', 'ollama').
        """
        raise NotImplementedError

    @abstractmethod
    def dimension(self) -> int:
        """
        Returns embedding vector dimension.
        """
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """
        Embed a batch of texts.

        Returns:
            numpy array with shape (N, dim)
        """
        raise NotImplementedError