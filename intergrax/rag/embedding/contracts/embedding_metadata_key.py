# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from enum import Enum


class EmbeddingMetadataKey(str, Enum):
    """
    Strongly typed metadata keys for embedding pipeline.
    """

    VECTOR = "embedding"

    PROVIDER = "embedding_provider"

    MODEL = "embedding_model"

    DIMENSION = "embedding_dimension"