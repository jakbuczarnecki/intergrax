# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray
from intergrax.utils import attribute_access
from openai import OpenAI


def embed_openai_compatible(
    client: OpenAI,
    *,
    model: str,
    texts: Sequence[str],
) -> NDArray[np.float32]:
    """Request and validate an OpenAI-compatible embeddings batch."""
    batch = list(texts)
    response = client.embeddings.create(model=model, input=batch)
    data = attribute_access.optional(response, "data", None)
    if data is None:
        raise ValueError("Embedding response is missing data")

    items = list(data)
    if len(items) != len(batch):
        raise ValueError(
            f"Embedding response count {len(items)} does not match "
            f"request count {len(batch)}"
        )

    ordered: list[NDArray[np.float32] | None] = [None] * len(batch)
    for item in items:
        index = attribute_access.optional(item, "index", None)
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("Embedding response item has an invalid index")
        if index < 0 or index >= len(batch):
            raise ValueError(f"Embedding response index out of range: {index}")
        if ordered[index] is not None:
            raise ValueError(f"Duplicate embedding response index: {index}")

        vector: Any = attribute_access.optional(item, "embedding", None)
        vector_array = np.asarray(vector)
        if isinstance(vector, (str, bytes)) or vector_array.ndim != 1:
            raise ValueError("Embedding must be a one-dimensional numeric sequence")
        if not all(isinstance(value, Real) for value in vector_array.tolist()):
            raise ValueError("Embedding must be a numeric sequence")
        ordered[index] = np.asarray(vector, dtype=np.float32)

    if any(vector is None for vector in ordered):
        raise ValueError("Embedding response is missing an index")

    dimensions = {vector.shape[0] for vector in ordered if vector is not None}
    if len(dimensions) != 1:
        raise ValueError("Embedding response vectors have inconsistent dimensions")

    return np.asarray(ordered, dtype=np.float32)
