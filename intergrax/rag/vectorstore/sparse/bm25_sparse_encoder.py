# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Hash-based sparse vectors for Qdrant native sparse indexes."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from intergrax.rag.vectorstore.sparse.lexical_index import tokenize

_SPARSE_DIM = 2**18


@dataclass(frozen=True)
class SparseVector:
    indices: List[int]
    values: List[float]


def encode_sparse_bm25(text: str, *, dim: int = _SPARSE_DIM) -> SparseVector:
    """
    Map tokens to stable bucket indices (Murmur-style hash) with sqrt-TF weights.

    Suitable for Qdrant ``SparseVector`` upsert/query — no external SPLADE model required.
    """
    tokens = tokenize(text)
    if not tokens:
        return SparseVector(indices=[], values=[])

    counts = Counter(tokens)
    bucket: Dict[int, float] = {}
    for term, tf in counts.items():
        idx = hash(term) % dim
        weight = 1.0 + math.log(1.0 + tf)
        bucket[idx] = max(bucket.get(idx, 0.0), weight)

    indices = sorted(bucket.keys())
    values = [float(bucket[i]) for i in indices]
    return SparseVector(indices=indices, values=values)
