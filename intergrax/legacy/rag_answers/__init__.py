# © Artur Czarnecki. All rights reserved.

"""
Legacy RAG answer stack (archived Phase U-Leg.2).

Not for production runtime, agents, or applications. Use
``intergrax.rag.retrieval.RetrievalService`` for all harness and agent paths.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "intergrax.legacy.rag_answers is archived; use intergrax.rag.retrieval.RetrievalService",
    DeprecationWarning,
    stacklevel=2,
)
