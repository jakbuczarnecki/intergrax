# © Artur Czarnecki. All rights reserved.

"""
Removed: use ``intergrax.rag.retrieval.RetrievalService``.

The former ``intergrax.rag.answers`` stack lives under ``intergrax.legacy.rag_answers``
for notebook compatibility only. Do not import from runtime, agents, or applications.
"""

from __future__ import annotations

raise ImportError(
    "intergrax.rag.answers was removed (Phase U-Leg.2). "
    "Use intergrax.rag.retrieval.RetrievalService instead. "
    "Notebooks may import intergrax.legacy.rag_answers explicitly."
)
