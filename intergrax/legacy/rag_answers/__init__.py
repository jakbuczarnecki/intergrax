# © Artur Czarnecki. All rights reserved.

"""
Legacy RAG answer stack (archived Phase U-Leg.2).

Not for production runtime, agents, or applications. Use
``intergrax.rag.retrieval.RetrievalService`` for all harness and agent paths.

**Removal timeline:** scheduled for removal after **2026-12-31** (M-RAG.68).
Migrate notebooks to ``RetrievalService`` + ``format_rag_context_text``.
"""

from __future__ import annotations

import warnings

_REMOVAL_DATE = "2026-12-31"

warnings.warn(
    f"intergrax.legacy.rag_answers is archived and will be removed after {_REMOVAL_DATE}; "
    "use intergrax.rag.retrieval.RetrievalService",
    DeprecationWarning,
    stacklevel=2,
)
