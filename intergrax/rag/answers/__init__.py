# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Deprecated: use ``intergrax.rag.retrieval.RetrievalService`` instead.

This package remains for backward-compatible tests only; do not import from
``runtime/`` or ``agents/`` production code (Phase Q-R.9).
"""

from __future__ import annotations

import warnings

warnings.warn(
    "intergrax.rag.answers is deprecated; use intergrax.rag.retrieval.RetrievalService",
    DeprecationWarning,
    stacklevel=2,
)
