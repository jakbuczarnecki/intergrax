# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vector index metric identifiers for integration administration contracts."""

from __future__ import annotations

from typing import Literal

Metric = Literal[
    "cosine",
    "dot",
    "euclidean",
]

__all__ = ["Metric"]
