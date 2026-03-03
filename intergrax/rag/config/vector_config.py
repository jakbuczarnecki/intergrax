# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Literal


VectorProvider = Literal[
    "chroma",
    "qdrant",
    "pinecone",
]


Metric = Literal[
    "cosine",
    "dot",
    "euclidean",
]