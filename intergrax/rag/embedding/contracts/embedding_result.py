# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document

@dataclass
class EmbeddingResult:
    documents: Sequence[Document]
    embeddings: Sequence[Sequence[float]]