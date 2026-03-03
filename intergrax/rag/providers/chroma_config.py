# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from chromadb.config import Settings as ChromaSettings


@dataclass(frozen=True)
class ChromaConfig:
    """
    Configuration model for Chroma vector store provider.

    The provider is responsible for creating and managing
    the Chroma client instance based on this configuration.
    """

    collection_name: str
    persist_directory: Optional[str] = None
    settings: Optional[ChromaSettings] = None