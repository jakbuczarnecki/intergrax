# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, List

from langchain_core.documents import Document

from intergrax.rag.contracts.metadata_provider import BaseMetadataProvider


class MetadataPipeline:
    """
    Deterministic pipeline executing metadata providers in sequence.
    """

    def __init__(
        self,
        providers: Iterable[BaseMetadataProvider],
    ) -> None:
        self._providers: List[BaseMetadataProvider] = list(providers)

    def enrich(
        self,
        documents: Sequence[Document],
        source: Path | str,
    ) -> Sequence[Document]:

        docs: Sequence[Document] = documents

        for provider in self._providers:
            docs = provider.enrich(docs, source)

        return docs