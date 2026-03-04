# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from intergrax.rag.document_loaders.contracts.metadata_provider import BaseMetadataProvider


class DefaultMetadataProvider(BaseMetadataProvider):
    """
    Default metadata enrichment provider.

    Replicates the behavior previously implemented inside DocumentsLoader.
    """

    @staticmethod
    def _stable_parent_id(path: Path) -> str:
        return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]

    def enrich(
        self,
        documents: Sequence[Document],
        source: Path | str,
    ) -> Sequence[Document]:

        p = Path(source).resolve()
        ext = p.suffix.lower()
        parent_id = self._stable_parent_id(p)

        for d in documents:

            md = d.metadata or {}

            md.setdefault("source_path", str(p))
            md.setdefault("source_name", p.name)
            md.setdefault("ext", ext)

            if "page" in md and "page_index" not in md:
                md["page_index"] = md["page"]

            md.setdefault("parent_id", parent_id)

            d.metadata = md

        return documents