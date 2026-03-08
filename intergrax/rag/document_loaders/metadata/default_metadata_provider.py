# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey
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

        source_str = str(source)

        try:
            p = Path(source_str)

            if p.exists():
                p = p.resolve()

                ext = p.suffix.lower()
                parent_id = self._stable_parent_id(p)

                source_path = str(p)
                source_name = p.name

            else:
                raise ValueError

        except Exception:

            ext = ""
            parent_id = hashlib.sha1(source_str.encode("utf-8")).hexdigest()[:16]

            source_path = source_str
            source_name = source_str.split("/")[-1]

        for d in documents:

            md = dict(d.metadata or {})

            md.setdefault(DocumentMetadataKey.SOURCE_PATH, source_path)
            md.setdefault(DocumentMetadataKey.SOURCE_NAME, source_name)
            md.setdefault(DocumentMetadataKey.EXTENSION, ext)

            if "page" in md and DocumentMetadataKey.PAGE_INDEX not in md:
                md[DocumentMetadataKey.PAGE_INDEX] = md.pop("page")

            md.setdefault(DocumentMetadataKey.PARENT_ID, parent_id)

            d.metadata = md

        return documents