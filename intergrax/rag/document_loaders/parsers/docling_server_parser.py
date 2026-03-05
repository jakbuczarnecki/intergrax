# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence
import httpx

from langchain_core.documents import Document

from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.contracts.metadata_contract import build_loader_metadata


class DoclingServerParser(BaseDocumentParser):

    @classmethod
    def parser_id(cls) -> str:
        return "docling.server"

    def is_available(self) -> bool:
        cfg = GLOBAL_DOCUMENT_LOADER_CONFIG
        return cfg.docling_mode == "server"

    def load(self, source: str) -> Sequence[Document]:

        cfg = GLOBAL_DOCUMENT_LOADER_CONFIG

        url = cfg.docling_server_url.rstrip("/") + "/convert"

        timeout = cfg.docling_server_timeout_seconds

        with open(source, "rb") as f:

            files = {"file": (source, f)}

            response = httpx.post(
                url,
                files=files,
                timeout=timeout
            )

        response.raise_for_status()

        payload = response.json()

        text = payload.get("markdown") or payload.get("text") or ""

        metadata = build_loader_metadata(
            source=source,
            parser=self.parser_id(),
            position=0,
        )

        return [
            Document(
                page_content=text,
                metadata=metadata,
            )
        ]