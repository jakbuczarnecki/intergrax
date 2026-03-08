# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict
import hashlib


from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey


def build_loader_metadata(
    *,
    source: str,
    parser: str,
    position: int,
) -> Dict[str, object]:
    """
    Build minimal metadata contract for documents produced by loaders.

    Required metadata fields:

    source
    parser
    document_id
    position
    """

    def _safe_document_id(source: str) -> str:
        """
        Generate a stable deterministic document identifier.

        The identifier must remain stable regardless of loader,
        parser, or ingestion pipeline.

        The source may represent:
        - filesystem path
        - URL
        - S3/GCS URI
        - database identifier
        - arbitrary string

        The ID must therefore be derived deterministically from source.
        """

        if not source:
            return "unknown_document"

        normalized = source.replace("\\", "/").rstrip("/")

        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()

        return digest[:16]
    

    doc_id = _safe_document_id(source)

    return {
        DocumentMetadataKey.SOURCE: source,
        DocumentMetadataKey.PARSER: parser,
        DocumentMetadataKey.DOCUMENT_ID: doc_id,
        DocumentMetadataKey.POSITION: position,
    }