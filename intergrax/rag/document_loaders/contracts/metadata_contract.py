# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path
from typing import Dict


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
        Extract a stable document identifier from source.

        Works for:
        - local paths
        - URLs
        - S3/GCS URIs
        - arbitrary strings
        """

        if not source:
            return "unknown"

        # normalize separators
        s = source.replace("\\", "/")

        # remove trailing slash
        if s.endswith("/"):
            s = s[:-1]

        # basename extraction
        if "/" in s:
            return s.rsplit("/", 1)[-1]

        return s
    

    return {
        "source": source,
        "parser": parser,
        "document_id": _safe_document_id(source),
        "position": position,
    }