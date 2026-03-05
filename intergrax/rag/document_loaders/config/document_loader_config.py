# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class DocumentLoaderConfig:
    """
    Global configuration for the document loader subsystem.
    """

    # Confidence score for builtin handlers
    default_builtin_handler_confidence: float = 0.8

    # Docling integration mode:
    #   none   -> disabled
    #   local  -> use local docling library
    #   server -> use docling server (docker)
    docling_mode: str = os.getenv(
        "INTERGRAX_DOCLING_MODE",
        "local"
    )

    # URL for Docling server
    docling_server_url: str = os.getenv(
        "INTERGRAX_DOCLING_SERVER_URL",
        "http://localhost:8000"
    )

    # request timeout
    docling_server_timeout_seconds: int = int(
        os.getenv("INTERGRAX_DOCLING_TIMEOUT", "120")
    )


GLOBAL_DOCUMENT_LOADER_CONFIG = DocumentLoaderConfig()