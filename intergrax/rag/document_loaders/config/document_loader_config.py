# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os


class DoclingMode(str, Enum):

    NONE = "none"
    LOCAL = "local"
    SERVER = "server"


def _read_docling_mode() -> DoclingMode:

    raw = os.getenv("INTERGRAX_DOCLING_MODE", "local").lower()

    try:
        return DoclingMode(raw)
    except ValueError:
        raise RuntimeError(
            f"Invalid INTERGRAX_DOCLING_MODE='{raw}'. "
            "Allowed values: none, local, server."
        )


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
    docling_mode: str =  _read_docling_mode()

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