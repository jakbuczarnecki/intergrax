# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""LangChain compatibility adapters."""

from intergrax.compat.langchain.documents import (
    LangChainCompatibilityUnavailableError,
    LangChainDocumentBridgeError,
    from_langchain_document,
    to_langchain_document,
)

__all__ = [
    "LangChainCompatibilityUnavailableError",
    "LangChainDocumentBridgeError",
    "from_langchain_document",
    "to_langchain_document",
]
