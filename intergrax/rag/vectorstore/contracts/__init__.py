# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Public native vector-store contracts."""

from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreContractError,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)

__all__ = [
    "MetadataFilter",
    "VectorStoreContractError",
    "VectorStoreHit",
    "VectorStoreRecord",
    "VectorStoreScope",
]
