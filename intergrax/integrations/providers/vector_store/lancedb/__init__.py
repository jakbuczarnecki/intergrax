# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.vector_store.lancedb.bundle import create_lancedb_vector_store
from intergrax.integrations.providers.vector_store.lancedb.register import register_lancedb_integration

__all__ = ["create_lancedb_vector_store", "register_lancedb_integration"]
