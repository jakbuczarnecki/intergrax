# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PostgreSQL document store integration provider."""

from intergrax.integrations.providers.document_store.postgresql.bundle import (
    create_postgresql_document_store,
)

__all__ = ["create_postgresql_document_store"]
