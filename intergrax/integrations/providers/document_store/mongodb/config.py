# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MongoDB document store integration configuration (Phase M.6 P2)."""

from __future__ import annotations

import os
import re

from intergrax.integrations._shared.config import BaseIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError

ENV_MONGODB_URI = "INTERGRAX_MONGODB_URI"
ENV_MONGODB_DATABASE = "INTERGRAX_MONGODB_DATABASE"
ENV_MONGODB_COLLECTION = "INTERGRAX_MONGODB_COLLECTION"

DEFAULT_DATABASE = "intergrax"
DEFAULT_COLLECTION = "intergrax_documents"

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(value: str, field: str) -> str:
    if not value or not _IDENTIFIER_PATTERN.match(value):
        raise IntegrationConfigurationError(
            f"Invalid MongoDB {field}: {value!r} (letters, digits, underscore; must start with letter/_)"
        )
    return value


class MongoDBIntegrationConfig(BaseIntegrationConfig):
    """
    MongoDB settings for partition-scoped ``DocumentStore``.

    Documents are stored with ``partition_key``, ``row_key``, and ``data`` fields.
    Recommended index: ``{partition_key: 1, row_key: 1}`` unique.
    """

    uri: str = ""
    database: str = DEFAULT_DATABASE
    collection_name: str = DEFAULT_COLLECTION

    def qualified_collection(self) -> tuple[str, str]:
        database = _validate_identifier(self.database, "database")
        collection = _validate_identifier(self.collection_name, "collection_name")
        return database, collection

    @classmethod
    def from_env(cls, **overrides: object) -> MongoDBIntegrationConfig:
        uri = os.environ.get(ENV_MONGODB_URI, "").strip()
        database = os.environ.get(ENV_MONGODB_DATABASE, DEFAULT_DATABASE).strip() or DEFAULT_DATABASE
        collection_name = (
            os.environ.get(ENV_MONGODB_COLLECTION, DEFAULT_COLLECTION).strip() or DEFAULT_COLLECTION
        )
        payload: dict[str, object] = {
            "uri": uri,
            "database": database,
            "collection_name": collection_name,
        }
        payload.update(overrides)
        return cls.model_validate(payload)
