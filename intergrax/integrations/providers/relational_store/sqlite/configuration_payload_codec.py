# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable codec for SQLite relational-store configuration payloads (TRACE-X-P5-R2-P2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from intergrax.integrations.contracts.existing_capability_configuration import (
    IntegrationConfigurationPayload,
)
from intergrax.integrations.contracts.integration_configuration_payload_codec import (
    IntegrationConfigurationPayloadCodec,
)
from intergrax.integrations.providers.relational_store.sqlite.configuration_realization import (
    SQLITE_RELATIONAL_STORE_CONFIGURATION_TYPE,
    SQLiteRelationalStoreConfigurationPayload,
)
from intergrax.knowledge.contracts.validation import JsonValue


class SQLiteRelationalStoreConfigurationPayloadCodec:
    """Provider-owned JSON codec — no reflection or pickle."""

    def configuration_type(self) -> str:
        return SQLITE_RELATIONAL_STORE_CONFIGURATION_TYPE

    def encode(self, payload: IntegrationConfigurationPayload) -> JsonValue:
        if not isinstance(payload, SQLiteRelationalStoreConfigurationPayload):
            raise TypeError("expected SQLiteRelationalStoreConfigurationPayload")
        material: dict[str, Any] = {
            "data_dir": payload.data_dir.as_posix(),
            "relational_db": (
                payload.relational_db.as_posix() if payload.relational_db is not None else None
            ),
        }
        return cast(JsonValue, material)

    def decode(self, payload: JsonValue) -> IntegrationConfigurationPayload:
        if not isinstance(payload, dict):
            raise ValueError("invalid sqlite configuration payload record")
        data_dir_raw = payload.get("data_dir")
        if type(data_dir_raw) is not str or not data_dir_raw:
            raise ValueError("invalid sqlite configuration data_dir")
        relational_raw = payload.get("relational_db")
        relational_db: Path | None
        if relational_raw is None:
            relational_db = None
        elif type(relational_raw) is str and relational_raw:
            relational_db = Path(relational_raw)
        else:
            raise ValueError("invalid sqlite configuration relational_db")
        return SQLiteRelationalStoreConfigurationPayload(
            data_dir=Path(data_dir_raw),
            relational_db=relational_db,
        )


def sqlite_relational_store_configuration_payload_codec() -> IntegrationConfigurationPayloadCodec:
    return cast(IntegrationConfigurationPayloadCodec, SQLiteRelationalStoreConfigurationPayloadCodec())


__all__ = [
    "SQLiteRelationalStoreConfigurationPayloadCodec",
    "sqlite_relational_store_configuration_payload_codec",
]
