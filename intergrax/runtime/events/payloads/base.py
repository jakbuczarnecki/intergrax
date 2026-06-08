# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed runtime event payload base (OBS-BUS-1, canon §42.23.1)."""

from __future__ import annotations

from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict


class RuntimeEventPayload(BaseModel):
    """
    Canonical typed body for ``RuntimeEvent.payload``.

    Storage envelope (dict on the bus)::

        {
            "payload_schema_id": "<schema_id>",
            "payload_schema_version": 1,
            "data": { ... model fields ... }
        }

    Legacy emitters may still use unstructured dicts until OBS-BUS-3 migration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: ClassVar[str]
    schema_version: ClassVar[int] = 1

    def to_envelope(self) -> dict[str, Any]:
        return {
            "payload_schema_id": self.__class__.schema_id,
            "payload_schema_version": self.__class__.schema_version,
            "data": self.model_dump(mode="json"),
        }

    @classmethod
    def from_envelope(cls, envelope: dict[str, Any]) -> Self:
        schema_id = envelope.get("payload_schema_id")
        if schema_id != cls.schema_id:
            raise ValueError(
                f"envelope schema_id mismatch: expected {cls.schema_id!r}, got {schema_id!r}"
            )
        data = envelope.get("data")
        if not isinstance(data, dict):
            raise ValueError("envelope missing typed data dict")
        return cls.model_validate(data)
