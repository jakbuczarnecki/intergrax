# © Artur Czarnecki. All rights reserved.

"""Host attestation contract for execution-evidence payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_HOST_ATTESTATION_V1: Final = "host_attestation.v1"
_NON_EMPTY = Field(min_length=1)


class HostAttestation(BaseModel):
    """Host-owned cryptographic attestation over canonical payload bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["host_attestation.v1"] = SCHEMA_HOST_ATTESTATION_V1
    attestation_id: str = _NON_EMPTY
    algorithm: str = _NON_EMPTY
    key_id: str = _NON_EMPTY
    payload_digest: str = _NON_EMPTY
    signature: str = _NON_EMPTY
    signed_at: datetime
    payload_schema: str = _NON_EMPTY

    @field_validator(
        "attestation_id",
        "algorithm",
        "key_id",
        "payload_digest",
        "signature",
        "payload_schema",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


@runtime_checkable
class HostAttestor(Protocol):
    """Injectable host signing boundary — Tier-2 must not import implementations."""

    def attest(
        self,
        payload: bytes,
        *,
        schema: str,
    ) -> HostAttestation:
        """Sign canonical payload bytes; return an explicit attestation record."""
        ...
