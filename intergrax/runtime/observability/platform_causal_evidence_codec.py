# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit platform_causal_evidence.v1 / v2 payload codec (NPSC-5F)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from intergrax.contracts.npsc5f_compatibility import (
    ForbiddenPlatformCausalEvidenceV1WriteError,
    UnknownPlatformCausalEvidenceSchemaError,
)
from intergrax.runtime.observability.causal_evidence import (
    PLATFORM_CAUSAL_EVIDENCE_SCHEMA,
    PlatformCausalEvidence,
)
from intergrax.runtime.observability.causal_evidence_legacy import (
    PLATFORM_CAUSAL_EVIDENCE_SCHEMA_V1,
    LegacyPlatformCausalEvidence,
)


@dataclass(frozen=True, slots=True)
class DecodedPlatformCausalEvidence:
    """Discriminated read result for one stored platform causal evidence payload."""

    kind: Literal["complete_v2", "legacy_incomplete_v1"]
    complete_v2: PlatformCausalEvidence | None = None
    legacy_v1: LegacyPlatformCausalEvidence | None = None

    def __post_init__(self) -> None:
        if self.kind == "complete_v2":
            if self.complete_v2 is None or self.legacy_v1 is not None:
                raise ValueError("complete_v2 decode state is inconsistent")
        elif self.kind == "legacy_incomplete_v1":
            if self.legacy_v1 is None or self.complete_v2 is not None:
                raise ValueError("legacy_incomplete_v1 decode state is inconsistent")
        else:
            raise ValueError("unknown decode kind")


def decode_platform_causal_evidence_payload(
    payload: object,
) -> DecodedPlatformCausalEvidence:
    """Parse one JSON payload dict by explicit schema_version (no silent upgrade)."""
    if not isinstance(payload, dict):
        raise ValueError("platform causal evidence payload must be a mapping")
    schema_version = payload.get("schema_version")
    if schema_version == PLATFORM_CAUSAL_EVIDENCE_SCHEMA:
        try:
            complete = PlatformCausalEvidence.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("invalid platform_causal_evidence.v2 payload") from exc
        return DecodedPlatformCausalEvidence(kind="complete_v2", complete_v2=complete)
    if schema_version == PLATFORM_CAUSAL_EVIDENCE_SCHEMA_V1:
        if "execution_id" in payload.get("target", {}):
            raise ValueError("platform_causal_evidence.v1 must not carry execution_id")
        try:
            legacy = LegacyPlatformCausalEvidence.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("invalid platform_causal_evidence.v1 payload") from exc
        return DecodedPlatformCausalEvidence(
            kind="legacy_incomplete_v1", legacy_v1=legacy
        )
    raise UnknownPlatformCausalEvidenceSchemaError(
        f"unsupported platform causal evidence schema: {schema_version!r}",
    )


def assert_v2_write_payload(payload: object) -> None:
    """Fail closed when a caller attempts to persist v1 platform evidence."""
    if not isinstance(payload, dict):
        return
    schema_version = payload.get("schema_version")
    if schema_version == PLATFORM_CAUSAL_EVIDENCE_SCHEMA_V1:
        raise ForbiddenPlatformCausalEvidenceV1WriteError(
            "platform_causal_evidence.v1 write is retired; use v2 only",
        )


def require_complete_v2(
    decoded: DecodedPlatformCausalEvidence,
) -> PlatformCausalEvidence:
    if decoded.kind != "complete_v2" or decoded.complete_v2 is None:
        raise ValueError("complete platform_causal_evidence.v2 required")
    return decoded.complete_v2
