# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral read-only effective profile execution provenance (TRACE-X-P5-R1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.effective_profile_revision_provenance_ref import (
    EffectiveProfileRevisionProvenanceRef,
)
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id


@dataclass(frozen=True, slots=True)
class ExecutionEffectiveProfileProvenance:
    """Immutable execution→revision provenance projection — not permission or activation."""

    tenant_id: str
    execution_id: ExecutionId
    revision_ref: EffectiveProfileRevisionProvenanceRef
    fingerprint: str


class ExecutionEffectiveProfileProvenanceReadStatus(StrEnum):
    """Whether profile provenance enrichment was requested for this reconstruction."""

    NOT_CONFIGURED = "not_configured"
    CONFIGURED = "configured"


@runtime_checkable
class ExecutionEffectiveProfileProvenanceReader(Protocol):
    """Read-only provenance lookup — no pin, activate, or resolve-latest semantics."""

    def read(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> ExecutionEffectiveProfileProvenance | None:
        """Return pinned provenance for one tenant-scoped execution, or ``None`` if absent."""


def require_tenant_id_for_profile_provenance(tenant_id: str) -> str:
    normalized = tenant_id.strip()
    if not normalized:
        raise ValueError("tenant_id is required for profile provenance read")
    return normalized


def validate_execution_effective_profile_provenance_record(
    record: ExecutionEffectiveProfileProvenance,
    *,
    expected_tenant_id: str,
    expected_execution_id: ExecutionId,
) -> None:
    tenant = require_tenant_id_for_profile_provenance(expected_tenant_id)
    if record.tenant_id != tenant:
        raise ValueError("profile provenance tenant mismatch")
    if validate_execution_id(record.execution_id) != validate_execution_id(expected_execution_id):
        raise ValueError("profile provenance execution_id mismatch")


__all__ = [
    "ExecutionEffectiveProfileProvenance",
    "ExecutionEffectiveProfileProvenanceReadStatus",
    "ExecutionEffectiveProfileProvenanceReader",
    "require_tenant_id_for_profile_provenance",
    "validate_execution_effective_profile_provenance_record",
]
