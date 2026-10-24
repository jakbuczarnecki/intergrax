# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Authoritative worker execution ownership revocation (TOOL-LIFE-X L3-R7).

Generic ``WorkerInstance.revision`` and lifecycle labels are not ownership epochs.
Certified owner-loss requires a durable, fenced revocation record from this port.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable


class CertifiedOwnerLossWorkerOwnershipIndeterminateError(RuntimeError):
    """Worker authority cannot be read reliably — certification denied (fail-closed)."""


class WorkerExecutionOwnershipNotRevokedError(RuntimeError):
    """Bound execution ownership epoch has no authoritative revocation on record."""


class WorkerExecutionOwnershipRevocationUnauthorizedError(RuntimeError):
    """Revocation rejected — caller identity does not match authoritative ownership."""


def worker_execution_revocation_proof_ref(
    *,
    tenant_id: str,
    operation_id: str,
    worker_instance_id: str,
    ownership_epoch: int,
    revocation_id: str,
) -> str:
    return (
        f"wexec-revoke:v1:{tenant_id}:{operation_id}:{worker_instance_id}"
        f":{ownership_epoch}:{revocation_id}"
    )


@dataclass(frozen=True, slots=True)
class WorkerExecutionOwnershipRevocationV1:
    tenant_id: str
    operation_id: str
    worker_instance_id: str
    ownership_epoch: int
    revocation_id: str
    revoked_at: datetime
    proof_ref: str


@runtime_checkable
class WorkerExecutionOwnershipRevocationPort(Protocol):
    """Single semantic authority for fenced worker execution ownership revocation."""

    def authoritative_revocation_proof_ref(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
    ) -> str | None:
        """Return opaque proof ref when revocation is durable; else ``None``."""
        ...

    def revoke_worker_execution_ownership(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
    ) -> WorkerExecutionOwnershipRevocationV1:
        """Record authoritative revocation for the bound epoch (composition-only)."""
        ...
