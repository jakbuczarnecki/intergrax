# © Artur Czarnecki. All rights reserved.

"""Multi-region security audit trail bookkeeping (AUDIT-IDEAL-23.1).

The "regions" are in-process bookkeeping: per-region lists of entry ids that live in this process
next to one primary trail. Nothing is replicated to another store, the stored entries carry no hash
chain or signature, and ``verify_replication`` never reads them back. See the classes below for what
the report does and does not prove.
"""

from __future__ import annotations

from intergrax.runtime.security.security_audit_trail import (
    SecurityAuditEntry,
    SecurityAuditSink,
    SecurityAuditTrail,
)
from pydantic import BaseModel, Field


class AuditRegionReplica(BaseModel):
    region_id: str
    entry_count: int = Field(ge=0)
    head_entry_id: str | None = None


class MultiRegionAuditTrailReport(BaseModel):
    """Consistency report of the in-process region bookkeeping (not a storage guarantee)."""

    schema_version: str = "1.0.0"
    regions: list[AuditRegionReplica] = Field(default_factory=list)
    replicated: bool = Field(
        description=(
            "The in-process per-region id lists agree with each other (same count and head); "
            "this does not prove that any replica store exists."
        ),
    )
    immutable: bool = Field(
        description=(
            "The in-process per-region id lists still start with the prefix sealed by seal_prefix; "
            "this does not prove that the stored entries are unchanged."
        ),
    )


class MultiRegionSecurityAuditTrail:
    """Append-only audit trail with in-process per-region bookkeeping over one primary trail.

    Each append goes to the primary trail and the new entry id is recorded for every configured
    region in this process. ``verify_replication`` compares those id lists with each other and with
    the prefix sealed by ``seal_prefix``; it does not detect stored entries that were deleted or
    altered in the primary trail, and the lists are empty after a restart.
    """

    def __init__(
        self,
        *,
        regions: tuple[str, ...],
        primary: SecurityAuditSink | None = None,
    ) -> None:
        if len(regions) < 2:
            raise ValueError("multi-region audit trail requires at least two regions")
        self._regions = regions
        # ``primary`` is the durable evidence-plane trail on product hosts (PDC-D6-R020);
        # the in-memory default is for tests and lab hosts only.
        self._primary: SecurityAuditSink = primary if primary is not None else SecurityAuditTrail()
        self._region_entry_ids: dict[str, list[str]] = {region: [] for region in regions}
        self._sealed_prefix: dict[str, tuple[str, ...]] = {region: () for region in regions}

    def append(
        self,
        *,
        tenant_id: str,
        action: str,
        actor_id: str,
        resource: str,
        metadata: dict[str, object] | None = None,
    ) -> SecurityAuditEntry:
        entry = self._primary.append(
            tenant_id=tenant_id,
            action=action,
            actor_id=actor_id,
            resource=resource,
            metadata=metadata,
        )
        for region in self._regions:
            self._region_entry_ids[region].append(entry.entry_id)
        return entry

    def list_entries(self, *, tenant_id: str | None = None) -> tuple[SecurityAuditEntry, ...]:
        return self._primary.list_entries(tenant_id=tenant_id)

    def verify_replication(self) -> MultiRegionAuditTrailReport:
        replicas = [
            AuditRegionReplica(
                region_id=region,
                entry_count=len(self._region_entry_ids[region]),
                head_entry_id=self._region_entry_ids[region][-1] if self._region_entry_ids[region] else None,
            )
            for region in self._regions
        ]
        counts = {replica.entry_count for replica in replicas}
        heads = {replica.head_entry_id for replica in replicas if replica.head_entry_id is not None}
        replicated = len(counts) == 1 and (not heads or len(heads) == 1)
        immutable = replicated and all(
            tuple(self._region_entry_ids[region][: len(self._sealed_prefix[region])]) == self._sealed_prefix[region]
            for region in self._regions
        )
        return MultiRegionAuditTrailReport(
            regions=replicas,
            replicated=replicated,
            immutable=immutable,
        )

    def seal_prefix(self) -> None:
        """Freeze current entries to detect tampering in immutability checks."""
        for region in self._regions:
            self._sealed_prefix[region] = tuple(self._region_entry_ids[region])
