# © Artur Czarnecki. All rights reserved.

"""Immutable multi-region security audit trail (AUDIT-IDEAL-23.1)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.runtime.security.security_audit_trail import SecurityAuditEntry, SecurityAuditTrail


class AuditRegionReplica(BaseModel):
    region_id: str
    entry_count: int = Field(ge=0)
    head_entry_id: str | None = None


class MultiRegionAuditTrailReport(BaseModel):
    schema_version: str = "1.0.0"
    regions: list[AuditRegionReplica] = Field(default_factory=list)
    replicated: bool
    immutable: bool


class MultiRegionSecurityAuditTrail:
    """Append-only audit trail replicated across configured regions."""

    def __init__(self, *, regions: tuple[str, ...]) -> None:
        if len(regions) < 2:
            raise ValueError("multi-region audit trail requires at least two regions")
        self._regions = regions
        self._primary = SecurityAuditTrail()
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

    def verify_replication(self) -> MultiRegionAuditTrailReport:
        replicas = [
            AuditRegionReplica(
                region_id=region,
                entry_count=len(self._region_entry_ids[region]),
                head_entry_id=self._region_entry_ids[region][-1]
                if self._region_entry_ids[region]
                else None,
            )
            for region in self._regions
        ]
        counts = {replica.entry_count for replica in replicas}
        heads = {replica.head_entry_id for replica in replicas if replica.head_entry_id is not None}
        replicated = len(counts) == 1 and (not heads or len(heads) == 1)
        immutable = replicated and all(
            tuple(self._region_entry_ids[region][: len(self._sealed_prefix[region])])
            == self._sealed_prefix[region]
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
