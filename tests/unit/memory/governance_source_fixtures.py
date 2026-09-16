# © Artur Czarnecki. All rights reserved.

"""Test doubles for canonical memory governance source authority (MEM-ENT-10B-R)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.data_classification import DataClassification
from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordGovernance,
    MemoryRecordSourceType,
    MemoryRecordTrust,
    MemoryTrustClass,
)
from intergrax.memory.contracts.entity_temporal_memory import EntityMemoryScope
from intergrax.memory.contracts.memory_security_governance import (
    CanonicalMemoryGovernanceSourceMismatch,
    CanonicalMemoryGovernanceSourceNotFound,
    CanonicalMemoryGovernanceSourceSnapshot,
    validate_canonical_governance_source_snapshot,
)


def permissive_governance_snapshot(
    scope: EntityMemoryScope,
    memory_id: str,
    revision: int,
) -> CanonicalMemoryGovernanceSourceSnapshot:
    return CanonicalMemoryGovernanceSourceSnapshot(
        scope=scope,
        memory_id=memory_id,
        revision=revision,
        provenance=MemoryProvenance(source_type=MemoryRecordSourceType.USER_EXPLICIT),
        trust=MemoryRecordTrust(trust_class=MemoryTrustClass.USER_EXPLICIT),
        governance=MemoryRecordGovernance(),
    )


@dataclass
class ScopedCanonicalGovernanceSourceAuthority:
    """Scope-bound fake ``CanonicalMemoryGovernanceSourceAuthority``."""

    scope: EntityMemoryScope
    snapshots: dict[tuple[str, int], CanonicalMemoryGovernanceSourceSnapshot] = field(
        default_factory=dict
    )
    default_factory: bool = True

    def resolve_canonical_governance_source(
        self,
        scope: EntityMemoryScope,
        memory_id: str,
        revision: int,
    ) -> CanonicalMemoryGovernanceSourceSnapshot:
        if (
            scope.tenant_id != self.scope.tenant_id
            or scope.user_id != self.scope.user_id
            or scope.workspace_id != self.scope.workspace_id
        ):
            raise CanonicalMemoryGovernanceSourceNotFound("canonical governance scope mismatch")
        key = (memory_id, revision)
        if key in self.snapshots:
            snapshot = self.snapshots[key]
        elif self.default_factory:
            snapshot = permissive_governance_snapshot(self.scope, memory_id, revision)
        else:
            raise CanonicalMemoryGovernanceSourceNotFound(
                f"canonical governance not found: {memory_id}@{revision}"
            )
        validate_canonical_governance_source_snapshot(scope, memory_id, revision, snapshot)
        return snapshot


def restricted_governance_snapshot(
    scope: EntityMemoryScope,
    memory_id: str,
    revision: int,
) -> CanonicalMemoryGovernanceSourceSnapshot:
    return CanonicalMemoryGovernanceSourceSnapshot(
        scope=scope,
        memory_id=memory_id,
        revision=revision,
        provenance=MemoryProvenance(source_type=MemoryRecordSourceType.USER_EXPLICIT),
        trust=MemoryRecordTrust(trust_class=MemoryTrustClass.USER_EXPLICIT),
        governance=MemoryRecordGovernance(data_classification=DataClassification.RESTRICTED),
    )


@dataclass
class PermissiveCanonicalGovernanceSourceAuthority:
    """Accepts any scope; returns mapped or default permissive governance snapshots."""

    snapshots: dict[tuple[str, int], CanonicalMemoryGovernanceSourceSnapshot] = field(
        default_factory=dict
    )

    def resolve_canonical_governance_source(
        self,
        scope: EntityMemoryScope,
        memory_id: str,
        revision: int,
    ) -> CanonicalMemoryGovernanceSourceSnapshot:
        key = (memory_id, revision)
        if key in self.snapshots:
            snapshot = self.snapshots[key]
        else:
            snapshot = permissive_governance_snapshot(scope, memory_id, revision)
        validate_canonical_governance_source_snapshot(scope, memory_id, revision, snapshot)
        return snapshot


def model_inference_governance_snapshot(
    scope: EntityMemoryScope,
    memory_id: str,
    revision: int,
) -> CanonicalMemoryGovernanceSourceSnapshot:
    return CanonicalMemoryGovernanceSourceSnapshot(
        scope=scope,
        memory_id=memory_id,
        revision=revision,
        provenance=MemoryProvenance(source_type=MemoryRecordSourceType.SESSION_EXTRACTION),
        trust=MemoryRecordTrust(trust_class=MemoryTrustClass.MODEL_INFERENCE),
        governance=MemoryRecordGovernance(),
    )
