# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded recovery for unresolved knowledge revision positions (TRACE-BITEMP-2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.bitemporal_knowledge import (
    KnowledgeOrderingScope,
    KnowledgeRevisionPosition,
    KnowledgeRevisionPositionLifecycle,
    KnowledgeRevisionResolutionReason,
    KnowledgeRevisionResolutionRecord,
    KnowledgeRevisionResolutionSource,
    RevisionOrderingAuthority,
)


@dataclass(frozen=True, slots=True)
class UnresolvedRevisionRecoveryResult:
    resolved: tuple[KnowledgeRevisionResolutionRecord, ...]
    remaining_unresolved: tuple[KnowledgeRevisionPosition, ...]


class UnresolvedRevisionRecovery:
    """Synchronous scanner/resolver for watermark-blocking positions."""

    def __init__(self, authority: RevisionOrderingAuthority) -> None:
        self._authority = authority

    def recover_scope(
        self,
        scope: KnowledgeOrderingScope,
        *,
        reason: KnowledgeRevisionResolutionReason = (
            KnowledgeRevisionResolutionReason.NO_CANONICAL_DURABLE_ACCEPTANCE
        ),
        source: KnowledgeRevisionResolutionSource = KnowledgeRevisionResolutionSource.RECOVERY,
        actor_identity: str | None = "unresolved-revision-recovery",
    ) -> UnresolvedRevisionRecoveryResult:
        resolution_authority = self._authority.acquire_resolution_authority(scope)
        resolved: list[KnowledgeRevisionResolutionRecord] = []
        for position in self._authority.unresolved_positions(scope):
            lifecycle = self._authority.position_lifecycle(position)
            if lifecycle is KnowledgeRevisionPositionLifecycle.ACCEPTED:
                continue
            if lifecycle is KnowledgeRevisionPositionLifecycle.TERMINAL_NON_COMMITTED:
                continue
            record = self._authority.resolve_unresolved_position(
                position=position,
                authority=resolution_authority,
                reason=reason,
                source=source,
                actor_identity=actor_identity,
            )
            resolved.append(record)
        remaining = self._authority.unresolved_positions(scope)
        return UnresolvedRevisionRecoveryResult(
            resolved=tuple(resolved),
            remaining_unresolved=remaining,
        )
