# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Strict orchestration Governance Evidence composition (GR-10-R14)."""

from __future__ import annotations

from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernanceEvidencePersistencePort,
)
from intergrax.runtime.events.evidence_persistence_adapter import as_evidence_persistence_port
from intergrax.runtime.governance.governance_evidence_composition import (
    build_governance_evidence_recorder,
    build_runtime_event_governance_evidence_persistence,
)
from intergrax.runtime.governance.governance_evidence_recorder import GovernanceEvidenceRecorder


class OrchestrationGovernanceEvidenceCompositionError(RuntimeError):
    """Fail closed when strict orchestration omits mandatory Governance Evidence persistence."""


def build_orchestration_governance_evidence_recorder(
    *,
    governance_evidence_persistence: GovernanceEvidencePersistencePort,
) -> GovernanceEvidenceRecorder:
    """Wire pluginable GR-8 persistence behind the shared recorder (non-authoritative)."""
    if governance_evidence_persistence is None:
        raise OrchestrationGovernanceEvidenceCompositionError(
            "governance_evidence_persistence must not be None for orchestration evidence",
        )
    return build_governance_evidence_recorder(persistence=governance_evidence_persistence)


def resolve_runtime_event_governance_evidence_persistence(
    runtime_event_persistence: object | None,
) -> GovernanceEvidencePersistencePort | None:
    """Project runtime event persistence into GR-8 port when available."""
    if runtime_event_persistence is None:
        return None
    if isinstance(runtime_event_persistence, EvidencePersistencePort):
        evidence_port = runtime_event_persistence
    else:
        evidence_port = as_evidence_persistence_port(runtime_event_persistence)
    return build_runtime_event_governance_evidence_persistence(
        evidence_persistence=evidence_port,
    )


def require_strict_orchestration_governance_evidence_persistence(
    *,
    explicit: GovernanceEvidencePersistencePort | None,
    runtime_event_persistence: object | None,
    production_mode: bool,
) -> GovernanceEvidencePersistencePort | None:
    """Strict production orchestration requires an injected or runtime-event-backed GR-8 port."""
    if explicit is not None:
        return explicit
    if not production_mode:
        return None
    resolved = resolve_runtime_event_governance_evidence_persistence(
        runtime_event_persistence,
    )
    if resolved is None:
        raise OrchestrationGovernanceEvidenceCompositionError(
            "strict orchestration requires GovernanceEvidencePersistencePort; "
            "inject governance_evidence_persistence or enable runtime event persistence",
        )
    return resolved


__all__ = [
    "OrchestrationGovernanceEvidenceCompositionError",
    "build_orchestration_governance_evidence_recorder",
    "require_strict_orchestration_governance_evidence_persistence",
    "resolve_runtime_event_governance_evidence_persistence",
]
