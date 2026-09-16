# © Artur Czarnecki. All rights reserved.

"""Composition-root wiring for Collaborative Decision Binding application boundary (MP-4R5)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from intergrax.collaborative_work.decision_binding_application import (
    CollaborativeDecisionBindingApplicationService,
    CollaborativeDecisionBindingEvidenceAdoption,
)
from intergrax.collaborative_work.decision_binding_service import CollaborativeDecisionBindingService
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositoriesWithArtifacts
from intergrax.collaborative_work.repository import (
    CollaborativeDecisionBindingRepository,
    WorkArtifactVersionRepository,
    WorkItemRepository,
)


def build_collaborative_decision_binding_service(
    *,
    work_item_repository: WorkItemRepository,
    work_artifact_version_repository: WorkArtifactVersionRepository,
    binding_repository: CollaborativeDecisionBindingRepository,
    enforcement_gate: CollaborativeWorkEnforcementGate,
    clock: Callable[[], datetime] | None = None,
) -> CollaborativeDecisionBindingService:
    """Construct the authoritative domain binding service (MP-4R4 surface)."""
    return CollaborativeDecisionBindingService(
        work_item_repository=work_item_repository,
        work_artifact_version_repository=work_artifact_version_repository,
        binding_repository=binding_repository,
        enforcement_gate=enforcement_gate,
        clock=clock,
    )


def build_collaborative_decision_binding_application_service(
    *,
    binding_service: CollaborativeDecisionBindingService,
    evidence_adoption: CollaborativeDecisionBindingEvidenceAdoption | None = None,
    clock: Callable[[], datetime] | None = None,
) -> CollaborativeDecisionBindingApplicationService:
    """Construct application orchestration with optional evidence adoption."""
    return CollaborativeDecisionBindingApplicationService(
        binding_service=binding_service,
        evidence_adoption=evidence_adoption,
        clock=clock,
    )


def build_collaborative_decision_binding_application_from_artifacts_bundle(
    *,
    bundle: CollaborativeWorkRepositoriesWithArtifacts,
    enforcement_gate: CollaborativeWorkEnforcementGate,
    evidence_adoption: CollaborativeDecisionBindingEvidenceAdoption | None = None,
    clock: Callable[[], datetime] | None = None,
) -> CollaborativeDecisionBindingApplicationService:
    """
    Production composition path: materialized Collaborative Work persistence + optional evidence.

    Default projection/persistence implementations are selected only by the composition root caller.
    """
    binding_service = build_collaborative_decision_binding_service(
        work_item_repository=bundle.work_item,
        work_artifact_version_repository=bundle.version,
        binding_repository=bundle.decision_binding,
        enforcement_gate=enforcement_gate,
        clock=clock,
    )
    return build_collaborative_decision_binding_application_service(
        binding_service=binding_service,
        evidence_adoption=evidence_adoption,
        clock=clock,
    )


__all__ = [
    "build_collaborative_decision_binding_application_from_artifacts_bundle",
    "build_collaborative_decision_binding_application_service",
    "build_collaborative_decision_binding_service",
]
