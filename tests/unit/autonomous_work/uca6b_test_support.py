# © Artur Czarnecki. All rights reserved.

"""UCA-6B test wiring — canonical discovery + recording UCA ports."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.autonomous_work.catalog_canonical_discovery_service import (
    CatalogCanonicalCapabilityDiscoveryService,
)
from intergrax.autonomous_work.worker_capability_recovery_coordinator import (
    WorkerCapabilityRecoveryCoordinator,
)
from intergrax.capability_acquisition.permit_acquisition_authorization import (
    PermitCapabilityAcquisitionAuthorizationPort,
)
from intergrax.capability_acquisition.acquisition_service import CapabilityAcquisitionService
from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    CapabilityAcquisitionEvidence,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.runtime import ToolRegistry
from tests.unit.autonomous_work.catalog_discovery_test_support import (
    catalog_discovery_dependencies,
    catalog_snapshot_from_registries,
    host_availability_for_entries,
    tool_catalog_entry,
)


@dataclass
class RecordingAcquisitionStrategy:
    """Strategy-opaque fake — proves AW does not branch on strategy_id."""

    strategy_id: str
    outcome: CapabilityAcquisitionOutcome = CapabilityAcquisitionOutcome.SUCCEEDED
    calls: int = 0

    @property
    def supported_kinds(self) -> frozenset[CapabilityKind]:
        return frozenset(CapabilityKind)

    def supports(self, request: CapabilityAcquisitionRequest) -> bool:
        return True

    def acquire(self, request: CapabilityAcquisitionRequest) -> CapabilityAcquisitionResult:
        self.calls += 1
        completed = request.requested_at
        evidence = None
        if self.outcome is CapabilityAcquisitionOutcome.SUCCEEDED:
            evidence = CapabilityAcquisitionEvidence(
                artifact_reference=f"artifact://{self.strategy_id}/{request.request_id}",
            )
        return CapabilityAcquisitionResult(
            request_id=request.request_id,
            gap_id=request.capability_gap.gap_id,
            strategy_id=self.strategy_id,
            outcome=self.outcome,
            reason_code=CapabilityAcquisitionReasonCode.NONE,
            started_at=request.requested_at,
            completed_at=completed,
            evidence=evidence,
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
        )


@dataclass
class RecordingAcquisitionBundle:
    strategy: RecordingAcquisitionStrategy
    service: CapabilityAcquisitionService


def build_recording_acquisition(
    *,
    strategy_id: str = "custom.external.v1",
    outcome: CapabilityAcquisitionOutcome = CapabilityAcquisitionOutcome.SUCCEEDED,
) -> RecordingAcquisitionBundle:
    strategy = RecordingAcquisitionStrategy(strategy_id=strategy_id, outcome=outcome)
    service = CapabilityAcquisitionService(
        (strategy,),
        authorization=PermitCapabilityAcquisitionAuthorizationPort(),
    )
    return RecordingAcquisitionBundle(strategy=strategy, service=service)


def build_test_coordinator(
    *,
    tool_registry: ToolRegistry,
    skill_registry: SkillRegistry,
    acquisition: RecordingAcquisitionBundle | None = None,
) -> WorkerCapabilityRecoveryCoordinator:
    bundle = acquisition or build_recording_acquisition()
    snapshot = catalog_snapshot_from_registries(
        tool_registry=tool_registry,
        skill_registry=skill_registry,
    )
    host_entries = [
        entry
        for entry in snapshot.entries
        if entry.identity.logical.logical_id
        in {reg.contract.tool_id for reg in tool_registry.list()}
        or entry.identity.logical.logical_id
        in {reg.manifest.skill_id for reg in skill_registry.list()}
    ]
    availability = host_availability_for_entries(*host_entries) if host_entries else None
    if availability is None:
        availability = host_availability_for_entries(
            *(tool_catalog_entry(reg.contract.tool_id) for reg in tool_registry.list()),
        )
    dependencies = catalog_discovery_dependencies(
        snapshot=snapshot,
        availability_evidence=availability,
    )
    discovery = CatalogCanonicalCapabilityDiscoveryService(
        dependencies=dependencies,
        skill_registry=skill_registry,
    )
    return WorkerCapabilityRecoveryCoordinator(
        discovery=discovery,
        acquisition=bundle.service,
    )
