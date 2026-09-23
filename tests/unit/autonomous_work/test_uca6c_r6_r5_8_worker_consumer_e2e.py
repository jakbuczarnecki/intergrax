# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.8 — worker consumer canonical fulfillment E2E."""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.autonomous_work.host_available_capability_binding_service import (
    HostAvailableCapabilityBindingService,
)
from intergrax.autonomous_work.host_available_tool_capability_binding_provider import (
    HostAvailableToolCapabilityBindingProvider,
)
from intergrax.autonomous_work.worker_capability_direct_reuse_fulfillment_service import (
    WorkerCapabilityDirectReuseFulfillmentService,
)
from intergrax.autonomous_work.worker_capability_fulfillment_coordinator import (
    WorkerCapabilityFulfillmentCoordinator,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_service import (
    WorkerRecoveryCapabilityFulfillmentService,
)
from intergrax.autonomous_work.worker_capability_recovery_coordinator import (
    WorkerCapabilityRecoveryCoordinator,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_coordinator import (
    WorkerQualifiedCapabilityResumeCoordinator,
)
from intergrax.capability_qualification.qualified_capability_binding_service import (
    QualifiedCapabilityBindingService,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    WorkerCapabilityAcquisitionRequest,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentDisposition,
    WorkerCapabilityFulfillmentRequest,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryPhase,
)
from intergrax.contracts.capability_acquisition.outcome import (
    CapabilityRealizationOutcome,
)
from intergrax.contracts.capability_acquisition.reason_code import (
    CapabilityRealizationReasonCode,
)
from intergrax.contracts.capability_acquisition.evidence import (
    CapabilityRealizationEvidence,
)
from intergrax.contracts.capability_acquisition.result import (
    CapabilityRealizationResult,
)
from intergrax.contracts.capability_catalog.evidence import (
    CapabilityDiscoveryAvailabilityEvidence,
)
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletionOutcome,
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_catalog.identity import CapabilitySourceKind
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_qualification.qualification_evidence import (
    CapabilityQualificationEvidence,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_reason_code import (
    CapabilityQualificationReasonCode,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)
from intergrax.contracts.execution_identity import TaskId
from tests.unit.autonomous_work.test_uca6b_worker_capability_recovery import (
    _PROFILE,
    _recovery_decision,
    _worker_need,
)
from tests.unit.autonomous_work.test_uca6c_worker_qualified_capability_resume import (
    _RecordingBindingProvider,
    _RecordingExecutionPort,
    _TENANT,
    _WORKER_ID,
    _authority_admission,
)
from tests.unit.autonomous_work.uca6c_worker_authority_fixtures import _READ
from tests.unit.autonomous_work.uca6b_test_support import build_recording_acquisition
from intergrax.autonomous_work.worker_capability_need_projection import (
    project_worker_capability_need_to_capability_need,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
_TASK_ID = TaskId("task_" + "f" * 32)
_FULFILLMENT_COORDINATOR_PATH = Path(
    "intergrax/autonomous_work/worker_capability_fulfillment_coordinator.py",
)


class _StaticDiscovery:
    def __init__(self, completion) -> None:
        self._completion = completion
        self.calls = 0

    def complete_discovery(self, request):
        self.calls += 1
        return self._completion


@dataclass
class _StatefulDiscovery:
    """First pass REALIZATION_REQUIRED, then DIRECT_REUSE after host visibility."""

    need_id: str
    catalog_key: CapabilityIdentityKey
    calls: int = 0
    second_pass_outcome: str = "direct_reuse"

    def complete_discovery(self, request):
        self.calls += 1
        if self.calls == 1:
            return build_discovery_completion(
                need_id=self.need_id,
                discovery_correlation_id="corr-r58-realize",
                federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
                created_at=_NOW,
                suitable_catalog_allowed_keys=(self.catalog_key,),
            )
        if self.second_pass_outcome == "direct_reuse":
            return build_discovery_completion(
                need_id=self.need_id,
                discovery_correlation_id="corr-r58-realize",
                federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
                created_at=_NOW,
                suitable_host_allowed_keys=(self.catalog_key,),
            )
        return build_discovery_completion(
            need_id=self.need_id,
            discovery_correlation_id="corr-r58-realize",
            federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
            created_at=_NOW,
            suitable_catalog_allowed_keys=(self.catalog_key,),
        )


@dataclass
class _RecordingQualification:
    calls: int = 0
    outcome: CapabilityQualificationOutcome = CapabilityQualificationOutcome.QUALIFIED

    def qualify(self, request):
        self.calls += 1
        evidence = None
        provider_id = None
        if self.outcome is CapabilityQualificationOutcome.QUALIFIED:
            provider_id = "test.qualification.v1"
            evidence = CapabilityQualificationEvidence(
                provider_id=provider_id,
                qualification_request_id=request.qualification_request_id,
                acquisition_request_id=request.acquisition_request_id,
                acquisition_strategy_id=request.strategy_id,
                gap_id=request.gap_id,
                artifact_reference="artifact://r58/test",
            )
        return CapabilityQualificationResult(
            qualification_request_id=request.qualification_request_id,
            acquisition_request_id=request.acquisition_request_id,
            gap_id=request.gap_id,
            strategy_id=request.strategy_id,
            provider_id=provider_id,
            outcome=self.outcome,
            reason_code=CapabilityQualificationReasonCode.NONE,
            started_at=request.requested_at,
            completed_at=request.requested_at,
            evidence=evidence,
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
        )


@dataclass
class _RecordingRealization:
    calls: int = 0
    outcome: CapabilityRealizationOutcome = CapabilityRealizationOutcome.SUCCEEDED

    def realize(self, request):
        self.calls += 1
        completed = request.requested_at
        need = request.realization_need
        if self.outcome is not CapabilityRealizationOutcome.SUCCEEDED:
            return CapabilityRealizationResult(
                request_id=request.request_id,
                realization_need_id=need.realization_need_id,
                provider_id="test.realization.v1",
                outcome=self.outcome,
                reason_code=CapabilityRealizationReasonCode.NONE,
                capability_identity=need.capability_identity,
                started_at=completed,
                completed_at=completed,
            )
        evidence = CapabilityRealizationEvidence.from_availability_evidence(
            CapabilityDiscoveryAvailabilityEvidence(
                host_available_keys=(need.capability_identity,),
            ),
        )
        return CapabilityRealizationResult(
            request_id=request.request_id,
            realization_need_id=need.realization_need_id,
            provider_id="test.realization.v1",
            outcome=CapabilityRealizationOutcome.SUCCEEDED,
            reason_code=CapabilityRealizationReasonCode.NONE,
            capability_identity=need.capability_identity,
            started_at=completed,
            completed_at=completed,
            evidence=evidence,
        )


@dataclass
class _RecordingDirectReuse:
    calls: int = 0

    def fulfill_direct_reuse(self, request, recovery):
        self.calls += 1
        from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
            WorkerCapabilityFulfillmentResult,
        )

        return WorkerCapabilityFulfillmentResult(
            disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED,
            provenance=recovery.provenance,
            recovery_outcome=recovery,
            decided_at=request.requested_at,
        )


def _acquisition_request() -> WorkerCapabilityAcquisitionRequest:
    need = _worker_need()
    return WorkerCapabilityAcquisitionRequest(
        need=need,
        recovery_decision=_recovery_decision(need),
        capability_profile_ref=_PROFILE,
    )


def _fulfillment_request(
    *,
    allow_generic_acquisition: bool = True,
) -> WorkerCapabilityFulfillmentRequest:
    return WorkerCapabilityFulfillmentRequest(
        acquisition_request=_acquisition_request(),
        worker_instance_id=_WORKER_ID,
        tenant_id=_TENANT,
        task_id=_TASK_ID,
        requested_at=_NOW,
        requested_authority_scopes=(_READ,),
        allow_generic_acquisition=allow_generic_acquisition,
    )


def _production_direct_reuse() -> WorkerCapabilityDirectReuseFulfillmentService:
    binding = HostAvailableCapabilityBindingService(
        (HostAvailableToolCapabilityBindingProvider(),),
    )
    return WorkerCapabilityDirectReuseFulfillmentService(
        binding=binding,
        execution=_RecordingExecutionPort(),
        authority_admission=_authority_admission(),
    )


def _build_fulfillment(
    discovery,
    *,
    acquisition=None,
    qualification=None,
    realization=None,
    direct_reuse=None,
) -> tuple[WorkerCapabilityFulfillmentCoordinator, object, object, object, object]:
    bundle = acquisition or build_recording_acquisition()
    qual = qualification or _RecordingQualification()
    real = realization or _RecordingRealization()
    reuse = direct_reuse or _RecordingDirectReuse()
    recovery = WorkerCapabilityRecoveryCoordinator(
        discovery=discovery,
        acquisition=bundle.service,
        qualification=qual,
    )
    resume = WorkerQualifiedCapabilityResumeCoordinator(
        binding=QualifiedCapabilityBindingService((_RecordingBindingProvider(),)),
        execution=_RecordingExecutionPort(),
        authority_admission=_authority_admission(),
    )
    coordinator = WorkerCapabilityFulfillmentCoordinator(
        recovery=recovery,
        resume=resume,
        direct_reuse=reuse,
        realization=real,
    )
    return coordinator, bundle, qual, real, reuse


def test_capability_gap_no_acquisition_no_execution() -> None:
    need = _worker_need()
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-gap-r58",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
    )
    discovery = _StaticDiscovery(completion)
    coordinator, bundle, _, _, _ = _build_fulfillment(discovery)
    result = coordinator.fulfill(
        _fulfillment_request(allow_generic_acquisition=False),
    )
    assert result.disposition is WorkerCapabilityFulfillmentDisposition.CAPABILITY_GAP
    assert result.capability_gap is not None
    assert bundle.strategy.calls == 0
    assert discovery.calls == 1


def test_blocked_discovery_not_capability_gap() -> None:
    need = _worker_need()
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-blocked-r58",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
        governance_blocked=True,
    )
    discovery = _StaticDiscovery(completion)
    coordinator, bundle, _, _, _ = _build_fulfillment(discovery)
    result = coordinator.fulfill(_fulfillment_request())
    assert (
        result.disposition is WorkerCapabilityFulfillmentDisposition.DISCOVERY_BLOCKED
    )
    assert result.capability_gap is None
    assert bundle.strategy.calls == 0
    assert completion.outcome is DiscoveryCompletionOutcome.BLOCKED


def test_realization_required_then_direct_reuse_no_generic_acquisition() -> None:
    need = _worker_need()
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    catalog_key = CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="builtin",
        source_kind=CapabilitySourceKind.BUILTIN,
        logical_id="document.parse_csv",
    )
    discovery = _StatefulDiscovery(
        canonical_id or "need",
        catalog_key,
    )
    execution = _RecordingExecutionPort()
    direct_reuse = WorkerCapabilityDirectReuseFulfillmentService(
        binding=HostAvailableCapabilityBindingService(
            (HostAvailableToolCapabilityBindingProvider(),),
        ),
        execution=execution,
        authority_admission=_authority_admission(),
    )
    coordinator, bundle, qual, real, reuse = _build_fulfillment(
        discovery,
        direct_reuse=direct_reuse,
    )
    result = coordinator.fulfill(_fulfillment_request())
    assert discovery.calls == 2
    assert real.calls == 1
    assert bundle.strategy.calls == 0
    assert qual.calls == 0
    assert execution.calls == 1
    assert (
        result.disposition
        is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED
    )
    assert (
        result.recovery_outcome is not None
        and result.recovery_outcome.phase is WorkerCapabilityRecoveryPhase.DIRECT_REUSE
    )
    assert result.recovery_outcome.discovery_completion is not None
    host_keys = result.recovery_outcome.discovery_completion.suitable_host_allowed_keys
    assert host_keys == (catalog_key,)


def test_qualification_fail_blocks_execution() -> None:
    need = _worker_need()
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-qual-fail",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
    )
    discovery = _StaticDiscovery(completion)
    qual = _RecordingQualification(outcome=CapabilityQualificationOutcome.FAILED)
    coordinator, bundle, _, _, _ = _build_fulfillment(discovery, qualification=qual)
    result = coordinator.fulfill(_fulfillment_request())
    assert bundle.strategy.calls == 1
    assert qual.calls == 1
    assert (
        result.disposition
        is WorkerCapabilityFulfillmentDisposition.QUALIFICATION_FAILED
    )


def test_direct_reuse_skips_acquisition() -> None:
    need = _worker_need()
    host_key = CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="builtin",
        source_kind=CapabilitySourceKind.BUILTIN,
        logical_id="document.parse_csv",
    )
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-direct-r58",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
        suitable_host_allowed_keys=(host_key,),
    )
    discovery = _StaticDiscovery(completion)
    reuse = _RecordingDirectReuse()
    coordinator, bundle, _, _, _ = _build_fulfillment(discovery, direct_reuse=reuse)
    result = coordinator.fulfill(_fulfillment_request())
    assert bundle.strategy.calls == 0
    assert reuse.calls == 1
    assert (
        result.disposition
        is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED
    )
    assert (
        result.recovery_outcome is not None
        and result.recovery_outcome.phase is WorkerCapabilityRecoveryPhase.DIRECT_REUSE
    )


def test_fulfillment_coordinator_static_gate_no_runtime_execution_imports() -> None:
    tree = ast.parse(_FULFILLMENT_COORDINATOR_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("intergrax.runtime.execution")
            assert not node.module.startswith("intergrax.runtime.codecraft")
            assert not node.module.startswith("intergrax.marketplace")


def test_fulfillment_coordinator_contract_abstraction_no_concrete_recovery_resume() -> (
    None
):
    tree = ast.parse(_FULFILLMENT_COORDINATOR_PATH.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    joined = "\n".join(imported)
    assert "worker_capability_recovery_coordinator" not in joined
    assert "worker_qualified_capability_resume_coordinator" not in joined


def test_public_fulfillment_request_has_no_post_realization_retry() -> None:
    fields = {
        field.name for field in dataclasses.fields(WorkerCapabilityFulfillmentRequest)
    }
    assert "post_realization_retry" not in fields


def test_realization_success_but_not_visible_fail_closed() -> None:
    need = _worker_need()
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    catalog_key = CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="builtin",
        source_kind=CapabilitySourceKind.BUILTIN,
        logical_id="document.parse_csv",
    )
    discovery = _StatefulDiscovery(
        canonical_id or "need",
        catalog_key,
        second_pass_outcome="still_realization_required",
    )
    coordinator, bundle, _, real, _ = _build_fulfillment(discovery)
    result = coordinator.fulfill(_fulfillment_request())
    assert real.calls == 1
    assert bundle.strategy.calls == 0
    assert discovery.calls == 2
    assert (
        result.disposition
        is WorkerCapabilityFulfillmentDisposition.REALIZATION_NOT_VISIBLE
    )


def test_realization_blocked_no_acquisition_no_execution() -> None:
    need = _worker_need()
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    catalog_key = CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="builtin",
        source_kind=CapabilitySourceKind.BUILTIN,
        logical_id="document.parse_csv",
    )
    discovery = _StatefulDiscovery(canonical_id or "need", catalog_key)
    real = _RecordingRealization(
        outcome=CapabilityRealizationOutcome.BLOCKED,
    )
    execution = _RecordingExecutionPort()
    coordinator, bundle, _, _, _ = _build_fulfillment(
        discovery,
        realization=real,
        direct_reuse=WorkerCapabilityDirectReuseFulfillmentService(
            binding=HostAvailableCapabilityBindingService(
                (HostAvailableToolCapabilityBindingProvider(),),
            ),
            execution=execution,
            authority_admission=_authority_admission(),
        ),
    )
    result = coordinator.fulfill(_fulfillment_request())
    assert real.calls == 1
    assert bundle.strategy.calls == 0
    assert execution.calls == 0
    assert (
        result.disposition is WorkerCapabilityFulfillmentDisposition.DISCOVERY_BLOCKED
    )


def test_direct_reuse_production_adapter_binding_and_execution() -> None:
    need = _worker_need()
    host_key = CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="builtin",
        source_kind=CapabilitySourceKind.BUILTIN,
        logical_id="document.parse_csv",
    )
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-direct-prod-r58",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
        suitable_host_allowed_keys=(host_key,),
    )
    discovery = _StaticDiscovery(completion)
    binding_provider = HostAvailableToolCapabilityBindingProvider()
    execution = _RecordingExecutionPort()
    direct_reuse = WorkerCapabilityDirectReuseFulfillmentService(
        binding=HostAvailableCapabilityBindingService((binding_provider,)),
        execution=execution,
        authority_admission=_authority_admission(),
    )
    coordinator, bundle, _, _, _ = _build_fulfillment(
        discovery,
        direct_reuse=direct_reuse,
    )
    result = coordinator.fulfill(_fulfillment_request())
    assert bundle.strategy.calls == 0
    assert execution.calls == 1
    assert execution.last_request is not None
    assert (
        result.disposition
        is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED
    )


@dataclass
class _RecordingFulfillmentPort:
    calls: int = 0

    def fulfill(self, request, *, decided_at=None):
        self.calls += 1
        from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
            WorkerCapabilityFulfillmentResult,
        )
        from intergrax.contracts.autonomous_work.worker_capability_recovery import (
            WorkerCapabilityRecoveryProvenance,
        )

        return WorkerCapabilityFulfillmentResult(
            disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED,
            provenance=WorkerCapabilityRecoveryProvenance(
                worker_need_id="need",
                canonical_need_id="canonical",
                discovery_correlation_id="corr",
                discovery_completion_outcome="direct_reuse",
                evidence_refs=(),
            ),
            decided_at=request.requested_at,
        )


@dataclass
class _StaticFulfillmentRequestBuilder:
    fulfillment_request: WorkerCapabilityFulfillmentRequest

    def build_fulfillment_request(self, *, episode, request):
        return self.fulfillment_request


@pytest.mark.asyncio
async def test_worker_recovery_orchestration_uses_fulfillment_port() -> None:
    from intergrax.contracts.autonomous_work.obstacle_recovery import RecoveryStrategy
    from tests.unit.autonomous_work.test_worker_recovery_orchestration import (
        _harness,
        _orchestration_request,
    )

    recording = _RecordingFulfillmentPort()
    recovery_fulfillment = WorkerRecoveryCapabilityFulfillmentService(
        fulfillment=recording,
    )
    service, _ = _harness()
    service._recovery_capability_fulfillment = recovery_fulfillment
    service._recovery_capability_fulfillment_request_builder = (
        _StaticFulfillmentRequestBuilder(_fulfillment_request())
    )
    orch_request = _orchestration_request(
        decision=__import__(
            "tests.unit.autonomous_work.test_worker_recovery_orchestration",
            fromlist=["_decision"],
        )._decision(strategy=RecoveryStrategy.ACQUIRE_CAPABILITY),
    )
    await service.orchestrate(orch_request)
    assert recording.calls == 1
