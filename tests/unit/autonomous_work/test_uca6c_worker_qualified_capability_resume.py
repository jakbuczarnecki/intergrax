# © Artur Czarnecki. All rights reserved.

"""UCA-6C — post-qualification resume, binding, and Execution Engine handoff."""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.autonomous_work.worker_qualified_capability_resume_coordinator import (
    WorkerQualifiedCapabilityResumeCoordinator,
    derive_qualified_capability_execution_request_id,
)
from intergrax.capability_qualification.qualified_capability_binding_service import (
    QualifiedCapabilityBindingService,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryProvenance,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionDisposition,
    WorkerQualifiedCapabilityExecutionRequest,
    WorkerQualifiedCapabilityExecutionResult,
    WorkerQualifiedCapabilityResumeOutcome,
    WorkerQualifiedCapabilityResumeRequest,
    derive_worker_capability_resume_operation_id,
)
from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    CapabilityAcquisitionEvidence,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
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
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingOutcome,
    QualifiedCapabilityBindingReasonCode,
    QualifiedCapabilityBindingRequest,
    QualifiedCapabilityBindingResult,
    QualifiedCapabilityExecutionTarget,
    derive_qualified_capability_binding_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_subject import (
    QualifiedCapabilitySubjectKind,
)
from intergrax.contracts.execution_identity import (
    ExecutionId,
    RunId,
    TaskId,
)
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)
_WORKER_ID = contract_suite.mint_worker_instance_id()
_TASK_ID = TaskId("task_" + "b" * 32)
_RUN_ID = RunId("run_" + "c" * 32)
_EXEC_ID = ExecutionId("exec_" + "d" * 32)
_TENANT = "tenant-uca6c"
_RECOVERY_DECISION = "recovery:uca6c:1"
_ACQ_REQUEST = "capability-acquisition-request:gap-1:nonce-1"
_QUAL_REQUEST = "capability-qualification-request:acq-1:qual-1"
_ARTIFACT = "artifact://external/custom-capability/v1"
_GAP = "capability-gap:need-1:missing"


@dataclass
class _RecordingBindingProvider:
    provider_id: str = "custom.external.binding"
    bind_calls: int = 0
    last_request: QualifiedCapabilityBindingRequest | None = None
    _cache: dict[str, QualifiedCapabilityBindingResult] = field(default_factory=dict)

    def supports(self, request: QualifiedCapabilityBindingRequest) -> bool:
        return (
            request.qualified_subject.subject_kind
            is QualifiedCapabilitySubjectKind.ARTIFACT_REFERENCE
        )

    def bind(
        self,
        request: QualifiedCapabilityBindingRequest,
    ) -> QualifiedCapabilityBindingResult:
        self.bind_calls += 1
        self.last_request = request
        cached = self._cache.get(request.binding_operation_id)
        if cached is not None:
            return cached
        completed = request.requested_at
        target = QualifiedCapabilityExecutionTarget(
            execution_target_reference=f"execution-target:{request.binding_operation_id}",
            binding_provider_id=self.provider_id,
            qualified_subject_reference=request.qualified_subject.qualified_subject_reference,
        )
        result = QualifiedCapabilityBindingResult(
            binding_operation_id=request.binding_operation_id,
            outcome=QualifiedCapabilityBindingOutcome.BOUND,
            reason_code=QualifiedCapabilityBindingReasonCode.NONE,
            provider_id=self.provider_id,
            execution_target=target,
            started_at=completed,
            completed_at=completed,
        )
        self._cache[request.binding_operation_id] = result
        return result


@dataclass
class _RecordingExecutionPort:
    calls: int = 0
    last_request: WorkerQualifiedCapabilityExecutionRequest | None = None

    def execute(
        self,
        request: WorkerQualifiedCapabilityExecutionRequest,
    ) -> WorkerQualifiedCapabilityExecutionResult:
        self.calls += 1
        self.last_request = request
        return WorkerQualifiedCapabilityExecutionResult(
            disposition=WorkerQualifiedCapabilityExecutionDisposition.DISPATCHED,
            execution_request_id=derive_qualified_capability_execution_request_id(
                resume_operation_id=request.resume_operation_id,
                binding_operation_id=request.binding_operation_id,
            ),
            run_id=_RUN_ID,
            execution_id=_EXEC_ID,
        )


@dataclass
class _BlockedBindingProvider:
    provider_id: str = "blocked.binding"

    def supports(self, request: QualifiedCapabilityBindingRequest) -> bool:
        return True

    def bind(
        self,
        request: QualifiedCapabilityBindingRequest,
    ) -> QualifiedCapabilityBindingResult:
        completed = request.requested_at
        return QualifiedCapabilityBindingResult(
            binding_operation_id=request.binding_operation_id,
            outcome=QualifiedCapabilityBindingOutcome.BLOCKED,
            reason_code=QualifiedCapabilityBindingReasonCode.POLICY_BLOCKED,
            started_at=completed,
            completed_at=completed,
        )


def _provenance() -> WorkerCapabilityRecoveryProvenance:
    return WorkerCapabilityRecoveryProvenance(
        worker_need_id="worker-need:uca6c:1",
        canonical_need_id="capability-need:tool:1",
        discovery_correlation_id="aw-canonical-discovery:worker-need:uca6c:1",
        discovery_completion_outcome="missing_capability",
        gap_id=_GAP,
        acquisition_request_id=_ACQ_REQUEST,
        acquisition_strategy_id="custom.external.v1",
        qualification_request_id=_QUAL_REQUEST,
    )


def _acquisition() -> CapabilityAcquisitionResult:
    return CapabilityAcquisitionResult(
        request_id=_ACQ_REQUEST,
        gap_id=_GAP,
        strategy_id="custom.external.v1",
        outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
        reason_code=CapabilityAcquisitionReasonCode.NONE,
        started_at=_NOW,
        completed_at=_NOW,
        evidence=CapabilityAcquisitionEvidence(artifact_reference=_ARTIFACT),
    )


def _qualification(
    *,
    outcome: CapabilityQualificationOutcome = CapabilityQualificationOutcome.QUALIFIED,
) -> CapabilityQualificationResult:
    evidence = None
    provider_id = None
    if outcome is CapabilityQualificationOutcome.QUALIFIED:
        provider_id = "custom.external.qualification"
        evidence = CapabilityQualificationEvidence(
            provider_id=provider_id,
            qualification_request_id=_QUAL_REQUEST,
            acquisition_request_id=_ACQ_REQUEST,
            acquisition_strategy_id="custom.external.v1",
            gap_id=_GAP,
            artifact_reference=_ARTIFACT,
        )
    return CapabilityQualificationResult(
        qualification_request_id=_QUAL_REQUEST,
        acquisition_request_id=_ACQ_REQUEST,
        gap_id=_GAP,
        strategy_id="custom.external.v1",
        provider_id=provider_id,
        outcome=outcome,
        reason_code=CapabilityQualificationReasonCode.NONE,
        started_at=_NOW,
        completed_at=_NOW,
        evidence=evidence,
    )


def _resume_request(
    qualification: CapabilityQualificationResult,
) -> WorkerQualifiedCapabilityResumeRequest:
    resume_id = derive_worker_capability_resume_operation_id(
        recovery_decision_id=_RECOVERY_DECISION,
        qualification_request_id=qualification.qualification_request_id,
    )
    return WorkerQualifiedCapabilityResumeRequest(
        worker_instance_id=_WORKER_ID,
        worker_need_id="worker-need:uca6c:1",
        recovery_decision_id=_RECOVERY_DECISION,
        provenance=_provenance(),
        acquisition_result=_acquisition(),
        qualification_result=qualification,
        resume_operation_id=resume_id,
        tenant_id=_TENANT,
        task_id=_TASK_ID,
        requested_at=_NOW,
    )


def _coordinator(
    binding_provider: _RecordingBindingProvider | _BlockedBindingProvider,
    execution: _RecordingExecutionPort | None = None,
) -> WorkerQualifiedCapabilityResumeCoordinator:
    binding_service = QualifiedCapabilityBindingService((binding_provider,))
    return WorkerQualifiedCapabilityResumeCoordinator(
        binding=binding_service,
        execution=execution or _RecordingExecutionPort(),
    )


def test_qualified_resume_binds_once_and_executes_through_port() -> None:
    binding = _RecordingBindingProvider()
    execution = _RecordingExecutionPort()
    coordinator = _coordinator(binding, execution)
    request = _resume_request(_qualification())
    result = coordinator.resume(request)

    assert result.outcome is WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED
    assert binding.bind_calls == 1
    assert execution.calls == 1
    assert result.provenance.qualified_subject_reference is not None
    assert result.provenance.binding_operation_id is not None
    assert result.provenance.execution_request_id is not None
    assert (
        result.provenance.qualification_request_id
        == request.qualification_result.qualification_request_id
    )


def test_succeeded_acquisition_without_qualification_does_not_execute() -> None:
    binding = _RecordingBindingProvider()
    execution = _RecordingExecutionPort()
    coordinator = _coordinator(binding, execution)
    blocked = _qualification(outcome=CapabilityQualificationOutcome.BLOCKED)
    result = coordinator.resume(_resume_request(blocked))

    assert (
        result.outcome
        is WorkerQualifiedCapabilityResumeOutcome.QUALIFICATION_NOT_ELIGIBLE
    )
    assert binding.bind_calls == 0
    assert execution.calls == 0


def test_failed_qualification_no_binding_or_execution() -> None:
    binding = _RecordingBindingProvider()
    execution = _RecordingExecutionPort()
    coordinator = _coordinator(binding, execution)
    failed = _qualification(outcome=CapabilityQualificationOutcome.FAILED)
    result = coordinator.resume(_resume_request(failed))

    assert (
        result.outcome
        is WorkerQualifiedCapabilityResumeOutcome.QUALIFICATION_NOT_ELIGIBLE
    )
    assert binding.bind_calls == 0
    assert execution.calls == 0


def test_hitl_qualification_does_not_bind() -> None:
    binding = _RecordingBindingProvider()
    execution = _RecordingExecutionPort()
    coordinator = _coordinator(binding, execution)
    hitl = _qualification(outcome=CapabilityQualificationOutcome.REQUIRES_HITL)
    result = coordinator.resume(_resume_request(hitl))

    assert (
        result.outcome
        is WorkerQualifiedCapabilityResumeOutcome.QUALIFICATION_NOT_ELIGIBLE
    )
    assert binding.bind_calls == 0
    assert execution.calls == 0


def test_binding_blocked_skips_execution() -> None:
    binding = _BlockedBindingProvider()
    execution = _RecordingExecutionPort()
    coordinator = _coordinator(binding, execution)
    result = coordinator.resume(_resume_request(_qualification()))

    assert result.outcome is WorkerQualifiedCapabilityResumeOutcome.BINDING_BLOCKED
    assert execution.calls == 0


def test_binding_unavailable_without_provider() -> None:
    execution = _RecordingExecutionPort()
    coordinator = WorkerQualifiedCapabilityResumeCoordinator(
        binding=QualifiedCapabilityBindingService(()),
        execution=execution,
    )
    result = coordinator.resume(_resume_request(_qualification()))

    assert result.outcome is WorkerQualifiedCapabilityResumeOutcome.BINDING_UNAVAILABLE
    assert execution.calls == 0


def test_custom_binding_plugin_without_aw_core_changes() -> None:
    binding = _RecordingBindingProvider(provider_id="vendor.acme.binding.v2")
    coordinator = _coordinator(binding)
    coordinator.resume(_resume_request(_qualification()))
    assert binding.last_request is not None
    assert binding.last_request.qualified_subject.subject_reference == _ARTIFACT


def test_resume_binding_idempotent_on_retry() -> None:
    binding = _RecordingBindingProvider()
    execution = _RecordingExecutionPort()
    coordinator = _coordinator(binding, execution)
    request = _resume_request(_qualification())
    first = coordinator.resume(request)
    second = coordinator.resume(request)

    assert first.outcome is WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED
    assert second.outcome is WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED
    assert binding.bind_calls == 2
    assert binding.last_request is not None
    assert (
        derive_qualified_capability_binding_operation_id(
            resume_operation_id=request.resume_operation_id,
            qualified_subject_reference=first.provenance.qualified_subject_reference
            or "",
        )
        == binding.last_request.binding_operation_id
    )


def test_uca6c_resume_core_forbidden_provider_routing() -> None:
    paths = (
        "worker_qualified_capability_resume_coordinator.py",
        "worker_qualified_capability_resume_ports.py",
    )
    package = importlib.import_module("intergrax.autonomous_work")
    base = Path(package.__file__).parent
    forbidden = (
        "codecraft",
        "CodeCraft",
        "startswith",
        "marketplace",
        "get_ephemeral_registry",
        "ToolRegistry",
        "SkillRegistry",
        "AgentRegistry",
    )
    for name in paths:
        source = (base / name).read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token.lower() not in source, f"{name} contains forbidden {token}"


def test_uca6c_resume_core_no_execution_runtime_import() -> None:
    path = Path(
        importlib.import_module(
            "intergrax.autonomous_work.worker_qualified_capability_resume_coordinator",
        ).__file__
        or ""
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    joined = "\n".join(modules)
    assert "intergrax.runtime.execution" not in joined
    assert "CodeCraftOrchestrator" not in joined
