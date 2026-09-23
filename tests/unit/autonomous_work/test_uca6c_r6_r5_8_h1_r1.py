# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.8-H1-R1 — Direct reuse provenance and worker fulfillment outcome semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.autonomous_work.worker_recovery_capability_fulfillment_service import (
    WorkerRecoveryCapabilityFulfillmentService,
)
from intergrax.contracts.autonomous_work.obstacle_recovery import RecoveryStrategy
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentDisposition,
    WorkerCapabilityFulfillmentRequest,
    WorkerCapabilityFulfillmentResult,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryProvenance,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionDisposition,
    WorkerQualifiedCapabilityExecutionResult,
)
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    WorkerRecoveryOrchestrationDisposition,
)
from tests.unit.autonomous_work.test_uca6c_worker_qualified_capability_resume import (
    _EXEC_ID,
)
from tests.unit.autonomous_work.test_uca6c_r6_r5_8_worker_consumer_e2e import (
    _fulfillment_request,
)
from tests.unit.autonomous_work.test_worker_recovery_orchestration import (
    _decision,
    _harness,
    _orchestration_request,
)
from tests.unit.autonomous_work.uca6b_test_support import build_recording_acquisition

pytestmark = [pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[3]
_DIRECT_REUSE_SERVICE = (
    _REPO
    / "intergrax"
    / "autonomous_work"
    / "worker_capability_direct_reuse_fulfillment_service.py"
)
_NOW = datetime(2026, 3, 20, 12, 0, tzinfo=UTC)


def test_direct_reuse_service_has_no_synthetic_acquisition_qualification_fields() -> None:
    text = _DIRECT_REUSE_SERVICE.read_text(encoding="utf-8")
    assert "qualification_request_id=" not in text
    assert "acquisition_request_id=" not in text
    assert "host-available:discovery:" not in text


@dataclass
class _SemanticFulfillmentPort:
    result: WorkerCapabilityFulfillmentResult
    calls: int = 0

    def fulfill(self, request: WorkerCapabilityFulfillmentRequest):
        self.calls += 1
        return self.result


@pytest.mark.asyncio
async def test_recovery_consumes_execution_dispatched_not_blind_escalation() -> None:
    provenance = WorkerCapabilityRecoveryProvenance(
        worker_need_id="need",
        canonical_need_id="canonical",
        discovery_correlation_id="corr",
        discovery_completion_outcome="direct_reuse",
        evidence_refs=(),
    )
    fulfillment_result = WorkerCapabilityFulfillmentResult(
        disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED,
        provenance=provenance,
        execution_result=WorkerQualifiedCapabilityExecutionResult(
            disposition=WorkerQualifiedCapabilityExecutionDisposition.DISPATCHED,
            execution_request_id="worker-qualified-capability-execution:reuse:bind",
            execution_id=_EXEC_ID,
        ),
        decided_at=_NOW,
    )
    recording = _SemanticFulfillmentPort(result=fulfillment_result)
    service, _ = _harness()
    service._recovery_capability_fulfillment = WorkerRecoveryCapabilityFulfillmentService(
        fulfillment=recording,
    )
    service._recovery_capability_fulfillment_request_builder = __import__(
        "tests.unit.autonomous_work.test_uca6c_r6_r5_8_worker_consumer_e2e",
        fromlist=["_StaticFulfillmentRequestBuilder"],
    )._StaticFulfillmentRequestBuilder(_fulfillment_request())
    acquisition = build_recording_acquisition()
    service._capability_port = acquisition
    orch = await service.orchestrate(
        _orchestration_request(
            decision=_decision(strategy=RecoveryStrategy.ACQUIRE_CAPABILITY),
        ),
    )
    assert recording.calls == 1
    assert acquisition.strategy.calls == 0
    assert (
        orch.disposition
        is WorkerRecoveryOrchestrationDisposition.ATTEMPT_DISPATCHED
    )
    assert orch.episode.last_execution_id == _EXEC_ID
    assert orch.episode.terminal_reason != "capability_fulfillment_deferred"


@pytest.mark.asyncio
async def test_recovery_fulfillment_missing_semantic_result_fail_closed() -> None:
    from intergrax.autonomous_work.recovery_orchestration_ports import (
        PortAvailabilityDisposition,
        WorkerRecoveryCapabilityFulfillmentResult,
    )

    class _MissingSemanticFulfillmentPort:
        def fulfill_recovery_capability(self, handoff):
            return WorkerRecoveryCapabilityFulfillmentResult(
                disposition=PortAvailabilityDisposition.AVAILABLE,
                fulfillment_result=None,
            )

    service, _ = _harness()
    service._recovery_capability_fulfillment = _MissingSemanticFulfillmentPort()
    service._recovery_capability_fulfillment_request_builder = __import__(
        "tests.unit.autonomous_work.test_uca6c_r6_r5_8_worker_consumer_e2e",
        fromlist=["_StaticFulfillmentRequestBuilder"],
    )._StaticFulfillmentRequestBuilder(_fulfillment_request())
    orch = await service.orchestrate(
        _orchestration_request(
            decision=_decision(strategy=RecoveryStrategy.ACQUIRE_CAPABILITY),
        ),
    )
    assert orch.disposition is WorkerRecoveryOrchestrationDisposition.ESCALATED
    assert orch.episode.terminal_reason == "capability_fulfillment_missing_result"


@pytest.mark.asyncio
async def test_recovery_capability_gap_maps_to_escalated_not_deferred() -> None:
    provenance = WorkerCapabilityRecoveryProvenance(
        worker_need_id="need",
        canonical_need_id="canonical",
        discovery_correlation_id="corr",
        discovery_completion_outcome="missing",
        evidence_refs=(),
    )
    fulfillment_result = WorkerCapabilityFulfillmentResult(
        disposition=WorkerCapabilityFulfillmentDisposition.CAPABILITY_GAP,
        provenance=provenance,
        decided_at=_NOW,
    )
    recording = _SemanticFulfillmentPort(result=fulfillment_result)
    service, _ = _harness()
    service._recovery_capability_fulfillment = WorkerRecoveryCapabilityFulfillmentService(
        fulfillment=recording,
    )
    service._recovery_capability_fulfillment_request_builder = __import__(
        "tests.unit.autonomous_work.test_uca6c_r6_r5_8_worker_consumer_e2e",
        fromlist=["_StaticFulfillmentRequestBuilder"],
    )._StaticFulfillmentRequestBuilder(_fulfillment_request())
    orch = await service.orchestrate(
        _orchestration_request(
            decision=_decision(strategy=RecoveryStrategy.ACQUIRE_CAPABILITY),
        ),
    )
    assert orch.disposition is WorkerRecoveryOrchestrationDisposition.ESCALATED
    assert orch.episode.terminal_reason == "capability_fulfillment_capability_gap"
