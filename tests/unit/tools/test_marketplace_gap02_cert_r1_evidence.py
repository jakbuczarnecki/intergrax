# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-CERT-R1 — certification evidence closure (tests/docs only)."""

from __future__ import annotations

import pytest

from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentDisposition,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletionOutcome,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingOutcome,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.contracts.tools.marketplace_handoff_reference import (
    derive_marketplace_gap_tool_handoff_id,
)
from tests.unit.tools.support.gap02_cert_harness import (
    CachedRecoveryPort,
    Gap02CertHarness,
    RecordingInvoker,
)

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-gap02-cert-a"
_TENANT_B = "tenant-gap02-cert-b"


def test_r1_c5_one_shared_store_same_acquisition_cross_tenant_denied() -> None:
    shared_recovery = "recovery:gap02:cert:r1:shared"
    shared_store = InMemoryDocumentStore()
    ha = Gap02CertHarness.build(_TENANT_A, store=shared_store)
    hb = Gap02CertHarness.build(_TENANT_B, store=shared_store)
    _, trace_a = ha.run_true_gap_recovery(shared_recovery)
    _, trace_b = hb.run_true_gap_recovery(shared_recovery)
    assert trace_a.acquisition_request_id == trace_b.acquisition_request_id
    assert trace_a.handoff_id != trace_b.handoff_id
    assert ha.stage_repo.get(tenant_id=_TENANT_A, handoff_id=trace_a.handoff_id) is not None
    assert hb.stage_repo.get(tenant_id=_TENANT_B, handoff_id=trace_b.handoff_id) is not None
    assert ha.stage_repo.get(tenant_id=_TENANT_A, handoff_id=trace_b.handoff_id) is None
    assert hb.stage_repo.get(tenant_id=_TENANT_B, handoff_id=trace_a.handoff_id) is None


def test_r1_c7_restart_safe_reconstruction_after_durable_intent_before_ee() -> None:
    harness = Gap02CertHarness.build(_TENANT_A)
    recovery_id = "recovery:gap02:cert:r1:c7"
    outcome, _ = harness.run_true_gap_recovery(recovery_id)
    acquire_at_intent = harness.gap_port.acquire_calls
    qual_at_intent = harness.qualification_adapter.calls
    exec_id = harness.execution_request_id_for_recovery(
        outcome,
        recovery_decision_id=recovery_id,
    )
    harness.fulfill_through_durable_intent_before_ee(
        outcome,
        recovery_decision_id=recovery_id,
    )
    assert harness.intent_repo.get(execution_request_id=exec_id) is not None
    restarted = harness.reconstruct_tool_domain()
    restarted.gap_port.acquire_calls = acquire_at_intent
    invoker = RecordingInvoker()
    result = restarted.build_fulfillment_coordinator(
        invoker,
        recovery=CachedRecoveryPort(outcome),
    ).fulfill(restarted.fulfillment_request(recovery_id))
    assert result.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED
    assert result.provenance.execution_request_id == exec_id
    assert restarted.gap_port.acquire_calls == acquire_at_intent
    assert restarted.qualification_adapter.calls == 0
    assert qual_at_intent == 1
    assert invoker.calls == 1


def test_r1_c2_activation_timeline_t0_through_t10() -> None:
    harness = Gap02CertHarness.build(_TENANT_A)
    recovery_id = "recovery:gap02:cert:r1:c2"
    checkpoints: list[int] = []

    def record() -> None:
        checkpoints.append(harness.materializer.physical_activations)

    harness.assert_no_tool_side_effects()
    record()
    completion = harness.complete_true_gap_discovery(recovery_id)
    assert completion.outcome is DiscoveryCompletionOutcome.MISSING_CAPABILITY
    record()
    record()
    acquisition = harness.acquire_marketplace_gap(recovery_id, completion)
    assert acquisition.outcome is CapabilityAcquisitionOutcome.SUCCEEDED
    handoff_id = derive_marketplace_gap_tool_handoff_id(
        tenant_id=_TENANT_A,
        operation_id=acquisition.request_id,
    )
    assert harness.stage_repo.get(tenant_id=_TENANT_A, handoff_id=handoff_id) is not None
    record()
    record()
    qualification = harness.qualify_acquisition(
        recovery_id,
        acquisition,
        completion=completion,
    )
    assert qualification.outcome is CapabilityQualificationOutcome.QUALIFIED
    record()
    outcome = harness.recovery_outcome_from_staged(
        recovery_decision_id=recovery_id,
        completion=completion,
        acquisition_result=acquisition,
        qualification_result=qualification,
    )
    binding = harness.bind_qualified_capability(outcome, recovery_decision_id=recovery_id)
    assert binding.outcome is QualifiedCapabilityBindingOutcome.BOUND
    record()
    harness.fulfill_through_durable_intent_before_ee(
        outcome,
        recovery_decision_id=recovery_id,
    )
    record()
    record()
    invoker = RecordingInvoker()
    dispatch = harness.build_fulfillment_coordinator(
        invoker,
        recovery=CachedRecoveryPort(outcome),
    ).fulfill(harness.fulfillment_request(recovery_id))
    assert dispatch.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED
    record()
    record()
    assert checkpoints == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert invoker.calls == 1
