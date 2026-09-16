# © Artur Czarnecki. All rights reserved.

"""ME-10-R1 — consumer identity, delivery idempotency, tenant correlation integrity."""

from __future__ import annotations

import ast
import importlib
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.marketplace import (
    CapabilityDiscoveryTraceFacts,
    CapabilityHandoffConsumerError,
    CapabilityHandoffDeliveryAdmissionError,
    CapabilityHandoffDeliveryDisposition,
    CapabilityHandoffEnvelope,
    CapabilityHandoffIdentityConflictError,
    CapabilityMarketplaceExplicitSelection,
    MarketplaceQueryContext,
)
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerTarget,
    CapabilityHandoffDeliveryAdmission,
    CapabilityHandoffDeliveryAdmissionResult,
)
from intergrax.marketplace.handoff_traceability import (
    CapabilityHandoffDeliveryService,
    InMemoryCapabilityHandoffDeliveryAdmission,
    MarketplaceHandoffConsumerIdentityMismatchError,
)
pytestmark = pytest.mark.unit

_SOURCE = CapabilitySourceIdentity(
    source_id="official.me10r1",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _release(version_label: str = "1.0.0") -> CapabilityReleaseIdentity:
    return CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tools.r1",
            ),
        ),
        publisher="pub",
        version_label=version_label,
        content_digest="sha256:r1",
    )


def _trace(
    *,
    tenant_id: str | None = "tenant-a",
    discovery_correlation_id: str = "discovery-1",
) -> CapabilityDiscoveryTraceFacts:
    return CapabilityDiscoveryTraceFacts(
        discovery_correlation_id=discovery_correlation_id,
        marketplace_query_context=MarketplaceQueryContext(tenant_id=tenant_id),
        visible_candidate_count=1,
        governed_admissible_count=1,
    )


def _selection(release: CapabilityReleaseIdentity) -> CapabilityMarketplaceExplicitSelection:
    return CapabilityMarketplaceExplicitSelection(
        selection_id="selection-1",
        discovery_correlation_id="discovery-1",
        selected_release=release,
        selector_id="selector-1",
    )


def _envelope(
    *,
    handoff_id: str = "handoff-1",
    tenant_id: str | None = "tenant-a",
    consumer_id: str = "consumer-B",
    release: CapabilityReleaseIdentity | None = None,
) -> CapabilityHandoffEnvelope:
    release = release or _release()
    return CapabilityHandoffEnvelope(
        handoff_id=handoff_id,
        tenant_id=tenant_id,
        selected_release=release,
        discovery_correlation_id="discovery-1",
        selection_id="selection-1",
        consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
        downstream_consumer_id=consumer_id,
        discovery_trace=_trace(tenant_id=tenant_id),
        explicit_selection=_selection(release),
        recorded_at=datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc),
    )


class _RecordingConsumer:
    def __init__(self, consumer_id: str) -> None:
        self._consumer_id = consumer_id
        self.calls = 0

    @property
    def consumer_id(self) -> str:
        return self._consumer_id

    def consume(self, envelope: CapabilityHandoffEnvelope) -> None:
        self.calls += 1


class _RecordingTrace:
    def __init__(self) -> None:
        self.calls = 0

    def record_handoff(self, envelope: CapabilityHandoffEnvelope) -> bool:
        self.calls += 1
        return True


class _FailingAdmission:
    def reserve(self, envelope: CapabilityHandoffEnvelope) -> CapabilityHandoffDeliveryAdmissionResult:
        raise RuntimeError("admission store unavailable")

    def mark_delivered(self, handoff_id: str) -> None:
        raise RuntimeError("admission store unavailable")

    def mark_delivery_failed(self, handoff_id: str) -> None:
        raise RuntimeError("admission store unavailable")


class _CustomAdmission:
    """Structural provider — no platform subclass."""

    def __init__(self) -> None:
        self.reserve_calls = 0
        self._inner = InMemoryCapabilityHandoffDeliveryAdmission()

    def reserve(self, envelope: CapabilityHandoffEnvelope) -> CapabilityHandoffDeliveryAdmissionResult:
        self.reserve_calls += 1
        return self._inner.reserve(envelope)

    def mark_delivered(self, handoff_id: str) -> None:
        self._inner.mark_delivered(handoff_id)

    def mark_delivery_failed(self, handoff_id: str) -> None:
        self._inner.mark_delivery_failed(handoff_id)


def _delivery(
    consumer: _RecordingConsumer,
    *,
    trace: _RecordingTrace | None = None,
    admission: CapabilityHandoffDeliveryAdmission | None = None,
) -> CapabilityHandoffDeliveryService:
    return CapabilityHandoffDeliveryService(
        consumer=consumer,
        delivery_admission=admission or InMemoryCapabilityHandoffDeliveryAdmission(),
        trace_evidence_consumer=trace,
    )


def test_reproducer_consumer_identity_mismatch_rejected_with_zero_side_effects() -> None:
    consumer = _RecordingConsumer("consumer-B")
    trace = _RecordingTrace()
    admission = InMemoryCapabilityHandoffDeliveryAdmission()
    service = _delivery(consumer, trace=trace, admission=admission)
    envelope = _envelope(consumer_id="consumer-A")
    with pytest.raises(MarketplaceHandoffConsumerIdentityMismatchError):
        service.deliver(envelope)
    assert consumer.calls == 0
    assert trace.calls == 0
    assert admission.delivered_handoff_ids() == ()


def test_duplicate_delivery_without_trace_provider_invokes_consumer_once() -> None:
    consumer = _RecordingConsumer("consumer-B")
    service = _delivery(consumer, trace=None)
    envelope = _envelope(consumer_id="consumer-B")
    first = service.deliver(envelope)
    second = service.deliver(envelope)
    assert first.disposition is CapabilityHandoffDeliveryDisposition.DELIVERED
    assert second.disposition is CapabilityHandoffDeliveryDisposition.DUPLICATE_SKIPPED
    assert consumer.calls == 1


def test_reproducer_tenant_none_vs_discovery_tenant_rejected_at_envelope_contract() -> None:
    release = _release()
    with pytest.raises(ValueError, match="tenant_id must match marketplace_query_context.tenant_id"):
        CapabilityHandoffEnvelope(
            handoff_id="handoff-1",
            tenant_id=None,
            selected_release=release,
            discovery_correlation_id="discovery-1",
            selection_id="selection-1",
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            downstream_consumer_id="consumer-B",
            discovery_trace=_trace(tenant_id="tenant-a"),
            explicit_selection=_selection(release),
            recorded_at=datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc),
        )


def test_tenant_none_none_accepted() -> None:
    release = _release()
    envelope = CapabilityHandoffEnvelope(
        handoff_id="handoff-1",
        tenant_id=None,
        selected_release=release,
        discovery_correlation_id="discovery-1",
        selection_id="selection-1",
        consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
        downstream_consumer_id="consumer-B",
        discovery_trace=_trace(tenant_id=None),
        explicit_selection=_selection(release),
        recorded_at=datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc),
    )
    consumer = _RecordingConsumer("consumer-B")
    result = _delivery(consumer).deliver(envelope)
    assert result.disposition is CapabilityHandoffDeliveryDisposition.DELIVERED


def test_handoff_identity_conflict_on_same_id_different_payload() -> None:
    consumer = _RecordingConsumer("consumer-B")
    service = _delivery(consumer)
    envelope_a = _envelope(handoff_id="h-conflict", consumer_id="consumer-B")
    service.deliver(envelope_a)
    envelope_b = _envelope(handoff_id="h-conflict", consumer_id="consumer-B", release=_release("2.0.0"))
    with pytest.raises(CapabilityHandoffIdentityConflictError):
        service.deliver(envelope_b)
    assert consumer.calls == 1


def test_admission_provider_failure_fails_closed_before_consumer() -> None:
    consumer = _RecordingConsumer("consumer-B")
    service = _delivery(consumer, admission=_FailingAdmission())
    with pytest.raises(CapabilityHandoffDeliveryAdmissionError, match="admission store unavailable"):
        service.deliver(_envelope(consumer_id="consumer-B"))
    assert consumer.calls == 0


def test_trace_provider_failure_after_successful_delivery_still_delivered() -> None:
    class _FailingTrace:
        def record_handoff(self, envelope: CapabilityHandoffEnvelope) -> bool:
            raise RuntimeError("trace backend down")

    consumer = _RecordingConsumer("consumer-B")
    service = _delivery(consumer, trace=_FailingTrace())
    result = service.deliver(_envelope(consumer_id="consumer-B"))
    assert result.disposition is CapabilityHandoffDeliveryDisposition.DELIVERED
    assert consumer.calls == 1


def test_with_trace_provider_idempotency_matches_without_trace() -> None:
    consumer = _RecordingConsumer("consumer-B")
    trace = _RecordingTrace()
    service = _delivery(consumer, trace=trace)
    envelope = _envelope(consumer_id="consumer-B")
    service.deliver(envelope)
    service.deliver(envelope)
    assert consumer.calls == 1
    assert trace.calls == 1


def test_structural_custom_delivery_admission_provider() -> None:
    consumer = _RecordingConsumer("consumer-B")
    custom = _CustomAdmission()
    service = _delivery(consumer, admission=custom)
    envelope = _envelope(consumer_id="consumer-B")
    service.deliver(envelope)
    service.deliver(envelope)
    assert custom.reserve_calls == 2
    assert consumer.calls == 1


def test_concurrent_duplicate_delivery_invokes_consumer_at_most_once() -> None:
    consumer = _RecordingConsumer("consumer-B")
    service = _delivery(consumer)
    envelope = _envelope(consumer_id="consumer-B", handoff_id="handoff-concurrent")
    barrier = threading.Barrier(2)
    results: list[CapabilityHandoffDeliveryDisposition] = []
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            barrier.wait()
            result = service.deliver(envelope)
            results.append(result.disposition)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert not errors
    assert consumer.calls == 1
    assert CapabilityHandoffDeliveryDisposition.DELIVERED in results
    assert (
        CapabilityHandoffDeliveryDisposition.DUPLICATE_SKIPPED in results
        or CapabilityHandoffDeliveryDisposition.IN_PROGRESS_SKIPPED in results
    )


def test_orchestrator_does_not_accept_arbitrary_downstream_consumer_id() -> None:
    package = importlib.import_module("intergrax.marketplace.handoff_traceability.orchestrator")
    root = Path(package.__file__).resolve()
    tree = ast.parse(root.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "execute_explicit_selection_handoff":
            continue
        arg_names = [arg.arg for arg in node.args.args]
        assert "downstream_consumer_id" not in arg_names
        return
    raise AssertionError("execute_explicit_selection_handoff not found")


def test_delivery_idempotency_not_gated_on_trace_evidence_only() -> None:
    package = importlib.import_module("intergrax.marketplace.handoff_traceability.delivery")
    src = Path(package.__file__).read_text(encoding="utf-8")
    assert "delivery_admission" in src
    assert "ALREADY_DELIVERED_IDENTICAL" in src
    assert "duplicate = not recorded" not in src


def test_consumer_failure_after_admission_propagates() -> None:
    class _FailingConsumer(_RecordingConsumer):
        def consume(self, envelope: CapabilityHandoffEnvelope) -> None:
            raise CapabilityHandoffConsumerError("downstream failed")

    consumer = _FailingConsumer("consumer-B")
    service = _delivery(consumer)
    with pytest.raises(CapabilityHandoffConsumerError, match="downstream failed"):
        service.deliver(_envelope(consumer_id="consumer-B"))
