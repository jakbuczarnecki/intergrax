# © Artur Czarnecki. All rights reserved.

"""ME-10-R2 — handoff admission lifecycle, retry safety, outcome uncertainty."""

from __future__ import annotations

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
    CapabilityHandoffDeliveryAdmissionVerdict,
    CapabilityHandoffDeliveryLifecycleTransitionError,
    CapabilityHandoffDeliveryOutcomeUncertainError,
)
from intergrax.marketplace.handoff_traceability import (
    CapabilityHandoffDeliveryService,
    InMemoryCapabilityHandoffDeliveryAdmission,
)

pytestmark = pytest.mark.unit

_SOURCE = CapabilitySourceIdentity(
    source_id="official.me10r2",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _release(version_label: str = "1.0.0") -> CapabilityReleaseIdentity:
    return CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tools.r2",
            ),
        ),
        publisher="pub",
        version_label=version_label,
        content_digest="sha256:r2",
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
        self._fail_until: int = 0

    def set_fail_first_n(self, n: int) -> None:
        self._fail_until = n

    @property
    def consumer_id(self) -> str:
        return self._consumer_id

    def consume(self, envelope: CapabilityHandoffEnvelope) -> None:
        self.calls += 1
        if self.calls <= self._fail_until:
            raise CapabilityHandoffConsumerError("downstream failed")


def _delivery(
    consumer: _RecordingConsumer,
    *,
    admission: CapabilityHandoffDeliveryAdmission | None = None,
) -> CapabilityHandoffDeliveryService:
    return CapabilityHandoffDeliveryService(
        consumer=consumer,
        delivery_admission=admission or InMemoryCapabilityHandoffDeliveryAdmission(),
    )


def test_reproducer_consumer_fail_then_retry_not_duplicate_skipped() -> None:
    """Phase A — must fail before R2 lifecycle fix."""
    consumer = _RecordingConsumer("consumer-B")
    consumer.set_fail_first_n(1)
    service = _delivery(consumer)
    envelope = _envelope(consumer_id="consumer-B", handoff_id="handoff-retry-r2")
    with pytest.raises(CapabilityHandoffConsumerError, match="downstream failed"):
        service.deliver(envelope)
    second = service.deliver(envelope)
    assert second.disposition is CapabilityHandoffDeliveryDisposition.DELIVERED
    assert consumer.calls == 2
    third = service.deliver(envelope)
    assert third.disposition is CapabilityHandoffDeliveryDisposition.DUPLICATE_SKIPPED
    assert consumer.calls == 2


def test_conflict_after_failed_attempt_keeps_original_binding() -> None:
    consumer = _RecordingConsumer("consumer-B")
    consumer.set_fail_first_n(1)
    service = _delivery(consumer)
    envelope_a = _envelope(handoff_id="h-x", consumer_id="consumer-B")
    with pytest.raises(CapabilityHandoffConsumerError):
        service.deliver(envelope_a)
    envelope_b = _envelope(
        handoff_id="h-x",
        consumer_id="consumer-B",
        release=_release("2.0.0"),
    )
    with pytest.raises(CapabilityHandoffIdentityConflictError):
        service.deliver(envelope_b)
    assert consumer.calls == 1


def test_conflict_after_successful_delivery() -> None:
    consumer = _RecordingConsumer("consumer-B")
    service = _delivery(consumer)
    envelope_a = _envelope(handoff_id="h-ok", consumer_id="consumer-B")
    service.deliver(envelope_a)
    envelope_b = _envelope(
        handoff_id="h-ok",
        consumer_id="consumer-B",
        release=_release("9.0.0"),
    )
    with pytest.raises(CapabilityHandoffIdentityConflictError):
        service.deliver(envelope_b)
    assert consumer.calls == 1


class _FailingReserveAdmission:
    def reserve(self, envelope: CapabilityHandoffEnvelope) -> CapabilityHandoffDeliveryAdmissionResult:
        raise RuntimeError("reserve unavailable")

    def mark_delivered(self, handoff_id: str) -> None:
        return

    def mark_delivery_failed(self, handoff_id: str) -> None:
        return


def test_reserve_failure_does_not_invoke_consumer() -> None:
    consumer = _RecordingConsumer("consumer-B")
    service = _delivery(consumer, admission=_FailingReserveAdmission())
    with pytest.raises(CapabilityHandoffDeliveryAdmissionError, match="reserve unavailable"):
        service.deliver(_envelope(consumer_id="consumer-B"))
    assert consumer.calls == 0


class _FailingReleaseAdmission:
    def __init__(self) -> None:
        self._inner = InMemoryCapabilityHandoffDeliveryAdmission()

    def reserve(self, envelope: CapabilityHandoffEnvelope) -> CapabilityHandoffDeliveryAdmissionResult:
        return self._inner.reserve(envelope)

    def mark_delivered(self, handoff_id: str) -> None:
        self._inner.mark_delivered(handoff_id)

    def mark_delivery_failed(self, handoff_id: str) -> None:
        raise RuntimeError("release store down")


def test_consumer_failure_when_release_fails_raises_lifecycle_transition_error() -> None:
    consumer = _RecordingConsumer("consumer-B")
    consumer.set_fail_first_n(1)
    service = _delivery(consumer, admission=_FailingReleaseAdmission())
    with pytest.raises(CapabilityHandoffDeliveryLifecycleTransitionError, match="release store down"):
        service.deliver(_envelope(consumer_id="consumer-B"))
    assert consumer.calls == 1


class _FailingMarkDeliveredAdmission:
    def __init__(self) -> None:
        self._inner = InMemoryCapabilityHandoffDeliveryAdmission()

    def reserve(self, envelope: CapabilityHandoffEnvelope) -> CapabilityHandoffDeliveryAdmissionResult:
        return self._inner.reserve(envelope)

    def mark_delivered(self, handoff_id: str) -> None:
        raise RuntimeError("commit failed")

    def mark_delivery_failed(self, handoff_id: str) -> None:
        self._inner.mark_delivery_failed(handoff_id)


def test_mark_delivered_failure_after_consumer_success_is_outcome_uncertain() -> None:
    consumer = _RecordingConsumer("consumer-B")
    service = _delivery(consumer, admission=_FailingMarkDeliveredAdmission())
    with pytest.raises(CapabilityHandoffDeliveryOutcomeUncertainError, match="commit failed"):
        service.deliver(_envelope(consumer_id="consumer-B"))
    assert consumer.calls == 1


def test_duplicate_after_delivered_skips_consumer() -> None:
    consumer = _RecordingConsumer("consumer-B")
    service = _delivery(consumer)
    envelope = _envelope(consumer_id="consumer-B")
    assert service.deliver(envelope).disposition is CapabilityHandoffDeliveryDisposition.DELIVERED
    assert service.deliver(envelope).disposition is CapabilityHandoffDeliveryDisposition.DUPLICATE_SKIPPED
    assert consumer.calls == 1


class _GatedSlowConsumer:
    def __init__(self, consumer_id: str) -> None:
        self._consumer_id = consumer_id
        self.calls = 0
        self.entered = threading.Event()
        self.proceed = threading.Event()

    @property
    def consumer_id(self) -> str:
        return self._consumer_id

    def consume(self, envelope: CapabilityHandoffEnvelope) -> None:
        self.calls += 1
        self.entered.set()
        assert self.proceed.wait(timeout=5.0)


def test_duplicate_while_in_progress_returns_in_progress_skipped() -> None:
    consumer = _GatedSlowConsumer("consumer-B")
    service = _delivery(consumer)
    envelope = _envelope(consumer_id="consumer-B", handoff_id="handoff-in-flight")
    result_holder: list[CapabilityHandoffDeliveryDisposition] = []
    errors: list[BaseException] = []

    def _first() -> None:
        try:
            result_holder.append(service.deliver(envelope).disposition)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=_first)
    thread.start()
    assert consumer.entered.wait(timeout=5.0)
    second = service.deliver(envelope)
    assert second.disposition is CapabilityHandoffDeliveryDisposition.IN_PROGRESS_SKIPPED
    consumer.proceed.set()
    thread.join(timeout=5.0)
    assert not errors
    assert CapabilityHandoffDeliveryDisposition.DELIVERED in result_holder
    assert consumer.calls == 1


def test_concurrent_different_handoff_ids_not_globally_blocked() -> None:
    barrier = threading.Barrier(2)
    consumer = _RecordingConsumer("consumer-B")
    service = _delivery(consumer)
    results: list[str] = []

    def _run(handoff_id: str) -> None:
        barrier.wait()
        disposition = service.deliver(
            _envelope(consumer_id="consumer-B", handoff_id=handoff_id),
        ).disposition
        results.append(f"{handoff_id}:{disposition.value}")

    threads = [
        threading.Thread(target=_run, args=("handoff-a",)),
        threading.Thread(target=_run, args=("handoff-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert consumer.calls == 2
    assert len(results) == 2


class _RecordingTrace:
    def __init__(self) -> None:
        self.calls = 0

    def record_handoff(self, envelope: CapabilityHandoffEnvelope) -> bool:
        self.calls += 1
        return True


def test_trace_failure_does_not_change_delivered_disposition() -> None:
    class _FailingTrace:
        def record_handoff(self, envelope: CapabilityHandoffEnvelope) -> bool:
            raise RuntimeError("trace down")

    consumer = _RecordingConsumer("consumer-B")
    service = CapabilityHandoffDeliveryService(
        consumer=consumer,
        delivery_admission=InMemoryCapabilityHandoffDeliveryAdmission(),
        trace_evidence_consumer=_FailingTrace(),
    )
    assert service.deliver(_envelope(consumer_id="consumer-B")).disposition is (
        CapabilityHandoffDeliveryDisposition.DELIVERED
    )


def test_lifecycle_semantics_without_trace_provider() -> None:
    consumer = _RecordingConsumer("consumer-B")
    consumer.set_fail_first_n(1)
    service = _delivery(consumer, admission=None)
    envelope = _envelope(consumer_id="consumer-B")
    with pytest.raises(CapabilityHandoffConsumerError):
        service.deliver(envelope)
    assert service.deliver(envelope).disposition is CapabilityHandoffDeliveryDisposition.DELIVERED


def test_lifecycle_semantics_with_trace_provider() -> None:
    consumer = _RecordingConsumer("consumer-B")
    trace = _RecordingTrace()
    service = CapabilityHandoffDeliveryService(
        consumer=consumer,
        delivery_admission=InMemoryCapabilityHandoffDeliveryAdmission(),
        trace_evidence_consumer=trace,
    )
    envelope = _envelope(consumer_id="consumer-B")
    service.deliver(envelope)
    service.deliver(envelope)
    assert consumer.calls == 1
    assert trace.calls == 1


class _StandaloneLifecycleAdmission:
    """Custom structural provider — does not subclass or delegate to reference impl."""

    def __init__(self) -> None:
        self._delivered: dict[str, CapabilityHandoffEnvelope] = {}
        self._in_progress: dict[str, CapabilityHandoffEnvelope] = {}
        self._retryable: dict[str, CapabilityHandoffEnvelope] = {}
        self.reserve_calls = 0

    def reserve(self, envelope: CapabilityHandoffEnvelope) -> CapabilityHandoffDeliveryAdmissionResult:
        self.reserve_calls += 1
        handoff_id = envelope.handoff_id
        delivered = self._delivered.get(handoff_id)
        if delivered is not None:
            if delivered != envelope:
                raise CapabilityHandoffIdentityConflictError("clash")
            return CapabilityHandoffDeliveryAdmissionResult(
                verdict=CapabilityHandoffDeliveryAdmissionVerdict.ALREADY_DELIVERED_IDENTICAL,
                handoff_id=handoff_id,
            )
        bound = self._retryable.get(handoff_id)
        if bound is not None:
            if bound != envelope:
                raise CapabilityHandoffIdentityConflictError("clash")
            del self._retryable[handoff_id]
            self._in_progress[handoff_id] = envelope
            return CapabilityHandoffDeliveryAdmissionResult(
                verdict=CapabilityHandoffDeliveryAdmissionVerdict.RESERVED_NEW,
                handoff_id=handoff_id,
            )
        active = self._in_progress.get(handoff_id)
        if active is None:
            self._in_progress[handoff_id] = envelope
            return CapabilityHandoffDeliveryAdmissionResult(
                verdict=CapabilityHandoffDeliveryAdmissionVerdict.RESERVED_NEW,
                handoff_id=handoff_id,
            )
        if active != envelope:
            raise CapabilityHandoffIdentityConflictError("clash")
        return CapabilityHandoffDeliveryAdmissionResult(
            verdict=CapabilityHandoffDeliveryAdmissionVerdict.IN_PROGRESS_IDENTICAL,
            handoff_id=handoff_id,
        )

    def mark_delivered(self, handoff_id: str) -> None:
        envelope = self._in_progress.pop(handoff_id, None)
        if envelope is None:
            raise CapabilityHandoffDeliveryAdmissionError("missing reservation")
        self._delivered[handoff_id] = envelope

    def mark_delivery_failed(self, handoff_id: str) -> None:
        envelope = self._in_progress.pop(handoff_id, None)
        if envelope is None:
            raise CapabilityHandoffDeliveryAdmissionError("missing reservation")
        self._retryable[handoff_id] = envelope


def test_custom_lifecycle_provider_is_pluggable() -> None:
    consumer = _RecordingConsumer("consumer-B")
    custom = _StandaloneLifecycleAdmission()
    service = _delivery(consumer, admission=custom)
    envelope = _envelope(consumer_id="consumer-B")
    service.deliver(envelope)
    service.deliver(envelope)
    assert custom.reserve_calls == 2
    assert consumer.calls == 1


def test_delivery_lifecycle_authority_separate_from_trace_evidence() -> None:
    delivery = importlib.import_module("intergrax.marketplace.handoff_traceability.delivery")
    src = Path(delivery.__file__).read_text(encoding="utf-8")
    assert "trace_evidence_consumer" in src
    assert "mark_delivered" in src
    assert "reserve(" in src
    assert "_delivery_lock" not in src
    assert "record_handoff" not in src.split("def deliver")[1].split("def _record_trace_observation")[0]


def test_contracts_module_does_not_import_reference_admission_provider() -> None:
    contracts = importlib.import_module("intergrax.contracts.marketplace.handoff_traceability")
    path = Path(contracts.__file__).read_text(encoding="utf-8")
    assert "InMemoryCapabilityHandoffDeliveryAdmission" not in path
    assert "intergrax.marketplace" not in path
