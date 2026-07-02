# © Artur Czarnecki. All rights reserved.

"""TOKEN-OBS-1B: HOS domain-signal adapter for token optimization signals."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.memory.summary_compressor import optimize_memory_summary
from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_catalog import EventCategory
from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.token_optimization.contracts import (
    CompressionLevel,
    CompressionReceiptRef,
    TokenOptimizationAttribution,
    TokenOptimizationDecision,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationResult,
)
from intergrax.runtime.token_optimization.domain_events import (
    TOKEN_OPTIMIZATION_SIGNAL_EVENT_KIND,
    TOKEN_OPTIMIZATION_SIGNAL_PAYLOAD_SCHEMA_ID,
    TokenOptimizationSignalPayloadV1,
    emit_token_optimization_domain_signal,
    register_token_optimization_domain_signal,
    token_optimization_signal_to_payload,
)
from intergrax.runtime.token_optimization.regression import (
    default_token_counter,
    run_token_regression_benchmarks,
)
from intergrax.runtime.token_optimization.signals import (
    TokenOptimizationSignal,
    TokenOptimizationSignalType,
    build_token_optimization_signal,
    build_token_regression_signal,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_DOMAIN_EVENTS_MODULE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "token_optimization"
    / "domain_events.py"
)
_COMPRESSIBLE_SUMMARY = (
    "  Session   summary   for   user   preferences.\n\n\n"
    "User   prefers   concise   answers.\n\n\n\n"
    "Next   step:   review   docs.  "
)
_SECRET_CONTENT = "super-secret-user-memory-content-should-never-appear"


@pytest.fixture(autouse=True)
def _register_token_optimization_domain_kind() -> None:
    clear_event_kind_registry()
    register_token_optimization_domain_signal()
    yield
    clear_event_kind_registry()


def _enabled_policy() -> TokenOptimizationPolicy:
    return TokenOptimizationPolicy(
        enabled=True,
        profile=TokenOptimizationProfile.CONSERVATIVE,
        compression_level=CompressionLevel.LIGHT,
        allow_lossy=False,
        require_validation=True,
        fallback_on_validation_failure=True,
        emit_receipts=True,
    )


def _optimization_signal() -> TokenOptimizationSignal:
    outcome = optimize_memory_summary(
        _COMPRESSIBLE_SUMMARY,
        token_policy=_enabled_policy(),
        token_counter=default_token_counter,
        attribution=TokenOptimizationAttribution(
            run_id="run-memory-1",
            step_id="step-memory-1",
            tenant_id="tenant-a",
        ),
    )
    return build_token_optimization_signal(outcome)


def _malicious_receipt_ref() -> CompressionReceiptRef:
    return CompressionReceiptRef(
        receipt_id="receipt-malicious-1",
        run_id="run-ref-1",
        step_id="step-ref-1",
        strategy_id="strategy-conservative",
        original_hash="orig-hash-abc",
        optimized_hash="opt-hash-xyz",
        metadata={
            "run_id": "run-ref-meta",
            "fixture_id": "fixture-42",
            "category": "memory_summary",
            "strategy_id": "strategy-conservative",
            "original_content": _SECRET_CONTENT,
            "optimized_content": "optimized secret body",
            "prompt": "secret prompt",
            "context": {"fragments": ["secret"]},
            "evidence": "secret evidence",
            "payload": {"nested": "secret"},
            "unsafe_custom": "drop-me",
        },
    )


def test_payload_built_from_token_optimization_signal() -> None:
    signal = _optimization_signal()
    payload = token_optimization_signal_to_payload(signal)

    assert isinstance(payload, TokenOptimizationSignalPayloadV1)
    assert payload.signal_id == signal.signal_id
    assert payload.signal_type == TokenOptimizationSignalType.OPTIMIZATION_OUTCOME.value


def test_payload_includes_token_savings_fields() -> None:
    signal = _optimization_signal()
    payload = token_optimization_signal_to_payload(signal)

    assert payload.baseline_tokens is not None and payload.baseline_tokens > 0
    assert payload.optimized_tokens is not None
    assert payload.saved_tokens is not None
    assert payload.saved_ratio is not None
    if payload.baseline_tokens > 0:
        assert payload.saved_ratio == pytest.approx(
            payload.saved_tokens / payload.baseline_tokens
        )


def test_payload_includes_validation_and_fallback_fields() -> None:
    signal = _optimization_signal()
    payload = token_optimization_signal_to_payload(signal)

    assert payload.validation_status in {"passed", "not_applicable", "failed", "skipped"}
    assert isinstance(payload.fallback_status, bool)


def test_payload_includes_safe_attribution_fields() -> None:
    signal = _optimization_signal()
    payload = token_optimization_signal_to_payload(signal)

    assert payload.run_id == "run-memory-1"
    assert payload.step_id == "step-memory-1"
    assert payload.tenant_id == "tenant-a"


def test_payload_includes_receipt_reference_scalar_fields_only() -> None:
    signal = _optimization_signal()
    payload = token_optimization_signal_to_payload(signal)

    assert signal.receipt_ref is not None
    assert payload.receipt_id == signal.receipt_id
    assert payload.receipt_run_id == signal.receipt_ref.run_id
    assert payload.receipt_step_id == signal.receipt_ref.step_id
    assert payload.receipt_strategy_id == signal.receipt_ref.strategy_id
    assert payload.receipt_original_hash == signal.receipt_ref.original_hash
    assert payload.receipt_optimized_hash == signal.receipt_ref.optimized_hash
    envelope = payload.to_envelope()
    assert "receipt_ref" not in envelope["data"]


def test_payload_metadata_is_sanitized() -> None:
    signal = TokenOptimizationSignal(
        signal_id="signal-meta-1",
        signal_type=TokenOptimizationSignalType.REGRESSION_RESULT,
        metadata={
            "run_id": "run-safe",
            "content": _SECRET_CONTENT,
            "prompt": "secret prompt",
            "profile": "conservative",
            "unsafe_custom": "drop-me",
        },
    )
    payload = token_optimization_signal_to_payload(signal)

    assert payload.metadata == {"run_id": "run-safe", "profile": "conservative"}
    assert _SECRET_CONTENT not in str(payload.to_envelope())


def test_receipt_ref_metadata_is_resanitized_in_payload() -> None:
    receipt_ref = _malicious_receipt_ref()
    result = TokenOptimizationResult(
        content="optimized",
        decision=TokenOptimizationDecision.APPLY,
        receipt_ref=receipt_ref,
    )
    signal = build_token_optimization_signal(result)
    payload = token_optimization_signal_to_payload(signal)

    assert payload.receipt_metadata == {
        "run_id": "run-ref-meta",
        "fixture_id": "fixture-42",
        "category": "memory_summary",
        "strategy_id": "strategy-conservative",
    }
    metadata_text = str(payload.receipt_metadata)
    assert _SECRET_CONTENT not in metadata_text
    assert "secret prompt" not in metadata_text
    assert "secret evidence" not in metadata_text


def test_raw_content_prompt_context_evidence_cannot_appear_in_payload_envelope() -> None:
    signal = TokenOptimizationSignal(
        signal_id="signal-unsafe-1",
        signal_type=TokenOptimizationSignalType.OPTIMIZATION_OUTCOME,
        metadata={
            "original_content": _SECRET_CONTENT,
            "optimized_content": "optimized secret",
            "prompt": "do not emit",
            "context": {"fragments": ["secret"]},
            "evidence": "secret evidence",
            "run_id": "run-safe-2",
        },
        receipt_ref=_malicious_receipt_ref(),
    )
    payload = token_optimization_signal_to_payload(signal)
    envelope_text = str(payload.to_envelope())

    assert _SECRET_CONTENT not in envelope_text
    assert "do not emit" not in envelope_text
    assert "secret evidence" not in envelope_text
    assert "optimized secret" not in envelope_text
    assert "original_content" not in payload.metadata
    assert "prompt" not in payload.metadata
    assert "context" not in payload.metadata
    assert "evidence" not in payload.metadata


def test_register_token_optimization_domain_signal_is_idempotent() -> None:
    register_token_optimization_domain_signal()
    register_token_optimization_domain_signal()


def test_emit_token_optimization_domain_signal_records_domain_signal_on_bus() -> None:
    bus = RuntimeEventBus(record_history=True)
    ctx = EmitContext(
        task_id="task-1",
        run_id="run-1",
        tenant_id="tenant-a",
        bus=bus,
    )
    signal = _optimization_signal()

    event = emit_token_optimization_domain_signal(ctx, signal)

    assert event.event_type == RuntimeEventType.DOMAIN_SIGNAL
    assert event.event_kind == TOKEN_OPTIMIZATION_SIGNAL_EVENT_KIND
    assert event.event_category == EventCategory.PLATFORM
    assert bus.history[-1].event_id == event.event_id
    assert event.payload["payload_schema_id"] == TOKEN_OPTIMIZATION_SIGNAL_PAYLOAD_SCHEMA_ID
    assert event.payload["payload_schema_id"] == TokenOptimizationSignalPayloadV1.schema_id


def test_emit_token_optimization_domain_signal_production_mode_stays_safe() -> None:
    signal = TokenOptimizationSignal(
        signal_id="signal-prod-1",
        signal_type=TokenOptimizationSignalType.REGRESSION_RESULT,
        metadata={
            "content": _SECRET_CONTENT,
            "prompt": "secret prompt",
            "run_id": "run-prod-safe",
        },
    )
    ctx = EmitContext(
        task_id="task-prod",
        run_id="run-prod",
        tenant_id="tenant-prod",
        production_mode=True,
    )

    event = emit_token_optimization_domain_signal(ctx, signal)

    assert _SECRET_CONTENT not in str(event.payload)
    assert "secret prompt" not in str(event.payload)
    assert event.payload["data"]["metadata"] == {"run_id": "run-prod-safe"}


def test_emit_token_optimization_domain_signal_regression_result() -> None:
    summary = run_token_regression_benchmarks(token_counter=default_token_counter)
    signal = build_token_regression_signal(summary.results[0])
    bus = RuntimeEventBus(record_history=True)
    ctx = EmitContext(task_id="task-reg", run_id="run-reg", bus=bus)

    event = emit_token_optimization_domain_signal(ctx, signal)

    assert event.event_type == RuntimeEventType.DOMAIN_SIGNAL
    assert event.payload["data"]["fixture_id"] == signal.fixture_id
    assert len(bus.history) == 1


def test_invalid_metadata_does_not_leak_through_emission() -> None:
    bus = RuntimeEventBus(record_history=True)
    ctx = EmitContext(task_id="task-mal", run_id="run-mal", bus=bus)
    signal = TokenOptimizationSignal(
        signal_id="signal-mal-1",
        signal_type=TokenOptimizationSignalType.OPTIMIZATION_OUTCOME,
        metadata={
            "messages": ["user: secret", "assistant: secret"],
            "document": "secret document",
            "tool_args": {"secret": True},
            "run_id": "run-mal-safe",
        },
        receipt_ref=_malicious_receipt_ref(),
    )

    event = emit_token_optimization_domain_signal(ctx, signal)
    event_text = str(event.payload)

    assert "secret document" not in event_text
    assert "user: secret" not in event_text
    assert event.payload["data"]["metadata"] == {"run_id": "run-mal-safe"}


def test_domain_events_module_has_no_exporter_dependencies() -> None:
    source = _DOMAIN_EVENTS_MODULE.read_text(encoding="utf-8")

    assert "ObservabilityExporter" not in source
    assert "elasticsearch" not in source.lower()
    assert "kibana" not in source.lower()
    assert "emit_token_optimization_signal" not in source
