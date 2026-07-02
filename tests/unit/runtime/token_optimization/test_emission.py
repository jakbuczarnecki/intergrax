# © Artur Czarnecki. All rights reserved.

"""TOKEN-OBS-1C: explicit opt-in token optimization emission helpers."""

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
    register_token_optimization_domain_signal,
)
from intergrax.runtime.token_optimization.emission import (
    TokenOptimizationEmissionResult,
    emit_token_optimization_outcome,
    emit_token_regression_result,
    emit_token_regression_summary,
)
from intergrax.runtime.token_optimization.regression import (
    default_token_counter,
    run_token_regression_benchmarks,
)
from intergrax.runtime.token_optimization.signals import TokenOptimizationSignalType

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_EMISSION_MODULE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "token_optimization"
    / "emission.py"
)
_SCOPE_GUARD_MODULES = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "token_optimization"
    / "tool_schema.py",
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "token_optimization"
    / "context_pack.py",
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "token_optimization"
    / "regression.py",
    Path(__file__).resolve().parents[4] / "intergrax" / "memory" / "summary_compressor.py",
)
_COMPRESSIBLE_SUMMARY = (
    "  Session   summary   for   user   preferences.\n\n\n"
    "User   prefers   concise   answers.\n\n\n\n"
    "Next   step:   review   docs.  "
)
_SECRET_CONTENT = "super-secret-user-memory-content-should-never-appear"
_EMISSION_HELPER_NAMES = (
    "emit_token_optimization_outcome",
    "emit_token_regression_result",
    "emit_token_regression_summary",
)


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


def _emit_context() -> tuple[EmitContext, RuntimeEventBus]:
    bus = RuntimeEventBus(record_history=True)
    ctx = EmitContext(
        task_id="task-1",
        run_id="run-1",
        tenant_id="tenant-a",
        bus=bus,
    )
    return ctx, bus


def _optimization_outcome():
    return optimize_memory_summary(
        _COMPRESSIBLE_SUMMARY,
        token_policy=_enabled_policy(),
        token_counter=default_token_counter,
        attribution=TokenOptimizationAttribution(
            run_id="run-memory-1",
            step_id="step-memory-1",
            tenant_id="tenant-a",
        ),
    )


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


def test_emit_token_optimization_outcome_records_domain_signal_on_bus() -> None:
    ctx, bus = _emit_context()
    outcome = _optimization_outcome()

    result = emit_token_optimization_outcome(ctx, outcome)

    assert isinstance(result, TokenOptimizationEmissionResult)
    assert result.emitted is True
    assert result.event is not None
    assert result.event.event_type == RuntimeEventType.DOMAIN_SIGNAL
    assert result.event.event_kind == TOKEN_OPTIMIZATION_SIGNAL_EVENT_KIND
    assert result.event.event_category == EventCategory.PLATFORM
    assert bus.history[-1].event_id == result.event.event_id
    assert len(bus.history) == 1


def test_emit_token_optimization_outcome_payload_schema_and_kind() -> None:
    ctx, _bus = _emit_context()
    outcome = _optimization_outcome()

    result = emit_token_optimization_outcome(ctx, outcome)

    assert isinstance(result.payload, TokenOptimizationSignalPayloadV1)
    assert result.event is not None
    assert result.event.payload["payload_schema_id"] == TOKEN_OPTIMIZATION_SIGNAL_PAYLOAD_SCHEMA_ID
    assert result.event.payload["payload_schema_id"] == TokenOptimizationSignalPayloadV1.schema_id
    assert result.signal.signal_type is TokenOptimizationSignalType.OPTIMIZATION_OUTCOME


def test_emit_token_optimization_outcome_includes_token_savings_fields() -> None:
    ctx, _bus = _emit_context()
    outcome = _optimization_outcome()

    result = emit_token_optimization_outcome(ctx, outcome)

    assert result.payload.baseline_tokens is not None and result.payload.baseline_tokens > 0
    assert result.payload.optimized_tokens is not None
    assert result.payload.saved_tokens is not None
    assert result.payload.saved_ratio is not None


def test_emit_token_optimization_outcome_includes_validation_and_fallback_fields() -> None:
    ctx, _bus = _emit_context()
    outcome = _optimization_outcome()

    result = emit_token_optimization_outcome(ctx, outcome)

    assert result.payload.validation_status in {"passed", "not_applicable", "failed", "skipped"}
    assert isinstance(result.payload.fallback_status, bool)


def test_emit_token_optimization_outcome_includes_safe_attribution_fields() -> None:
    ctx, _bus = _emit_context()
    outcome = _optimization_outcome()

    result = emit_token_optimization_outcome(ctx, outcome)

    assert result.payload.run_id == "run-memory-1"
    assert result.payload.step_id == "step-memory-1"
    assert result.payload.tenant_id == "tenant-a"


def test_emit_token_optimization_outcome_sanitizes_metadata() -> None:
    ctx, _bus = _emit_context()
    outcome = _optimization_outcome()

    result = emit_token_optimization_outcome(
        ctx,
        outcome,
        metadata={
            "run_id": "run-safe",
            "content": _SECRET_CONTENT,
            "prompt": "secret prompt",
            "profile": "conservative",
            "unsafe_custom": "drop-me",
        },
    )

    assert result.metadata.get("run_id") == "run-safe"
    assert result.metadata.get("profile") == "conservative"
    assert "content" not in result.metadata
    assert "prompt" not in result.metadata
    assert "unsafe_custom" not in result.metadata
    assert _SECRET_CONTENT not in str(result.event.payload)


def test_emit_token_optimization_outcome_receipt_ref_metadata_is_sanitized() -> None:
    ctx, _bus = _emit_context()
    token_result = TokenOptimizationResult(
        content="optimized",
        decision=TokenOptimizationDecision.APPLY,
        receipt_ref=_malicious_receipt_ref(),
    )

    result = emit_token_optimization_outcome(ctx, token_result)

    assert result.payload.receipt_metadata == {
        "run_id": "run-ref-meta",
        "fixture_id": "fixture-42",
        "category": "memory_summary",
        "strategy_id": "strategy-conservative",
    }
    metadata_text = str(result.payload.receipt_metadata)
    assert _SECRET_CONTENT not in metadata_text
    assert "secret prompt" not in metadata_text
    assert "secret evidence" not in metadata_text


def test_emit_token_optimization_outcome_raw_content_cannot_appear_in_event_payload() -> None:
    ctx, _bus = _emit_context()
    token_result = TokenOptimizationResult(
        content="optimized",
        decision=TokenOptimizationDecision.APPLY,
        receipt_ref=_malicious_receipt_ref(),
        metadata={
            "original_content": _SECRET_CONTENT,
            "optimized_content": "optimized secret",
            "prompt": "do not emit",
            "context": {"fragments": ["secret"]},
            "evidence": "secret evidence",
            "run_id": "run-safe-2",
        },
    )

    result = emit_token_optimization_outcome(ctx, token_result)
    assert result.event is not None
    envelope_text = str(result.event.payload)

    assert _SECRET_CONTENT not in envelope_text
    assert "do not emit" not in envelope_text
    assert "secret evidence" not in envelope_text
    assert "optimized secret" not in envelope_text
    assert "original_content" not in result.payload.metadata
    assert "prompt" not in result.payload.metadata
    assert "context" not in result.payload.metadata
    assert "evidence" not in result.payload.metadata


def test_emit_token_regression_result_records_domain_signal() -> None:
    ctx, bus = _emit_context()
    summary = run_token_regression_benchmarks(token_counter=default_token_counter)
    regression_result = summary.results[0]

    result = emit_token_regression_result(ctx, regression_result)

    assert result.emitted is True
    assert result.event is not None
    assert result.event.event_type == RuntimeEventType.DOMAIN_SIGNAL
    assert result.event.event_kind == TOKEN_OPTIMIZATION_SIGNAL_EVENT_KIND
    assert result.signal.signal_type is TokenOptimizationSignalType.REGRESSION_RESULT
    assert result.event.payload["data"]["fixture_id"] == regression_result.fixture_id
    assert len(bus.history) == 1


def test_emit_token_regression_summary_records_aggregate_domain_signal() -> None:
    ctx, bus = _emit_context()
    summary = run_token_regression_benchmarks(token_counter=default_token_counter)

    result = emit_token_regression_summary(ctx, summary)

    assert result.emitted is True
    assert result.event is not None
    assert result.event.event_type == RuntimeEventType.DOMAIN_SIGNAL
    assert result.signal.signal_type is TokenOptimizationSignalType.REGRESSION_SUMMARY
    assert result.payload.baseline_tokens == summary.total_baseline_tokens
    assert result.payload.metadata.get("total_fixtures") == summary.total_fixtures
    assert len(bus.history) == 1


def test_emit_token_optimization_outcome_dry_run_returns_signal_without_event() -> None:
    ctx, bus = _emit_context()
    outcome = _optimization_outcome()

    result = emit_token_optimization_outcome(ctx, outcome, emit=False)

    assert isinstance(result.payload, TokenOptimizationSignalPayloadV1)
    assert result.event is None
    assert result.emitted is False
    assert result.signal.signal_id
    assert len(bus.history) == 0


def test_emit_token_regression_result_dry_run_returns_signal_without_event() -> None:
    ctx, bus = _emit_context()
    summary = run_token_regression_benchmarks(token_counter=default_token_counter)

    result = emit_token_regression_result(ctx, summary.results[0], emit=False)

    assert result.event is None
    assert result.emitted is False
    assert len(bus.history) == 0


@pytest.mark.parametrize("module_path", _SCOPE_GUARD_MODULES, ids=lambda p: p.name)
def test_optimizer_and_regression_modules_do_not_import_emission_helpers(
    module_path: Path,
) -> None:
    source = module_path.read_text(encoding="utf-8")

    assert "token_optimization.emission" not in source
    for helper_name in _EMISSION_HELPER_NAMES:
        assert helper_name not in source


def test_emission_module_has_no_exporter_dependencies() -> None:
    source = _EMISSION_MODULE.read_text(encoding="utf-8")

    assert "ObservabilityExporter" not in source
    assert "elasticsearch" not in source.lower()
    assert "kibana" not in source.lower()
