# © Artur Czarnecki. All rights reserved.

"""TOKEN-OBS-1A: token optimization domain signal model and safe emission tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.memory.summary_compressor import (
    optimize_memory_summary,
)
from intergrax.runtime.token_optimization.contracts import (
    CompressionLevel,
    CompressionReceiptRef,
    TokenOptimizationAttribution,
    TokenOptimizationDecision,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationResult,
)
from intergrax.runtime.token_optimization.regression import (
    default_token_counter,
    run_token_regression_benchmarks,
)
from intergrax.runtime.token_optimization.signals import (
    InMemoryTokenOptimizationSignalSink,
    NoOpTokenOptimizationSignalSink,
    TokenOptimizationSignalType,
    build_token_optimization_signal,
    build_token_regression_signal,
    build_token_regression_summary_signal,
    emit_token_optimization_signal,
    sanitize_signal_metadata,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SIGNALS_MODULE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "token_optimization"
    / "signals.py"
)
_COMPRESSIBLE_SUMMARY = (
    "  Session   summary   for   user   preferences.\n\n\n"
    "User   prefers   concise   answers.\n\n\n\n"
    "Next   step:   review   docs.  "
)
_SECRET_CONTENT = "super-secret-user-memory-content-should-never-appear"


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


def test_signal_built_from_memory_summary_optimization_outcome() -> None:
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
    signal = build_token_optimization_signal(outcome)

    assert signal.signal_type is TokenOptimizationSignalType.OPTIMIZATION_OUTCOME
    assert signal.baseline_tokens is not None
    assert signal.optimized_tokens is not None
    assert signal.saved_tokens == signal.baseline_tokens - signal.optimized_tokens
    assert signal.run_id == "run-memory-1"
    assert signal.step_id == "step-memory-1"
    assert signal.tenant_id == "tenant-a"


def test_signal_built_from_token_regression_result() -> None:
    summary = run_token_regression_benchmarks(token_counter=default_token_counter)
    result = summary.results[0]
    signal = build_token_regression_signal(result)

    assert signal.signal_type is TokenOptimizationSignalType.REGRESSION_RESULT
    assert signal.fixture_id == result.fixture_id
    assert signal.baseline_tokens == result.baseline_tokens
    assert signal.optimized_tokens == result.optimized_tokens
    assert signal.saved_tokens == result.saved_tokens
    assert signal.saved_ratio == result.saved_ratio


def test_signal_includes_token_savings_fields() -> None:
    outcome = optimize_memory_summary(
        _COMPRESSIBLE_SUMMARY,
        token_policy=_enabled_policy(),
        token_counter=default_token_counter,
    )
    signal = build_token_optimization_signal(outcome)

    assert signal.baseline_tokens is not None and signal.baseline_tokens > 0
    assert signal.optimized_tokens is not None
    assert signal.saved_tokens is not None
    assert signal.saved_ratio is not None
    if signal.baseline_tokens > 0:
        assert signal.saved_ratio == pytest.approx(
            signal.saved_tokens / signal.baseline_tokens
        )


def test_signal_includes_validation_and_fallback_fields() -> None:
    outcome = optimize_memory_summary(
        _COMPRESSIBLE_SUMMARY,
        token_policy=_enabled_policy(),
        token_counter=default_token_counter,
    )
    signal = build_token_optimization_signal(outcome)

    assert signal.validation_status in {"passed", "not_applicable", "failed", "skipped"}
    assert isinstance(signal.fallback_status, bool)


def test_signal_includes_receipt_ref_when_available() -> None:
    outcome = optimize_memory_summary(
        _COMPRESSIBLE_SUMMARY,
        token_policy=_enabled_policy(),
        token_counter=default_token_counter,
    )
    signal = build_token_optimization_signal(outcome)

    assert outcome.receipt is not None
    assert signal.receipt_id == outcome.receipt.receipt_id
    assert signal.receipt_ref is not None
    assert signal.receipt_ref.receipt_id == outcome.receipt.receipt_id


def test_signal_includes_attribution_fields_from_metadata() -> None:
    summary = run_token_regression_benchmarks()
    result = summary.results[0]
    signal = build_token_regression_signal(
        result,
        metadata={
            "run_id": "run-reg-1",
            "step_id": "step-reg-1",
            "tenant_id": "tenant-reg",
        },
    )

    assert signal.run_id == "run-reg-1"
    assert signal.step_id == "step-reg-1"
    assert signal.tenant_id == "tenant-reg"


def test_unsafe_metadata_keys_are_dropped() -> None:
    sanitized = sanitize_signal_metadata(
        {
            "run_id": "run-safe",
            "content": _SECRET_CONTENT,
            "prompt": "secret prompt",
            "profile": "conservative",
            "unsafe_custom": "drop-me",
        }
    )

    assert sanitized == {"run_id": "run-safe", "profile": "conservative"}
    assert "content" not in sanitized
    assert "prompt" not in sanitized
    assert "unsafe_custom" not in sanitized


def test_raw_content_fields_are_not_emitted_in_signal_metadata() -> None:
    outcome = optimize_memory_summary(
        _COMPRESSIBLE_SUMMARY,
        token_policy=_enabled_policy(),
        token_counter=default_token_counter,
    )
    signal = build_token_optimization_signal(
        outcome,
        metadata={
            "original_content": outcome.original_content,
            "optimized_content": outcome.optimized_content,
            "prompt": "do not emit",
            "context": {"fragments": ["secret"]},
            "evidence": "secret evidence",
            "run_id": "run-safe-2",
        },
    )

    metadata_text = str(signal.metadata)
    assert _SECRET_CONTENT not in metadata_text
    assert "do not emit" not in metadata_text
    assert "secret evidence" not in metadata_text
    assert signal.metadata.get("run_id") == "run-safe-2"
    assert "original_content" not in signal.metadata
    assert "optimized_content" not in signal.metadata
    assert "context" not in signal.metadata
    assert "evidence" not in signal.metadata


def test_long_strings_are_limited_by_sanitizer() -> None:
    long_value = "x" * 300
    sanitized = sanitize_signal_metadata({"description": long_value})

    assert "description" in sanitized
    assert len(sanitized["description"]) <= 160
    assert sanitized["description"].endswith("...")


def test_nested_unsafe_dict_and_list_payloads_are_dropped() -> None:
    sanitized = sanitize_signal_metadata(
        {
            "run_id": "run-nested",
            "payload": {"nested": "secret"},
            "failure_reasons": ["a", "b", "c"],
            "changed": True,
        }
    )

    assert sanitized == {"run_id": "run-nested", "changed": True}


def test_in_memory_sink_receives_emitted_signal() -> None:
    outcome = optimize_memory_summary(
        _COMPRESSIBLE_SUMMARY,
        token_policy=_enabled_policy(),
        token_counter=default_token_counter,
    )
    signal = build_token_optimization_signal(outcome)
    sink = InMemoryTokenOptimizationSignalSink()

    result = emit_token_optimization_signal(signal, sink)

    assert result.accepted is True
    assert result.signal is signal
    assert len(sink.signals) == 1
    assert sink.signals[0] is signal


def test_no_op_sink_accepts_signal_and_stores_nothing() -> None:
    summary = run_token_regression_benchmarks()
    signal = build_token_regression_summary_signal(summary)
    sink = NoOpTokenOptimizationSignalSink()

    result = emit_token_optimization_signal(signal, sink)

    assert result.accepted is True
    assert not hasattr(sink, "signals") or getattr(sink, "signals", None) in (None, [])


def test_signals_module_has_no_hos_or_exporter_dependencies() -> None:
    source = _SIGNALS_MODULE.read_text(encoding="utf-8")

    assert "emit_domain_signal" not in source
    assert "ObservabilityExporter" not in source
    assert "RuntimeEvent" not in source
    assert "intergrax.runtime.events" not in source
    assert "intergrax.runtime.observability" not in source


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
            "failure_reasons": ["a", "b"],
            "unsafe_custom": "drop-me",
        },
    )


def test_receipt_ref_metadata_unsafe_keys_are_sanitized_before_signal() -> None:
    receipt_ref = _malicious_receipt_ref()
    result = TokenOptimizationResult(
        content="optimized",
        decision=TokenOptimizationDecision.APPLY,
        receipt_ref=receipt_ref,
    )
    signal = build_token_optimization_signal(result)

    assert signal.receipt_ref is not None
    assert signal.receipt_ref is not receipt_ref
    assert signal.receipt_ref.receipt_id == "receipt-malicious-1"
    assert signal.receipt_ref.run_id == "run-ref-1"
    assert signal.receipt_ref.step_id == "step-ref-1"
    assert signal.receipt_ref.strategy_id == "strategy-conservative"
    assert signal.receipt_ref.original_hash == "orig-hash-abc"
    assert signal.receipt_ref.optimized_hash == "opt-hash-xyz"
    assert signal.receipt_ref.metadata == {
        "run_id": "run-ref-meta",
        "fixture_id": "fixture-42",
        "category": "memory_summary",
        "strategy_id": "strategy-conservative",
    }


def test_receipt_ref_metadata_drops_raw_content_prompt_context_evidence() -> None:
    receipt_ref = _malicious_receipt_ref()
    result = TokenOptimizationResult(
        content="optimized",
        decision=TokenOptimizationDecision.APPLY,
        receipt_ref=receipt_ref,
    )
    signal = build_token_optimization_signal(result)

    assert signal.receipt_ref is not None
    metadata_text = str(signal.receipt_ref.metadata)
    assert _SECRET_CONTENT not in metadata_text
    assert "secret prompt" not in metadata_text
    assert "secret evidence" not in metadata_text
    assert "optimized secret body" not in metadata_text
    assert "original_content" not in signal.receipt_ref.metadata
    assert "optimized_content" not in signal.receipt_ref.metadata
    assert "prompt" not in signal.receipt_ref.metadata
    assert "context" not in signal.receipt_ref.metadata
    assert "evidence" not in signal.receipt_ref.metadata


def test_receipt_ref_metadata_preserves_safe_scalar_keys() -> None:
    receipt_ref = CompressionReceiptRef(
        receipt_id="receipt-safe-meta",
        metadata={
            "run_id": "run-safe-meta",
            "fixture_id": "fixture-safe",
            "category": "tool_schema",
            "strategy_id": "strategy-light",
        },
    )
    result = TokenOptimizationResult(
        content="optimized",
        decision=TokenOptimizationDecision.APPLY,
        receipt_ref=receipt_ref,
    )
    signal = build_token_optimization_signal(result)

    assert signal.receipt_ref is not None
    assert signal.receipt_ref.metadata == {
        "run_id": "run-safe-meta",
        "fixture_id": "fixture-safe",
        "category": "tool_schema",
        "strategy_id": "strategy-light",
    }


def test_receipt_ref_metadata_drops_nested_dict_and_list_payloads() -> None:
    receipt_ref = CompressionReceiptRef(
        receipt_id="receipt-nested-meta",
        metadata={
            "run_id": "run-nested-meta",
            "payload": {"nested": "secret"},
            "failure_reasons": ["a", "b"],
            "changed": True,
        },
    )
    result = TokenOptimizationResult(
        content="optimized",
        decision=TokenOptimizationDecision.APPLY,
        receipt_ref=receipt_ref,
    )
    signal = build_token_optimization_signal(result)

    assert signal.receipt_ref is not None
    assert signal.receipt_ref.metadata == {"run_id": "run-nested-meta", "changed": True}


def test_receipt_ref_identity_fields_preserved_after_sanitization() -> None:
    receipt_ref = _malicious_receipt_ref()
    result = TokenOptimizationResult(
        content="optimized",
        decision=TokenOptimizationDecision.APPLY,
        receipt_ref=receipt_ref,
    )
    signal = build_token_optimization_signal(result)

    assert signal.receipt_ref is not None
    assert signal.receipt_ref.receipt_id == receipt_ref.receipt_id
    assert signal.receipt_ref.run_id == receipt_ref.run_id
    assert signal.receipt_ref.step_id == receipt_ref.step_id
    assert signal.receipt_ref.strategy_id == receipt_ref.strategy_id
    assert signal.receipt_ref.original_hash == receipt_ref.original_hash
    assert signal.receipt_ref.optimized_hash == receipt_ref.optimized_hash
