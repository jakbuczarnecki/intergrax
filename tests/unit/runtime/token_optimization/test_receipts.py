# © Artur Czarnecki. All rights reserved.

"""TOKEN-1C: compression receipt builder and integrity validation tests."""

from __future__ import annotations

import hashlib

import pytest

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
    StrategySafetyClass,
    TokenCategory,
    TokenOptimizationAttribution,
    TokenOptimizationBypassReason,
    TokenOptimizationDecision,
    TokenOptimizationMechanism,
    TokenOptimizationRequest,
    TokenOptimizationResult,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
    TokenOptimizationStrategyRef,
    TokenSavingsClaimConfidence,
    TokenSavingsMeasurement,
)
from intergrax.runtime.token_optimization.receipts import (
    CompressionReceipt,
    CompressionReceiptValidationStatus,
    build_compression_receipt,
    hash_content,
    make_compression_receipt_ref,
    validate_receipt_integrity,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_ORIGINAL = '{"tools": [{"name": "search", "description": "Find items"}]}'
_OPTIMIZED = '{"tools":[{"name":"search","description":"Find items"}]}'
_SECRET_ORIGINAL = "super-secret-api-key-value-12345"
_SECRET_OPTIMIZED = "super-secret-api-key-value-12345"


def _strategy(*, strategy_id: str = "tool_catalog.minimize") -> TokenOptimizationStrategyRef:
    return TokenOptimizationStrategyRef(
        strategy_id=strategy_id,
        mechanism=TokenOptimizationMechanism.TOOL_CATALOG_COMPACTION,
        kind=TokenOptimizationStrategyKind.SCHEMA_MINIMIZATION,
        safety_class=StrategySafetyClass.LOSSLESS,
        plugin_id="builtin.tool_catalog",
    )


def _request(*, strategy: TokenOptimizationStrategyRef | None = None) -> TokenOptimizationRequest:
    return TokenOptimizationRequest(
        content=_ORIGINAL,
        source_type=TokenOptimizationSourceType.TOOL_CATALOG,
        attribution=TokenOptimizationAttribution(
            run_id="run-abc",
            step_id="step-xyz",
            plugin_id="builtin.tool_catalog",
        ),
        strategy=strategy or _strategy(),
    )


def _measurement() -> TokenSavingsMeasurement:
    return TokenSavingsMeasurement(
        baseline_tokens=100,
        optimized_tokens=75,
        saved_tokens=25,
        saved_ratio=0.25,
        confidence=TokenSavingsClaimConfidence.MEASURED,
        category=TokenCategory.TOOL_CATALOG,
        source_type=TokenOptimizationSourceType.TOOL_CATALOG,
    )


def _result(
    *,
    decision: TokenOptimizationDecision = TokenOptimizationDecision.APPLY,
    fallback_used: bool = False,
    bypass_reason: TokenOptimizationBypassReason | None = None,
    measurement: TokenSavingsMeasurement | None = None,
    validation: ProtectedRegionValidationResult | None = None,
    strategy: TokenOptimizationStrategyRef | None = None,
) -> TokenOptimizationResult:
    return TokenOptimizationResult(
        content=_OPTIMIZED,
        decision=decision,
        measurement=measurement,
        validation=validation,
        strategy=strategy,
        fallback_used=fallback_used,
        bypass_reason=bypass_reason,
    )


def test_hash_content_returns_stable_sha256_hex_digest() -> None:
    expected = hashlib.sha256(_ORIGINAL.encode("utf-8")).hexdigest()
    assert hash_content(_ORIGINAL) == expected
    assert hash_content(_ORIGINAL) == hash_content(_ORIGINAL)


def test_hash_content_rejects_unsupported_algorithms() -> None:
    with pytest.raises(ValueError, match="unsupported hash algorithm"):
        hash_content(_ORIGINAL, algorithm="md5")


def test_build_compression_receipt_creates_original_and_optimized_hashes() -> None:
    receipt = build_compression_receipt(
        original_content=_ORIGINAL,
        optimized_content=_OPTIMIZED,
        request=_request(),
        result=_result(),
        receipt_id="receipt-test-1",
        created_at="2026-07-01T12:00:00+00:00",
    )
    assert receipt.original_hash == hash_content(_ORIGINAL)
    assert receipt.optimized_hash == hash_content(_OPTIMIZED)


def test_build_compression_receipt_copies_decision_fallback_bypass_from_result() -> None:
    receipt = build_compression_receipt(
        original_content=_ORIGINAL,
        optimized_content=_ORIGINAL,
        request=_request(),
        result=_result(
            decision=TokenOptimizationDecision.FALLBACK,
            fallback_used=True,
            bypass_reason=TokenOptimizationBypassReason.VALIDATION_FAILED,
        ),
        receipt_id="receipt-test-2",
        created_at="2026-07-01T12:00:00+00:00",
    )
    assert receipt.decision is TokenOptimizationDecision.FALLBACK
    assert receipt.fallback_used is True
    assert receipt.bypass_reason is TokenOptimizationBypassReason.VALIDATION_FAILED


def test_build_compression_receipt_records_measurement_from_result() -> None:
    measurement = _measurement()
    receipt = build_compression_receipt(
        original_content=_ORIGINAL,
        optimized_content=_OPTIMIZED,
        request=_request(),
        result=_result(measurement=measurement),
        receipt_id="receipt-test-3",
        created_at="2026-07-01T12:00:00+00:00",
    )
    assert receipt.measurement is measurement
    assert receipt.measurement is not None
    assert receipt.measurement.saved_tokens == 25


def test_build_compression_receipt_records_protected_region_validation_from_result() -> None:
    validation = ProtectedRegionValidationResult(
        status=ProtectedRegionValidationStatus.PASSED,
        regions_checked=2,
        regions_preserved=2,
        regions_failed=0,
    )
    receipt = build_compression_receipt(
        original_content=_ORIGINAL,
        optimized_content=_OPTIMIZED,
        request=_request(),
        result=_result(validation=validation),
        receipt_id="receipt-test-4",
        created_at="2026-07-01T12:00:00+00:00",
    )
    assert receipt.validation is validation
    assert receipt.validation is not None
    assert receipt.validation.status is ProtectedRegionValidationStatus.PASSED


def test_build_compression_receipt_accepts_injected_receipt_id() -> None:
    receipt = build_compression_receipt(
        original_content=_ORIGINAL,
        optimized_content=_OPTIMIZED,
        request=_request(),
        result=_result(),
        receipt_id="receipt-injected",
        created_at="2026-07-01T12:00:00+00:00",
    )
    assert receipt.receipt_id == "receipt-injected"


def test_build_compression_receipt_derives_receipt_id_deterministically() -> None:
    request = _request()
    result = _result()
    first = build_compression_receipt(
        original_content=_ORIGINAL,
        optimized_content=_OPTIMIZED,
        request=request,
        result=result,
        created_at="2026-07-01T12:00:00+00:00",
    )
    second = build_compression_receipt(
        original_content=_ORIGINAL,
        optimized_content=_OPTIMIZED,
        request=request,
        result=result,
        created_at="2026-07-01T12:00:00+00:00",
    )
    assert first.receipt_id == second.receipt_id
    assert first.receipt_id.startswith("receipt_")


def test_build_compression_receipt_prefers_result_strategy_over_request_strategy() -> None:
    request_strategy = _strategy(strategy_id="request.strategy")
    result_strategy = _strategy(strategy_id="result.strategy")
    receipt = build_compression_receipt(
        original_content=_ORIGINAL,
        optimized_content=_OPTIMIZED,
        request=_request(strategy=request_strategy),
        result=_result(strategy=result_strategy),
        receipt_id="receipt-test-5",
        created_at="2026-07-01T12:00:00+00:00",
    )
    assert receipt.strategy is result_strategy
    assert receipt.strategy is not None
    assert receipt.strategy.strategy_id == "result.strategy"


def test_make_compression_receipt_ref_maps_receipt_fields() -> None:
    receipt = build_compression_receipt(
        original_content=_ORIGINAL,
        optimized_content=_OPTIMIZED,
        request=_request(),
        result=_result(),
        receipt_id="receipt-ref-test",
        created_at="2026-07-01T12:00:00+00:00",
    )
    ref = make_compression_receipt_ref(receipt)
    assert ref.receipt_id == "receipt-ref-test"
    assert ref.run_id == "run-abc"
    assert ref.step_id == "step-xyz"
    assert ref.strategy_id == "tool_catalog.minimize"
    assert ref.original_hash == receipt.original_hash
    assert ref.optimized_hash == receipt.optimized_hash
    assert ref.metadata == receipt.metadata


def test_validate_receipt_integrity_passes_for_matching_content() -> None:
    receipt = build_compression_receipt(
        original_content=_ORIGINAL,
        optimized_content=_OPTIMIZED,
        request=_request(),
        result=_result(),
        receipt_id="receipt-validate-pass",
        created_at="2026-07-01T12:00:00+00:00",
    )
    outcome = validate_receipt_integrity(
        receipt,
        original_content=_ORIGINAL,
        optimized_content=_OPTIMIZED,
    )
    assert outcome.status is CompressionReceiptValidationStatus.PASSED
    assert outcome.failures == ()


def test_validate_receipt_integrity_fails_for_mismatched_original_content() -> None:
    receipt = build_compression_receipt(
        original_content=_ORIGINAL,
        optimized_content=_OPTIMIZED,
        request=_request(),
        result=_result(),
        receipt_id="receipt-validate-mismatch",
        created_at="2026-07-01T12:00:00+00:00",
    )
    outcome = validate_receipt_integrity(
        receipt,
        original_content="different content",
        optimized_content=_OPTIMIZED,
    )
    assert outcome.status is CompressionReceiptValidationStatus.FAILED
    assert "original_content hash mismatch" in outcome.failures


def test_validate_receipt_integrity_fails_for_missing_required_fields() -> None:
    receipt = CompressionReceipt(
        receipt_id="",
        created_at="2026-07-01T12:00:00+00:00",
        source_type=TokenOptimizationSourceType.TOOL_CATALOG,
        decision=TokenOptimizationDecision.APPLY,
        original_hash="",
        optimized_hash="abc123",
    )
    outcome = validate_receipt_integrity(receipt)
    assert outcome.status is CompressionReceiptValidationStatus.FAILED
    assert "receipt_id must not be empty" in outcome.failures
    assert "original_hash must not be empty" in outcome.failures


def test_validate_receipt_integrity_failure_messages_exclude_raw_content() -> None:
    receipt = build_compression_receipt(
        original_content=_SECRET_ORIGINAL,
        optimized_content=_SECRET_OPTIMIZED,
        request=_request(),
        result=_result(),
        receipt_id="receipt-secret",
        created_at="2026-07-01T12:00:00+00:00",
    )
    outcome = validate_receipt_integrity(
        receipt,
        original_content="tampered-secret",
        optimized_content=_SECRET_OPTIMIZED,
    )
    joined = " ".join(outcome.failures)
    assert _SECRET_ORIGINAL not in joined
    assert _SECRET_OPTIMIZED not in joined
    assert "tampered-secret" not in joined


def test_validate_receipt_integrity_fails_when_protected_region_validation_failed() -> None:
    validation = ProtectedRegionValidationResult(
        status=ProtectedRegionValidationStatus.FAILED,
        regions_checked=1,
        regions_preserved=0,
        regions_failed=1,
        failures=("missing path",),
    )
    receipt = build_compression_receipt(
        original_content=_ORIGINAL,
        optimized_content=_OPTIMIZED,
        request=_request(),
        result=_result(validation=validation),
        receipt_id="receipt-validation-failed",
        created_at="2026-07-01T12:00:00+00:00",
    )
    outcome = validate_receipt_integrity(
        receipt,
        original_content=_ORIGINAL,
        optimized_content=_OPTIMIZED,
    )
    assert outcome.status is CompressionReceiptValidationStatus.FAILED
    assert "protected_region_validation_failed" in outcome.failures
