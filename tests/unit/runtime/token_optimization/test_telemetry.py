# © Artur Czarnecki. All rights reserved.

"""TOKEN-6A-lite: token savings telemetry payload shape tests."""

from __future__ import annotations

import ast
from pathlib import Path

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
    build_compression_receipt,
)
from intergrax.runtime.token_optimization.telemetry import (
    TokenOptimizationTelemetryEventType,
    TokenOptimizationTelemetryPayload,
    TokenOptimizationTelemetryValidationStatus,
    build_token_savings_telemetry_payload,
    token_savings_payload_to_attributes,
    validate_token_savings_telemetry_payload,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_ORIGINAL = '{"tools": [{"name": "search", "description": "Find items"}]}'
_OPTIMIZED = '{"tools":[{"name":"search","description":"Find items"}]}'
_SECRET_ORIGINAL = "super-secret-api-key-value-12345"
_SECRET_OPTIMIZED = "super-secret-api-key-value-12345"
_TELEMETRY_MODULE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "token_optimization"
    / "telemetry.py"
)


def _strategy() -> TokenOptimizationStrategyRef:
    return TokenOptimizationStrategyRef(
        strategy_id="tool_catalog.minimize",
        mechanism=TokenOptimizationMechanism.TOOL_CATALOG_COMPACTION,
        kind=TokenOptimizationStrategyKind.SCHEMA_MINIMIZATION,
        safety_class=StrategySafetyClass.LOSSLESS,
        plugin_id="builtin.tool_catalog",
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


def _receipt_with_measurement(
    *,
    fallback_used: bool = False,
    bypass_reason: TokenOptimizationBypassReason | None = None,
    validation: ProtectedRegionValidationResult | None = None,
) -> CompressionReceipt:
    request = TokenOptimizationRequest(
        content=_ORIGINAL,
        source_type=TokenOptimizationSourceType.TOOL_CATALOG,
        attribution=TokenOptimizationAttribution(
            run_id="run-abc",
            step_id="step-xyz",
            plugin_id="builtin.tool_catalog",
        ),
        strategy=_strategy(),
    )
    result = TokenOptimizationResult(
        content=_OPTIMIZED,
        decision=TokenOptimizationDecision.FALLBACK if fallback_used else TokenOptimizationDecision.APPLY,
        measurement=_measurement(),
        validation=validation,
        strategy=_strategy(),
        fallback_used=fallback_used,
        bypass_reason=bypass_reason,
    )
    return build_compression_receipt(
        original_content=_ORIGINAL,
        optimized_content=_OPTIMIZED,
        request=request,
        result=result,
        receipt_id="receipt-telemetry-1",
        created_at="2026-07-01T12:00:00+00:00",
    )


def test_build_token_savings_telemetry_payload_from_receipt_with_measurement() -> None:
    receipt = _receipt_with_measurement()
    payload = build_token_savings_telemetry_payload(receipt=receipt)

    assert payload.event_type is TokenOptimizationTelemetryEventType.TOKEN_SAVINGS
    assert payload.receipt_id == "receipt-telemetry-1"
    assert payload.run_id == "run-abc"
    assert payload.step_id == "step-xyz"
    assert payload.source_type is TokenOptimizationSourceType.TOOL_CATALOG
    assert payload.token_category is TokenCategory.TOOL_CATALOG
    assert payload.strategy_id == "tool_catalog.minimize"
    assert payload.plugin_id == "builtin.tool_catalog"


def test_build_payload_includes_token_fields_and_confidence() -> None:
    payload = build_token_savings_telemetry_payload(receipt=_receipt_with_measurement())

    assert payload.baseline_tokens == 100
    assert payload.optimized_tokens == 75
    assert payload.saved_tokens == 25
    assert payload.saved_ratio == pytest.approx(0.25)
    assert payload.savings_confidence is TokenSavingsClaimConfidence.MEASURED


def test_build_payload_includes_validation_status_when_present() -> None:
    validation = ProtectedRegionValidationResult(
        status=ProtectedRegionValidationStatus.PASSED,
        regions_checked=2,
        regions_preserved=2,
        regions_failed=0,
    )
    payload = build_token_savings_telemetry_payload(
        receipt=_receipt_with_measurement(validation=validation),
    )
    assert payload.validation_status is ProtectedRegionValidationStatus.PASSED


def test_build_payload_represents_fallback_used_and_bypass_reason() -> None:
    payload = build_token_savings_telemetry_payload(
        receipt=_receipt_with_measurement(
            fallback_used=True,
            bypass_reason=TokenOptimizationBypassReason.VALIDATION_FAILED,
        ),
    )
    assert payload.fallback_used is True
    assert payload.bypass_reason is TokenOptimizationBypassReason.VALIDATION_FAILED
    assert payload.decision is TokenOptimizationDecision.FALLBACK


def test_build_payload_without_measurement_has_none_token_fields() -> None:
    receipt = CompressionReceipt(
        receipt_id="receipt-no-measurement",
        created_at="2026-07-01T12:00:00+00:00",
        source_type=TokenOptimizationSourceType.TOOL_CATALOG,
        decision=TokenOptimizationDecision.BYPASS,
        original_hash="abc",
        optimized_hash="def",
        bypass_reason=TokenOptimizationBypassReason.DISABLED,
    )
    payload = build_token_savings_telemetry_payload(receipt=receipt)

    assert payload.baseline_tokens is None
    assert payload.optimized_tokens is None
    assert payload.saved_tokens is None
    assert payload.saved_ratio is None
    assert payload.savings_confidence is None


def test_build_payload_accepts_workflow_id_override_and_metadata_merge() -> None:
    receipt = _receipt_with_measurement()
    payload = build_token_savings_telemetry_payload(
        receipt=receipt,
        workflow_id="workflow-override",
        metadata={"proof_case": "tool_catalog"},
    )
    assert payload.workflow_id == "workflow-override"
    assert payload.metadata["proof_case"] == "tool_catalog"


def test_validate_token_savings_telemetry_payload_passes_valid_payload() -> None:
    payload = build_token_savings_telemetry_payload(receipt=_receipt_with_measurement())
    outcome = validate_token_savings_telemetry_payload(payload)

    assert outcome.status is TokenOptimizationTelemetryValidationStatus.PASSED
    assert outcome.failures == ()


def test_validate_token_savings_telemetry_payload_fails_inconsistent_saved_tokens() -> None:
    base = build_token_savings_telemetry_payload(receipt=_receipt_with_measurement())
    invalid = TokenOptimizationTelemetryPayload(
        event_type=base.event_type,
        receipt_id=base.receipt_id,
        source_type=base.source_type,
        decision=base.decision,
        baseline_tokens=100,
        optimized_tokens=75,
        saved_tokens=10,
        saved_ratio=0.25,
    )
    outcome = validate_token_savings_telemetry_payload(invalid)

    assert outcome.status is TokenOptimizationTelemetryValidationStatus.FAILED
    assert "saved_tokens must equal baseline_tokens - optimized_tokens" in outcome.failures


def test_validate_token_savings_telemetry_payload_fails_invalid_saved_ratio() -> None:
    base = build_token_savings_telemetry_payload(receipt=_receipt_with_measurement())
    invalid = TokenOptimizationTelemetryPayload(
        event_type=base.event_type,
        receipt_id=base.receipt_id,
        source_type=base.source_type,
        decision=base.decision,
        baseline_tokens=100,
        optimized_tokens=50,
        saved_tokens=50,
        saved_ratio=1.5,
    )
    outcome = validate_token_savings_telemetry_payload(invalid)

    assert outcome.status is TokenOptimizationTelemetryValidationStatus.FAILED
    assert "saved_ratio must be between 0.0 and 1.0 when baseline_tokens > 0" in outcome.failures


def test_validate_passes_when_validation_status_failed() -> None:
    payload = build_token_savings_telemetry_payload(
        receipt=_receipt_with_measurement(
            validation=ProtectedRegionValidationResult(
                status=ProtectedRegionValidationStatus.FAILED,
                regions_checked=1,
                regions_preserved=0,
                regions_failed=1,
                failures=("missing path",),
            ),
        ),
    )
    outcome = validate_token_savings_telemetry_payload(payload)
    assert outcome.status is TokenOptimizationTelemetryValidationStatus.PASSED
    assert payload.validation_status is ProtectedRegionValidationStatus.FAILED


def test_token_savings_payload_to_attributes_returns_safe_scalar_namespaced_fields() -> None:
    payload = build_token_savings_telemetry_payload(receipt=_receipt_with_measurement())
    attributes = token_savings_payload_to_attributes(payload)

    assert attributes["intergrax.token_optimization.event_type"] == "token_optimization.savings"
    assert attributes["intergrax.token_optimization.receipt_id"] == "receipt-telemetry-1"
    assert attributes["intergrax.token_optimization.run_id"] == "run-abc"
    assert attributes["intergrax.token_optimization.step_id"] == "step-xyz"
    assert attributes["intergrax.token_optimization.source_type"] == "tool_catalog"
    assert attributes["intergrax.token_optimization.token_category"] == "tool_catalog"
    assert attributes["intergrax.token_optimization.strategy_id"] == "tool_catalog.minimize"
    assert attributes["intergrax.token_optimization.plugin_id"] == "builtin.tool_catalog"
    assert attributes["intergrax.token_optimization.decision"] == "apply"
    assert attributes["intergrax.token_optimization.fallback_used"] is False
    assert attributes["intergrax.token_optimization.baseline_tokens"] == 100
    assert attributes["intergrax.token_optimization.optimized_tokens"] == 75
    assert attributes["intergrax.token_optimization.saved_tokens"] == 25
    assert attributes["intergrax.token_optimization.saved_ratio"] == pytest.approx(0.25)
    assert attributes["intergrax.token_optimization.savings_confidence"] == "measured"

    for value in attributes.values():
        assert isinstance(value, (str, int, float, bool, type(None)))


def test_attributes_exclude_raw_original_and_optimized_content() -> None:
    receipt = build_compression_receipt(
        original_content=_SECRET_ORIGINAL,
        optimized_content=_SECRET_OPTIMIZED,
        request=TokenOptimizationRequest(
            content=_SECRET_ORIGINAL,
            source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
            strategy=_strategy(),
        ),
        result=TokenOptimizationResult(
            content=_SECRET_OPTIMIZED,
            decision=TokenOptimizationDecision.APPLY,
            measurement=TokenSavingsMeasurement(
                baseline_tokens=50,
                optimized_tokens=40,
                saved_tokens=10,
                saved_ratio=0.2,
                confidence=TokenSavingsClaimConfidence.MEASURED,
                category=TokenCategory.INPUT_CONTEXT,
                source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
            ),
        ),
        receipt_id="receipt-secret-telemetry",
        created_at="2026-07-01T12:00:00+00:00",
    )
    attributes = token_savings_payload_to_attributes(
        build_token_savings_telemetry_payload(receipt=receipt),
    )
    serialized = str(attributes)
    assert _SECRET_ORIGINAL not in serialized
    assert _SECRET_OPTIMIZED not in serialized
    assert "metadata" not in attributes


def test_no_telemetry_emission_is_performed() -> None:
    source = _TELEMETRY_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_import_prefixes = (
        "intergrax.runtime.observability",
        "logging",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in forbidden_import_prefixes:
                    assert not alias.name.startswith(prefix)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for prefix in forbidden_import_prefixes:
                assert not module.startswith(prefix)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "info",
                "debug",
                "warning",
                "error",
                "emit",
            }:
                pytest.fail("telemetry module must not emit logs or events")
