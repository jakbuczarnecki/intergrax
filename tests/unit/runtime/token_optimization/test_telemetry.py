# © Artur Czarnecki. All rights reserved.

"""TOKEN-6A-lite / TOKEN-6A: token savings telemetry payload and summary tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.token_optimization.context_pack import optimize_context_pack
from intergrax.runtime.token_optimization.contracts import (
    CompressionLevel,
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
    StrategySafetyClass,
    TokenCategory,
    TokenOptimizationAttribution,
    TokenOptimizationBypassReason,
    TokenOptimizationDecision,
    TokenOptimizationMechanism,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
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
    TokenOptimizationCounterSnapshot,
    TokenOptimizationTelemetryEventType,
    TokenOptimizationTelemetryPayload,
    TokenOptimizationTelemetrySummary,
    TokenOptimizationTelemetrySummaryValidationStatus,
    TokenOptimizationTelemetryValidationStatus,
    build_token_optimization_counter_snapshot,
    build_token_optimization_telemetry_summary,
    build_token_savings_telemetry_payload,
    token_optimization_summary_to_attributes,
    token_savings_payload_to_attributes,
    validate_token_optimization_telemetry_summary,
    validate_token_savings_telemetry_payload,
)
from intergrax.runtime.token_optimization.tool_schema import optimize_tool_schema_catalog

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


def _sample_catalog() -> dict[str, object]:
    return {
        "tools": [
            {
                "name": "search_files",
                "description": "  Search   the   workspace  ",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
    }


def test_counter_snapshot_from_measured_receipt() -> None:
    receipt = _receipt_with_measurement()
    snapshot = build_token_optimization_counter_snapshot(receipts=[receipt])

    assert snapshot.total_receipts == 1
    assert snapshot.applied_count == 1
    assert snapshot.receipts_with_measurement_count == 1
    assert snapshot.baseline_tokens == 100
    assert snapshot.optimized_tokens == 75
    assert snapshot.saved_tokens == 25
    assert snapshot.saved_ratio == pytest.approx(0.25)


def test_counter_snapshot_aggregates_multiple_receipts() -> None:
    receipt_one = _receipt_with_measurement()
    receipt_two = build_compression_receipt(
        original_content='{"a": 1}',
        optimized_content='{"a":1}',
        request=TokenOptimizationRequest(
            content='{"a": 1}',
            source_type=TokenOptimizationSourceType.STRUCTURED_DATA,
            strategy=_strategy(),
        ),
        result=TokenOptimizationResult(
            content='{"a":1}',
            decision=TokenOptimizationDecision.APPLY,
            measurement=TokenSavingsMeasurement(
                baseline_tokens=20,
                optimized_tokens=10,
                saved_tokens=10,
                saved_ratio=0.5,
                confidence=TokenSavingsClaimConfidence.MEASURED,
                category=TokenCategory.INPUT_CONTEXT,
                source_type=TokenOptimizationSourceType.STRUCTURED_DATA,
            ),
        ),
        receipt_id="receipt-telemetry-2",
        created_at="2026-07-01T12:00:00+00:00",
    )
    snapshot = build_token_optimization_counter_snapshot(receipts=[receipt_one, receipt_two])

    assert snapshot.total_receipts == 2
    assert snapshot.applied_count == 2
    assert snapshot.baseline_tokens == 120
    assert snapshot.optimized_tokens == 85
    assert snapshot.saved_tokens == 35


def test_duplicate_receipt_id_is_not_double_counted() -> None:
    receipt = _receipt_with_measurement()
    duplicate = CompressionReceipt(
        receipt_id=receipt.receipt_id,
        created_at=receipt.created_at,
        source_type=receipt.source_type,
        decision=receipt.decision,
        original_hash=receipt.original_hash,
        optimized_hash=receipt.optimized_hash,
        measurement=receipt.measurement,
    )
    tool_outcome = optimize_tool_schema_catalog(
        _sample_catalog(),
        token_policy=_enabled_policy(),
        token_counter=len,
    )
    assert tool_outcome.receipt is not None
    snapshot = build_token_optimization_counter_snapshot(
        receipts=[receipt, duplicate, tool_outcome.receipt],
        tool_schema_outcomes=[tool_outcome],
    )

    assert snapshot.total_receipts == 2


def test_counter_snapshot_counts_tool_schema_changed() -> None:
    outcome = optimize_tool_schema_catalog(_sample_catalog(), token_policy=_enabled_policy())
    snapshot = build_token_optimization_counter_snapshot(tool_schema_outcomes=[outcome])

    assert snapshot.total_tool_schema_outcomes == 1
    assert snapshot.tool_schema_changed_count == 1


def test_counter_snapshot_counts_context_pack_changed() -> None:
    from intergrax.runtime.token_optimization.context_pack import ContextFragment

    fragments = [
        ContextFragment(
            fragment_id="frag_1",
            content="  spaced   content  ",
            source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        )
    ]
    outcome = optimize_context_pack(fragments, token_policy=_enabled_policy())
    snapshot = build_token_optimization_counter_snapshot(context_pack_outcomes=[outcome])

    assert snapshot.total_context_pack_outcomes == 1
    assert snapshot.context_pack_changed_count == 1


def test_counter_snapshot_includes_receipt_measurement_from_outcome() -> None:
    def counter(text: str) -> int:
        return len(text)

    outcome = optimize_tool_schema_catalog(
        _sample_catalog(),
        token_policy=_enabled_policy(),
        token_counter=counter,
    )
    snapshot = build_token_optimization_counter_snapshot(tool_schema_outcomes=[outcome])

    assert snapshot.total_receipts == 1
    assert snapshot.receipts_with_measurement_count == 1
    assert snapshot.baseline_tokens > 0
    assert snapshot.saved_tokens > 0


def test_counter_snapshot_without_measurement_has_zero_token_totals() -> None:
    receipt = CompressionReceipt(
        receipt_id="receipt-no-measurement",
        created_at="2026-07-01T12:00:00+00:00",
        source_type=TokenOptimizationSourceType.TOOL_CATALOG,
        decision=TokenOptimizationDecision.BYPASS,
        original_hash="abc",
        optimized_hash="def",
        bypass_reason=TokenOptimizationBypassReason.DISABLED,
    )
    snapshot = build_token_optimization_counter_snapshot(receipts=[receipt])

    assert snapshot.baseline_tokens == 0
    assert snapshot.optimized_tokens == 0
    assert snapshot.saved_tokens == 0
    assert snapshot.saved_ratio == 0.0


def test_telemetry_summary_event_type_is_summary() -> None:
    summary = build_token_optimization_telemetry_summary(
        receipts=[_receipt_with_measurement()],
    )
    assert summary.event_type is TokenOptimizationTelemetryEventType.TOKEN_OPTIMIZATION_SUMMARY
    assert summary.event_type.value == "token_optimization.summary"


def test_telemetry_summary_includes_workflow_id_when_provided() -> None:
    summary = build_token_optimization_telemetry_summary(
        receipts=[_receipt_with_measurement()],
        workflow_id="workflow-telemetry-1",
    )
    assert summary.workflow_id == "workflow-telemetry-1"


def test_telemetry_summary_excludes_raw_content() -> None:
    summary = build_token_optimization_telemetry_summary(
        receipts=[_receipt_with_measurement()],
        metadata={"proof_case": "safe", "original_content": "must-not-appear"},
    )
    serialized = str(summary.metadata) + str(summary.snapshot.metadata)
    assert _ORIGINAL not in serialized
    assert _OPTIMIZED not in serialized
    assert "original_content" not in summary.metadata
    assert summary.metadata.get("proof_case") == "safe"


def test_summary_metadata_keeps_allowlisted_safe_scalar_keys() -> None:
    summary = build_token_optimization_telemetry_summary(
        receipts=[_receipt_with_measurement()],
        metadata={
            "proof_case": "tool_catalog",
            "run_id": "run-1",
            "agent_id": "agent-1",
            "tenant_id": "tenant-1",
        },
    )
    assert summary.metadata["proof_case"] == "tool_catalog"
    assert summary.metadata["run_id"] == "run-1"
    assert summary.metadata["agent_id"] == "agent-1"
    assert summary.metadata["tenant_id"] == "tenant-1"


def test_summary_metadata_drops_non_allowlisted_neutral_keys() -> None:
    summary = build_token_optimization_telemetry_summary(
        receipts=[_receipt_with_measurement()],
        metadata={"note": "harmless note", "comment": "debug info"},
    )
    assert "note" not in summary.metadata
    assert "comment" not in summary.metadata


def test_summary_metadata_drops_forbidden_keys() -> None:
    summary = build_token_optimization_telemetry_summary(
        receipts=[_receipt_with_measurement()],
        metadata={"original_content": "secret", "raw_context": "ctx"},
    )
    assert "original_content" not in summary.metadata
    assert "raw_context" not in summary.metadata


def test_summary_metadata_drops_long_string_values() -> None:
    long_value = "x" * 200
    summary = build_token_optimization_telemetry_summary(
        receipts=[_receipt_with_measurement()],
        metadata={"proof_case": long_value},
    )
    assert "proof_case" not in summary.metadata


def test_summary_metadata_drops_multiline_string_values() -> None:
    summary = build_token_optimization_telemetry_summary(
        receipts=[_receipt_with_measurement()],
        metadata={"proof_case": "line-one\nline-two"},
    )
    assert "proof_case" not in summary.metadata


def test_summary_metadata_drops_non_scalar_values() -> None:
    summary = build_token_optimization_telemetry_summary(
        receipts=[_receipt_with_measurement()],
        metadata={"proof_case": {"nested": "dict"}},
    )
    assert "proof_case" not in summary.metadata


def test_summary_metadata_does_not_pass_raw_content_under_neutral_key() -> None:
    raw_content = '{"large":"raw context or schema content that should not be exported"}'
    summary = build_token_optimization_telemetry_summary(
        receipts=[_receipt_with_measurement()],
        metadata={"note": raw_content},
    )
    attributes = token_optimization_summary_to_attributes(summary)

    assert "note" not in summary.metadata
    assert "intergrax.token_optimization.metadata.note" not in attributes
    assert raw_content not in str(summary.metadata)
    assert raw_content not in str(attributes)


def test_summary_metadata_allows_safe_proof_case() -> None:
    summary = build_token_optimization_telemetry_summary(
        receipts=[_receipt_with_measurement()],
        metadata={"proof_case": "tool_catalog"},
    )
    attributes = token_optimization_summary_to_attributes(summary)

    assert summary.metadata["proof_case"] == "tool_catalog"
    assert attributes["intergrax.token_optimization.metadata.proof_case"] == "tool_catalog"


def test_summary_attributes_do_not_expose_dropped_metadata() -> None:
    summary = build_token_optimization_telemetry_summary(
        receipts=[_receipt_with_measurement()],
        metadata={"note": "should-not-export", "proof_case": "safe"},
    )
    attributes = token_optimization_summary_to_attributes(summary)

    assert "intergrax.token_optimization.metadata.note" not in attributes
    assert attributes["intergrax.token_optimization.metadata.proof_case"] == "safe"


def test_validate_telemetry_summary_fails_for_non_allowlisted_metadata_key() -> None:
    summary = build_token_optimization_telemetry_summary(receipts=[_receipt_with_measurement()])
    invalid = TokenOptimizationTelemetrySummary(
        event_type=summary.event_type,
        snapshot=summary.snapshot,
        metadata={"note": "unsafe"},
    )
    outcome = validate_token_optimization_telemetry_summary(invalid)
    assert outcome.status is TokenOptimizationTelemetrySummaryValidationStatus.FAILED
    assert "metadata key is not allowed: note" in outcome.failures


def test_validate_telemetry_summary_fails_for_long_metadata_string() -> None:
    summary = build_token_optimization_telemetry_summary(receipts=[_receipt_with_measurement()])
    invalid = TokenOptimizationTelemetrySummary(
        event_type=summary.event_type,
        snapshot=summary.snapshot,
        metadata={"proof_case": "x" * 200},
    )
    outcome = validate_token_optimization_telemetry_summary(invalid)
    assert outcome.status is TokenOptimizationTelemetrySummaryValidationStatus.FAILED
    assert "metadata string value is too long: proof_case" in outcome.failures


def test_validate_telemetry_summary_fails_for_multiline_metadata_string() -> None:
    summary = build_token_optimization_telemetry_summary(receipts=[_receipt_with_measurement()])
    invalid = TokenOptimizationTelemetrySummary(
        event_type=summary.event_type,
        snapshot=summary.snapshot,
        metadata={"proof_case": "line-one\nline-two"},
    )
    outcome = validate_token_optimization_telemetry_summary(invalid)
    assert outcome.status is TokenOptimizationTelemetrySummaryValidationStatus.FAILED
    assert "metadata string value must be single-line: proof_case" in outcome.failures


def test_validate_telemetry_summary_fails_for_non_scalar_metadata_value() -> None:
    summary = build_token_optimization_telemetry_summary(receipts=[_receipt_with_measurement()])
    invalid = TokenOptimizationTelemetrySummary(
        event_type=summary.event_type,
        snapshot=summary.snapshot,
        metadata={"proof_case": ["list", "value"]},
    )
    outcome = validate_token_optimization_telemetry_summary(invalid)
    assert outcome.status is TokenOptimizationTelemetrySummaryValidationStatus.FAILED
    assert "metadata value is not safe scalar: proof_case" in outcome.failures


def test_validate_telemetry_summary_fails_for_forbidden_metadata_key() -> None:
    summary = build_token_optimization_telemetry_summary(receipts=[_receipt_with_measurement()])
    invalid = TokenOptimizationTelemetrySummary(
        event_type=summary.event_type,
        snapshot=summary.snapshot,
        metadata={"original_content": "secret"},
    )
    outcome = validate_token_optimization_telemetry_summary(invalid)
    assert outcome.status is TokenOptimizationTelemetrySummaryValidationStatus.FAILED
    assert "metadata key is not allowed: original_content" in outcome.failures


def test_validate_telemetry_summary_passes_for_valid_summary() -> None:
    summary = build_token_optimization_telemetry_summary(
        receipts=[_receipt_with_measurement()],
    )
    outcome = validate_token_optimization_telemetry_summary(summary)
    assert outcome.status is TokenOptimizationTelemetrySummaryValidationStatus.PASSED
    assert outcome.failures == ()


def test_validate_telemetry_summary_fails_for_negative_counters() -> None:
    summary = build_token_optimization_telemetry_summary(receipts=[_receipt_with_measurement()])
    invalid = TokenOptimizationTelemetrySummary(
        event_type=summary.event_type,
        workflow_id=summary.workflow_id,
        snapshot=TokenOptimizationCounterSnapshot(
            total_receipts=-1,
            baseline_tokens=summary.snapshot.baseline_tokens,
            optimized_tokens=summary.snapshot.optimized_tokens,
            saved_tokens=summary.snapshot.saved_tokens,
            saved_ratio=summary.snapshot.saved_ratio,
        ),
        receipt_ids=summary.receipt_ids,
    )
    outcome = validate_token_optimization_telemetry_summary(invalid)
    assert outcome.status is TokenOptimizationTelemetrySummaryValidationStatus.FAILED
    assert "total_receipts must not be negative" in outcome.failures


def test_validate_telemetry_summary_fails_for_inconsistent_saved_tokens() -> None:
    summary = build_token_optimization_telemetry_summary(receipts=[_receipt_with_measurement()])
    invalid = TokenOptimizationTelemetrySummary(
        event_type=summary.event_type,
        workflow_id=summary.workflow_id,
        snapshot=TokenOptimizationCounterSnapshot(
            total_receipts=1,
            receipts_with_measurement_count=1,
            baseline_tokens=100,
            optimized_tokens=75,
            saved_tokens=10,
            saved_ratio=0.25,
        ),
        receipt_ids=summary.receipt_ids,
    )
    outcome = validate_token_optimization_telemetry_summary(invalid)
    assert outcome.status is TokenOptimizationTelemetrySummaryValidationStatus.FAILED
    assert "saved_tokens must equal baseline_tokens - optimized_tokens" in outcome.failures


def test_validate_telemetry_summary_fails_for_duplicate_receipt_ids() -> None:
    summary = build_token_optimization_telemetry_summary(receipts=[_receipt_with_measurement()])
    invalid = TokenOptimizationTelemetrySummary(
        event_type=summary.event_type,
        snapshot=summary.snapshot,
        receipt_ids=("receipt-telemetry-1", "receipt-telemetry-1"),
    )
    outcome = validate_token_optimization_telemetry_summary(invalid)
    assert outcome.status is TokenOptimizationTelemetrySummaryValidationStatus.FAILED
    assert "receipt_ids must be unique" in outcome.failures


def test_summary_attributes_are_namespaced_and_scalar_only() -> None:
    summary = build_token_optimization_telemetry_summary(
        receipts=[_receipt_with_measurement()],
        workflow_id="workflow-attrs",
    )
    attributes = token_optimization_summary_to_attributes(summary)

    assert attributes["intergrax.token_optimization.event_type"] == "token_optimization.summary"
    assert attributes["intergrax.token_optimization.workflow_id"] == "workflow-attrs"
    assert attributes["intergrax.token_optimization.total_receipts"] == 1
    assert attributes["intergrax.token_optimization.applied_count"] == 1
    assert attributes["intergrax.token_optimization.baseline_tokens"] == 100
    assert attributes["intergrax.token_optimization.saved_tokens"] == 25
    for value in attributes.values():
        assert isinstance(value, (str, int, float, bool, type(None)))


def test_integration_summary_from_real_tool_schema_and_context_pack_outcomes() -> None:
    from intergrax.runtime.token_optimization.context_pack import ContextFragment

    def counter(text: str) -> int:
        return len(text)

    tool_outcome = optimize_tool_schema_catalog(
        _sample_catalog(),
        token_policy=_enabled_policy(),
        token_counter=counter,
    )
    context_outcome = optimize_context_pack(
        [
            ContextFragment(
                fragment_id="frag_1",
                content="  spaced   evidence  ",
                source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
            )
        ],
        token_policy=_enabled_policy(),
        token_counter=counter,
    )
    summary = build_token_optimization_telemetry_summary(
        tool_schema_outcomes=[tool_outcome],
        context_pack_outcomes=[context_outcome],
        workflow_id="workflow-integration",
    )
    attributes = token_optimization_summary_to_attributes(summary)

    assert summary.snapshot.saved_tokens > 0
    assert attributes["intergrax.token_optimization.saved_tokens"] > 0
    serialized = str(summary.metadata) + str(summary.snapshot.metadata) + str(attributes)
    assert tool_outcome.original_content not in serialized
    assert tool_outcome.optimized_content not in serialized
    assert context_outcome.original_content not in serialized
    assert context_outcome.optimized_content not in serialized
    for forbidden in ("original_content", "optimized_content", "raw_context", "raw_prompt"):
        assert forbidden not in summary.metadata
        assert forbidden not in attributes
