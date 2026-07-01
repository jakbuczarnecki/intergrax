# © Artur Czarnecki. All rights reserved.

"""TOKEN-1A: Token Optimization contract validation tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.token_optimization.contracts import (
    CompressionLevel,
    OutputProfile,
    ProtectedRegionKind,
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
    StrategySafetyClass,
    TokenCategory,
    TokenOptimizationAttribution,
    TokenOptimizationBypassReason,
    TokenOptimizationDecision,
    TokenOptimizationMechanism,
    TokenOptimizationPluginCapability,
    TokenOptimizationPluginDescriptor,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationRequest,
    TokenOptimizationResult,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
    TokenOptimizationStrategyRef,
    TokenSavingsClaimConfidence,
    TokenSavingsMeasurement,
    TokenUsageMeasurement,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_token_optimization_policy_default_safe_values() -> None:
    policy = TokenOptimizationPolicy()
    assert policy.enabled is False
    assert policy.profile is TokenOptimizationProfile.OFF
    assert policy.compression_level is CompressionLevel.OFF
    assert policy.allow_lossy is False
    assert policy.require_validation is True
    assert policy.fallback_on_validation_failure is True
    assert policy.emit_receipts is True
    assert policy.emit_observability is True


def test_attribution_carries_run_step_model_provider_source_category() -> None:
    attribution = TokenOptimizationAttribution(
        run_id="run-1",
        step_id="step-2",
        model="gpt-test",
        provider="openai",
        source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
        token_category=TokenCategory.INPUT_CONTEXT,
        optimization_profile=TokenOptimizationProfile.CONSERVATIVE,
    )
    assert attribution.run_id == "run-1"
    assert attribution.step_id == "step-2"
    assert attribution.model == "gpt-test"
    assert attribution.provider == "openai"
    assert attribution.source_type is TokenOptimizationSourceType.TOOL_OUTPUT
    assert attribution.token_category is TokenCategory.INPUT_CONTEXT


def test_savings_measurement_validates_saved_tokens_and_ratio() -> None:
    measurement = TokenSavingsMeasurement(
        baseline_tokens=100,
        optimized_tokens=75,
        saved_tokens=25,
        saved_ratio=0.25,
        confidence=TokenSavingsClaimConfidence.MEASURED,
        category=TokenCategory.TOOL_CATALOG,
        source_type=TokenOptimizationSourceType.TOOL_CATALOG,
    )
    assert measurement.saved_tokens == 25
    assert measurement.saved_ratio == pytest.approx(0.25)


def test_savings_measurement_rejects_mismatched_saved_tokens() -> None:
    with pytest.raises(ValueError, match="saved_tokens must equal"):
        TokenSavingsMeasurement(
            baseline_tokens=100,
            optimized_tokens=80,
            saved_tokens=10,
            saved_ratio=0.20,
            confidence=TokenSavingsClaimConfidence.MEASURED,
            category=TokenCategory.TOTAL,
            source_type=TokenOptimizationSourceType.UNKNOWN,
        )


def test_savings_measurement_rejects_invalid_saved_ratio() -> None:
    with pytest.raises(ValueError, match="saved_ratio must be between"):
        TokenSavingsMeasurement(
            baseline_tokens=100,
            optimized_tokens=50,
            saved_tokens=50,
            saved_ratio=1.5,
            confidence=TokenSavingsClaimConfidence.ESTIMATED,
            category=TokenCategory.TOTAL,
            source_type=TokenOptimizationSourceType.UNKNOWN,
        )


def test_negative_token_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="tokens cannot be negative"):
        TokenUsageMeasurement(
            tokens=-1,
            category=TokenCategory.OUTPUT,
            source_type=TokenOptimizationSourceType.OUTPUT,
        )

    with pytest.raises(ValueError, match="baseline_tokens cannot be negative"):
        TokenSavingsMeasurement(
            baseline_tokens=-1,
            optimized_tokens=0,
            saved_tokens=-1,
            saved_ratio=0.0,
            confidence=TokenSavingsClaimConfidence.NOT_COMPARABLE,
            category=TokenCategory.TOTAL,
            source_type=TokenOptimizationSourceType.UNKNOWN,
        )


def test_plugin_descriptor_declares_custom_optimizer_capability() -> None:
    capability = TokenOptimizationPluginCapability(
        mechanism=TokenOptimizationMechanism.STRUCTURED_DATA_COMPRESSION,
        strategy_kind=TokenOptimizationStrategyKind.LOSSLESS_STRUCTURAL_COMPRESSION,
        source_types=(TokenOptimizationSourceType.STRUCTURED_DATA,),
        lossless=True,
        requires_validation=True,
    )
    descriptor = TokenOptimizationPluginDescriptor(
        plugin_id="acme.optimizer",
        name="Acme Optimizer",
        version="0.1.0",
        capabilities=(capability,),
    )
    assert descriptor.plugin_id == "acme.optimizer"
    assert descriptor.capabilities[0].lossless is True
    assert descriptor.capabilities[0].source_types == (
        TokenOptimizationSourceType.STRUCTURED_DATA,
    )


def test_request_and_result_construct_without_execution() -> None:
    strategy = TokenOptimizationStrategyRef(
        strategy_id="tool_catalog.minimize",
        mechanism=TokenOptimizationMechanism.TOOL_CATALOG_COMPACTION,
        kind=TokenOptimizationStrategyKind.SCHEMA_MINIMIZATION,
        safety_class=StrategySafetyClass.LOSSLESS,
    )
    request = TokenOptimizationRequest(
        content='{"tools": []}',
        source_type=TokenOptimizationSourceType.TOOL_CATALOG,
        strategy=strategy,
    )
    result = TokenOptimizationResult(
        content=request.content,
        decision=TokenOptimizationDecision.BYPASS,
        strategy=strategy,
        bypass_reason=TokenOptimizationBypassReason.DISABLED,
    )
    assert request.policy.enabled is False
    assert result.decision is TokenOptimizationDecision.BYPASS
    assert result.bypass_reason is TokenOptimizationBypassReason.DISABLED


def test_protected_region_validation_result_pass_and_fail_states() -> None:
    passed = ProtectedRegionValidationResult(
        status=ProtectedRegionValidationStatus.PASSED,
        regions_checked=3,
        regions_preserved=3,
        regions_failed=0,
    )
    failed = ProtectedRegionValidationResult(
        status=ProtectedRegionValidationStatus.FAILED,
        regions_checked=2,
        regions_preserved=1,
        regions_failed=1,
        failures=(f"missing {ProtectedRegionKind.PATH.value}",),
    )
    assert passed.status is ProtectedRegionValidationStatus.PASSED
    assert failed.status is ProtectedRegionValidationStatus.FAILED
    assert failed.regions_failed == 1
