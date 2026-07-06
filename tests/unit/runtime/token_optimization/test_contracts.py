# © Artur Czarnecki. All rights reserved.

"""Token Optimization contract validation tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.token_optimization.contracts import (
    CompressionLevel,
    ContextDeduplicationMetadata,
    ContextFragmentPackingMetadata,
    ContextFragmentPriority,
    ContextPackingBudget,
    ContextPackingDecision,
    ContextPackingDecisionKind,
    ContextPackingReceiptMetadata,
    OutputProfile,
    ProtectedRegionKind,
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
    StrategySafetyClass,
    TokenCategory,
    TokenOptimizationAttribution,
    TokenOptimizationBypassReason,
    TokenOptimizationDecision,
    TokenOptimizationLayerContext,
    TokenOptimizationLayerDecision,
    TokenOptimizationLayerDescriptor,
    TokenOptimizationLayerRef,
    TokenOptimizationLayerRequest,
    TokenOptimizationLayerResult,
    TokenOptimizationMechanism,
    TokenOptimizationPipelineConfig,
    TokenOptimizationPipelineMode,
    TokenOptimizationPipelineResult,
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


def test_context_fragment_priority_enum_values() -> None:
    assert ContextFragmentPriority.MUST_KEEP.value == "must_keep"
    assert ContextFragmentPriority.HIGH_PRIORITY.value == "high_priority"
    assert ContextFragmentPriority.COMPRESSIBLE.value == "compressible"
    assert ContextFragmentPriority.DROPPABLE.value == "droppable"


def test_context_packing_budget_accepts_valid_budgets() -> None:
    budget = ContextPackingBudget(
        max_input_tokens=8000,
        reserved_output_tokens=1000,
        target_context_tokens=6000,
        hard_context_limit=7000,
    )
    assert budget.max_input_tokens == 8000
    assert budget.target_context_tokens == 6000


def test_context_packing_budget_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="max_input_tokens cannot be negative"):
        ContextPackingBudget(max_input_tokens=-1)


def test_context_packing_budget_rejects_target_exceeding_hard_limit() -> None:
    with pytest.raises(ValueError, match="target_context_tokens cannot exceed"):
        ContextPackingBudget(target_context_tokens=9000, hard_context_limit=8000)


def test_context_packing_budget_rejects_reserved_output_exceeding_max_input() -> None:
    with pytest.raises(ValueError, match="reserved_output_tokens cannot exceed"):
        ContextPackingBudget(max_input_tokens=1000, reserved_output_tokens=1500)


def test_context_packing_decision_validates_fragment_id() -> None:
    with pytest.raises(ValueError, match="fragment_id cannot be empty"):
        ContextPackingDecision(
            fragment_id="",
            decision=ContextPackingDecisionKind.KEEP,
            priority=ContextFragmentPriority.MUST_KEEP,
        )


def test_context_packing_decision_validates_token_math() -> None:
    decision = ContextPackingDecision(
        fragment_id="frag-1",
        decision=ContextPackingDecisionKind.COMPACT,
        priority=ContextFragmentPriority.COMPRESSIBLE,
        original_tokens=100,
        optimized_tokens=80,
        saved_tokens=20,
    )
    assert decision.saved_tokens == 20

    with pytest.raises(ValueError, match="saved_tokens must equal"):
        ContextPackingDecision(
            fragment_id="frag-1",
            decision=ContextPackingDecisionKind.COMPACT,
            priority=ContextFragmentPriority.COMPRESSIBLE,
            original_tokens=100,
            optimized_tokens=80,
            saved_tokens=10,
        )


def test_context_deduplication_metadata_rejects_empty_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate_fragment_ids cannot contain"):
        ContextDeduplicationMetadata(duplicate_fragment_ids=("dup-1", ""))

    with pytest.raises(ValueError, match="dedupe_key cannot be empty"):
        ContextDeduplicationMetadata(dedupe_key="")


def test_context_fragment_packing_metadata_rejects_required_and_droppable() -> None:
    with pytest.raises(ValueError, match="required fragments cannot have droppable"):
        ContextFragmentPackingMetadata(
            required=True,
            priority=ContextFragmentPriority.DROPPABLE,
        )


def test_context_fragment_packing_metadata_rejects_protected_and_droppable() -> None:
    with pytest.raises(ValueError, match="protected fragments cannot have droppable"):
        ContextFragmentPackingMetadata(
            protected=True,
            priority=ContextFragmentPriority.DROPPABLE,
        )


def test_context_packing_receipt_metadata_validates_totals_and_strategy_breakdown() -> None:
    receipt = ContextPackingReceiptMetadata(
        total_original_tokens=200,
        total_optimized_tokens=150,
        total_saved_tokens=50,
        strategy_breakdown={"context_pack.compact": 30, "context_pack.drop": 20},
    )
    assert receipt.total_saved_tokens == 50

    with pytest.raises(ValueError, match="total_saved_tokens must equal"):
        ContextPackingReceiptMetadata(
            total_original_tokens=200,
            total_optimized_tokens=150,
            total_saved_tokens=40,
        )

    with pytest.raises(ValueError, match="strategy_breakdown"):
        ContextPackingReceiptMetadata(strategy_breakdown={"context_pack.drop": -1})


def _sample_strategy() -> TokenOptimizationStrategyRef:
    return TokenOptimizationStrategyRef(
        strategy_id="builtin.structural_compaction",
        mechanism=TokenOptimizationMechanism.RAG_CONTEXT_PACK_COMPRESSION,
        kind=TokenOptimizationStrategyKind.LOSSLESS_STRUCTURAL_COMPRESSION,
        safety_class=StrategySafetyClass.LOSSLESS,
    )


def test_token_optimization_layer_decision_enum_values() -> None:
    assert TokenOptimizationLayerDecision.APPLY.value == "apply"
    assert TokenOptimizationLayerDecision.BYPASS.value == "bypass"
    assert TokenOptimizationLayerDecision.FALLBACK.value == "fallback"
    assert TokenOptimizationLayerDecision.OVERRIDE_PREVIOUS.value == "override_previous"
    assert TokenOptimizationLayerDecision.REVERT_TO_ORIGINAL.value == "revert_to_original"
    assert TokenOptimizationLayerDecision.FAILED.value == "failed"


def test_token_optimization_layer_descriptor_validates_required_fields() -> None:
    strategy = _sample_strategy()
    descriptor = TokenOptimizationLayerDescriptor(
        layer_id="builtin.structural_compaction",
        name="Structural Compaction",
        version="1.0.0",
        strategy=strategy,
        built_in=True,
        supported_source_types=(TokenOptimizationSourceType.RAG_CONTEXT_PACK,),
    )
    assert descriptor.built_in is True
    assert descriptor.strategy.strategy_id == "builtin.structural_compaction"

    with pytest.raises(ValueError, match="layer_id cannot be empty"):
        TokenOptimizationLayerDescriptor(
            layer_id="",
            name="Structural Compaction",
            version="1.0.0",
            strategy=strategy,
        )

    with pytest.raises(ValueError, match="name cannot be empty"):
        TokenOptimizationLayerDescriptor(
            layer_id="builtin.structural_compaction",
            name="",
            version="1.0.0",
            strategy=strategy,
        )

    with pytest.raises(ValueError, match="version cannot be empty"):
        TokenOptimizationLayerDescriptor(
            layer_id="builtin.structural_compaction",
            name="Structural Compaction",
            version="",
            strategy=strategy,
        )

    with pytest.raises(ValueError, match="plugin_id cannot be empty"):
        TokenOptimizationLayerDescriptor(
            layer_id="custom.company.domain_dedupe",
            name="Domain Dedupe",
            version="0.1.0",
            strategy=strategy,
            plugin_id="",
        )


def test_token_optimization_layer_context_rejects_negative_index_and_empty_ids() -> None:
    context = TokenOptimizationLayerContext(
        pipeline_id="default",
        layer_index=0,
        previous_layer_ids=("builtin.dedupe",),
        applied_layer_ids=("builtin.dedupe",),
    )
    assert context.layer_index == 0

    with pytest.raises(ValueError, match="layer_index cannot be negative"):
        TokenOptimizationLayerContext(layer_index=-1)

    with pytest.raises(ValueError, match="previous_layer_ids cannot contain"):
        TokenOptimizationLayerContext(previous_layer_ids=("layer-1", ""))

    with pytest.raises(ValueError, match="applied_layer_ids cannot contain"):
        TokenOptimizationLayerContext(applied_layer_ids=("",))


def test_token_optimization_layer_request_accepts_original_and_current_content() -> None:
    request = TokenOptimizationLayerRequest(
        original_content="baseline text",
        current_content="baseline text",
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
    )
    assert request.original_content == request.current_content

    request_after_layer = TokenOptimizationLayerRequest(
        original_content="baseline text",
        current_content="compacted text",
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
    )
    assert request_after_layer.original_content != request_after_layer.current_content

    empty_request = TokenOptimizationLayerRequest(
        original_content="",
        current_content="",
        source_type=TokenOptimizationSourceType.UNKNOWN,
    )
    assert empty_request.original_content == ""
    assert empty_request.current_content == ""


def test_token_optimization_layer_result_rejects_empty_layer_id() -> None:
    with pytest.raises(ValueError, match="layer_id cannot be empty"):
        TokenOptimizationLayerResult(
            layer_id="",
            output_content="output",
            decision=TokenOptimizationLayerDecision.APPLY,
        )


def test_token_optimization_layer_result_requires_override_metadata_for_override_previous() -> None:
    result = TokenOptimizationLayerResult(
        layer_id="custom.override",
        output_content="restored",
        decision=TokenOptimizationLayerDecision.OVERRIDE_PREVIOUS,
        previous_changes_overridden=True,
        overridden_layer_ids=("builtin.structural_compaction",),
        override_reason="prior compaction harmed protected regions",
    )
    assert result.previous_changes_overridden is True

    with pytest.raises(ValueError, match="previous_changes_overridden must be True"):
        TokenOptimizationLayerResult(
            layer_id="custom.override",
            output_content="restored",
            decision=TokenOptimizationLayerDecision.OVERRIDE_PREVIOUS,
            previous_changes_overridden=False,
        )

    with pytest.raises(ValueError, match="override_reason should be provided"):
        TokenOptimizationLayerResult(
            layer_id="custom.override",
            output_content="restored",
            decision=TokenOptimizationLayerDecision.APPLY,
            previous_changes_overridden=True,
        )


def test_token_optimization_layer_result_requires_override_metadata_for_revert_to_original() -> None:
    result = TokenOptimizationLayerResult(
        layer_id="custom.revert",
        output_content="baseline text",
        decision=TokenOptimizationLayerDecision.REVERT_TO_ORIGINAL,
        previous_changes_overridden=True,
        override_reason="reverted to immutable baseline",
    )
    assert result.decision is TokenOptimizationLayerDecision.REVERT_TO_ORIGINAL

    with pytest.raises(ValueError, match="previous_changes_overridden must be True"):
        TokenOptimizationLayerResult(
            layer_id="custom.revert",
            output_content="baseline text",
            decision=TokenOptimizationLayerDecision.REVERT_TO_ORIGINAL,
            previous_changes_overridden=False,
        )


def test_token_optimization_layer_ref_validates_layer_id_and_order() -> None:
    layer_ref = TokenOptimizationLayerRef(
        layer_id="builtin.priority_classification",
        order=2,
        plugin_id="acme.optimizer",
        version="1.0.0",
    )
    assert layer_ref.layer_id == "builtin.priority_classification"
    assert layer_ref.order == 2

    with pytest.raises(ValueError, match="layer_id cannot be empty"):
        TokenOptimizationLayerRef(layer_id="")

    with pytest.raises(ValueError, match="order cannot be negative"):
        TokenOptimizationLayerRef(layer_id="builtin.dedupe", order=-1)


def test_token_optimization_pipeline_mode_enum_values() -> None:
    assert TokenOptimizationPipelineMode.DEFAULT.value == "default"
    assert TokenOptimizationPipelineMode.REPLACE.value == "replace"


def test_token_optimization_pipeline_config_accepts_default_with_empty_layers() -> None:
    config = TokenOptimizationPipelineConfig(pipeline_id="platform-default")
    assert config.mode is TokenOptimizationPipelineMode.DEFAULT
    assert config.layers == ()


def test_token_optimization_pipeline_config_rejects_replace_with_empty_layers() -> None:
    with pytest.raises(ValueError, match="layers must not be empty when mode is REPLACE"):
        TokenOptimizationPipelineConfig(
            pipeline_id="custom-pipeline",
            mode=TokenOptimizationPipelineMode.REPLACE,
            layers=(),
        )


def test_token_optimization_pipeline_config_rejects_duplicate_enabled_layers() -> None:
    layers = (
        TokenOptimizationLayerRef(layer_id="builtin.dedupe"),
        TokenOptimizationLayerRef(layer_id="builtin.dedupe"),
    )
    with pytest.raises(ValueError, match="enabled layer_id values must be unique"):
        TokenOptimizationPipelineConfig(
            pipeline_id="custom-pipeline",
            mode=TokenOptimizationPipelineMode.REPLACE,
            layers=layers,
        )

    config_with_repeat_allowed = TokenOptimizationPipelineConfig(
        pipeline_id="custom-pipeline",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=layers,
        allow_repeated_layers=True,
    )
    assert len(config_with_repeat_allowed.layers) == 2

    config_with_disabled_duplicate = TokenOptimizationPipelineConfig(
        pipeline_id="custom-pipeline",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(
            TokenOptimizationLayerRef(layer_id="builtin.dedupe", enabled=True),
            TokenOptimizationLayerRef(layer_id="builtin.dedupe", enabled=False),
        ),
    )
    assert len(config_with_disabled_duplicate.layers) == 2


def test_token_optimization_pipeline_result_validates_pipeline_id_and_ids() -> None:
    result = TokenOptimizationPipelineResult(
        pipeline_id="platform-default",
        original_content="baseline",
        final_content="optimized",
        applied_layer_ids=("builtin.dedupe",),
        bypassed_layer_ids=("builtin.priority_classification",),
    )
    assert result.pipeline_id == "platform-default"

    with pytest.raises(ValueError, match="pipeline_id cannot be empty"):
        TokenOptimizationPipelineResult(
            pipeline_id="",
            original_content="baseline",
            final_content="optimized",
        )

    with pytest.raises(ValueError, match="applied_layer_ids cannot contain"):
        TokenOptimizationPipelineResult(
            pipeline_id="platform-default",
            original_content="baseline",
            final_content="optimized",
            applied_layer_ids=("",),
        )
