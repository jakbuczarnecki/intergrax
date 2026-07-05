# © Artur Czarnecki. All rights reserved.

"""TOKEN-OPT-3D: BudgetAwareContextPackingLayer unit tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.token_optimization.contracts import (
    ContextFragmentPriority,
    ContextPackingDecisionKind,
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
    StrategySafetyClass,
    TokenOptimizationBypassReason,
    TokenOptimizationLayerDecision,
    TokenOptimizationLayerRequest,
    TokenOptimizationMechanism,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
)
from intergrax.runtime.token_optimization.layers.budget_aware_packing import (
    BudgetAwareContextPackingLayer,
    BudgetAwareContextPackingLayerConfig,
    BudgetAwarePackingFragment,
    BudgetAwarePackingInput,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_LAYER_ID = "builtin.budget_aware_context_packing"


def _enabled_policy() -> TokenOptimizationPolicy:
    return TokenOptimizationPolicy(
        enabled=True,
        profile=TokenOptimizationProfile.CONSERVATIVE,
    )


def _fragment(
    fragment_id: str,
    content: str,
    priority: ContextFragmentPriority,
) -> BudgetAwarePackingFragment:
    return BudgetAwarePackingFragment(
        fragment_id=fragment_id,
        content=content,
        priority=priority,
    )


def _packing_input(
    *fragments: BudgetAwarePackingFragment,
) -> BudgetAwarePackingInput:
    return BudgetAwarePackingInput(fragments=fragments)


def _request(
    *,
    current_content: str = "assembled-current",
    packing_input: BudgetAwarePackingInput | None = None,
    metadata: dict[str, object] | None = None,
) -> TokenOptimizationLayerRequest:
    request_metadata: dict[str, object] = dict(metadata or {})
    if packing_input is not None:
        request_metadata["packing_input"] = packing_input
    return TokenOptimizationLayerRequest(
        original_content="original-baseline",
        current_content=current_content,
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        policy=_enabled_policy(),
        metadata=request_metadata,
    )


def _layer(
    max_chars: int,
    **kwargs: object,
) -> BudgetAwareContextPackingLayer:
    return BudgetAwareContextPackingLayer(
        config=BudgetAwareContextPackingLayerConfig(max_chars=max_chars, **kwargs),
    )


def test_budget_aware_context_packing_layer_exposes_descriptor() -> None:
    layer = _layer(100)
    descriptor = layer.descriptor

    assert descriptor.layer_id == _LAYER_ID
    assert descriptor.name == "Budget-Aware Context Packing"
    assert descriptor.version == "1"
    assert descriptor.strategy.mechanism is TokenOptimizationMechanism.RAG_CONTEXT_PACK_COMPRESSION
    assert descriptor.strategy.kind is TokenOptimizationStrategyKind.RANKING_PRUNING
    assert descriptor.safety_class is StrategySafetyClass.LOSSLESS
    assert descriptor.built_in is True
    assert descriptor.requires_validation is True
    assert descriptor.supported_source_types == (
        TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        TokenOptimizationSourceType.RETRIEVED_EVIDENCE,
    )


def test_config_rejects_max_chars_below_or_equal_zero() -> None:
    with pytest.raises(ValueError, match="max_chars must be > 0"):
        BudgetAwareContextPackingLayerConfig(max_chars=0)
    with pytest.raises(ValueError, match="max_chars must be > 0"):
        BudgetAwareContextPackingLayerConfig(max_chars=-1)


def test_fragment_rejects_empty_fragment_id() -> None:
    with pytest.raises(ValueError, match="fragment_id cannot be empty"):
        BudgetAwarePackingFragment(
            fragment_id="",
            content="text",
            priority=ContextFragmentPriority.MUST_KEEP,
        )


def test_packing_input_rejects_duplicate_fragment_id() -> None:
    fragment = _fragment("dup", "a", ContextFragmentPriority.MUST_KEEP)
    with pytest.raises(ValueError, match="fragment ids must be unique"):
        BudgetAwarePackingInput(fragments=(fragment, fragment))


def test_missing_packing_input_returns_bypass() -> None:
    layer = _layer(100)
    result = layer.optimize(_request())

    assert result.decision is TokenOptimizationLayerDecision.BYPASS
    assert result.output_content == "assembled-current"
    assert result.bypass_reason is TokenOptimizationBypassReason.NOT_APPLICABLE


def test_empty_fragments_returns_bypass() -> None:
    layer = _layer(100)
    result = layer.optimize(_request(packing_input=_packing_input()))

    assert result.decision is TokenOptimizationLayerDecision.BYPASS
    assert result.output_content == "assembled-current"
    assert result.bypass_reason is TokenOptimizationBypassReason.NOT_APPLICABLE


def test_must_keep_fragments_are_always_included() -> None:
    layer = _layer(50)
    packing_input = _packing_input(
        _fragment("mk1", "alpha", ContextFragmentPriority.MUST_KEEP),
        _fragment("drop1", "x" * 100, ContextFragmentPriority.DROPPABLE),
    )
    result = layer.optimize(_request(packing_input=packing_input))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.output_content == "alpha"
    assert "mk1" in {d["fragment_id"] for d in result.metadata["packing_decisions"]}


def test_high_priority_preferred_over_compressible_and_droppable() -> None:
    layer = _layer(12)
    packing_input = _packing_input(
        _fragment("comp", "compress", ContextFragmentPriority.COMPRESSIBLE),
        _fragment("high", "hi", ContextFragmentPriority.HIGH_PRIORITY),
        _fragment("drop", "dropme", ContextFragmentPriority.DROPPABLE),
        _fragment("mk", "mk", ContextFragmentPriority.MUST_KEEP),
    )
    result = layer.optimize(_request(packing_input=packing_input))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.output_content == "hi\nmk"
    decisions = {d["fragment_id"]: d for d in result.metadata["packing_decisions"]}
    assert decisions["high"]["decision"] == ContextPackingDecisionKind.KEEP.value
    assert decisions["comp"]["decision"] == ContextPackingDecisionKind.DROP.value
    assert decisions["drop"]["decision"] == ContextPackingDecisionKind.DROP.value


def test_droppable_dropped_first_under_budget_pressure() -> None:
    layer = _layer(
        20,
        include_droppable_when_budget_available=True,
    )
    packing_input = _packing_input(
        _fragment("mk", "must", ContextFragmentPriority.MUST_KEEP),
        _fragment("high", "priority", ContextFragmentPriority.HIGH_PRIORITY),
        _fragment("drop", "droppable", ContextFragmentPriority.DROPPABLE),
    )
    result = layer.optimize(_request(packing_input=packing_input))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert "droppable" not in result.output_content
    decisions = {d["fragment_id"]: d for d in result.metadata["packing_decisions"]}
    assert decisions["drop"]["decision"] == ContextPackingDecisionKind.DROP.value
    assert decisions["high"]["decision"] == ContextPackingDecisionKind.KEEP.value


def test_droppable_excluded_by_default_even_when_budget_remains() -> None:
    layer = _layer(100)
    packing_input = _packing_input(
        _fragment("mk", "keep", ContextFragmentPriority.MUST_KEEP),
        _fragment("drop", "optional", ContextFragmentPriority.DROPPABLE),
    )
    result = layer.optimize(_request(packing_input=packing_input))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.output_content == "keep"
    decisions = {d["fragment_id"]: d for d in result.metadata["packing_decisions"]}
    assert decisions["drop"]["decision"] == ContextPackingDecisionKind.DROP.value
    assert decisions["drop"]["reason"] == "droppable_excluded_by_default"


def test_droppable_included_when_include_droppable_when_budget_available_true() -> None:
    layer = _layer(100, include_droppable_when_budget_available=True)
    packing_input = _packing_input(
        _fragment("mk", "keep", ContextFragmentPriority.MUST_KEEP),
        _fragment("drop", "optional", ContextFragmentPriority.DROPPABLE),
    )
    result = layer.optimize(_request(packing_input=packing_input))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.output_content == "keep\noptional"
    decisions = {d["fragment_id"]: d for d in result.metadata["packing_decisions"]}
    assert decisions["drop"]["decision"] == ContextPackingDecisionKind.KEEP.value


def test_compressible_whitespace_compaction_used_under_pressure() -> None:
    layer = _layer(14)
    packing_input = _packing_input(
        _fragment("mk", "mk", ContextFragmentPriority.MUST_KEEP),
        _fragment(
            "comp",
            "  hello   world  ",
            ContextFragmentPriority.COMPRESSIBLE,
        ),
    )
    result = layer.optimize(_request(packing_input=packing_input))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.output_content == "mk\nhello world"
    decisions = {d["fragment_id"]: d for d in result.metadata["packing_decisions"]}
    assert decisions["comp"]["decision"] == ContextPackingDecisionKind.COMPACT.value
    assert decisions["comp"]["reason"] == "compressible_whitespace_compacted"
    assert result.metadata["compacted_fragment_count"] == 1


def test_compressible_dropped_if_still_does_not_fit_after_compaction() -> None:
    layer = _layer(10)
    packing_input = _packing_input(
        _fragment("mk", "must", ContextFragmentPriority.MUST_KEEP),
        _fragment(
            "comp",
            "  too   long   content  ",
            ContextFragmentPriority.COMPRESSIBLE,
        ),
    )
    result = layer.optimize(_request(packing_input=packing_input))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.output_content == "must"
    decisions = {d["fragment_id"]: d for d in result.metadata["packing_decisions"]}
    assert decisions["comp"]["decision"] == ContextPackingDecisionKind.DROP.value


def test_must_keep_over_budget_returns_fallback() -> None:
    layer = _layer(5)
    packing_input = _packing_input(
        _fragment("mk1", "must_keep", ContextFragmentPriority.MUST_KEEP),
        _fragment("high", "hi", ContextFragmentPriority.HIGH_PRIORITY),
    )
    current = "fallback-content"
    result = layer.optimize(
        _request(current_content=current, packing_input=packing_input),
    )

    assert result.decision is TokenOptimizationLayerDecision.FALLBACK
    assert result.fallback_used is True
    assert result.output_content == current
    assert result.metadata["fallback_reason"] == "must_keep_exceeds_char_budget"
    assert result.metadata["budget_unit"] == "chars"
    assert result.metadata["must_keep_chars"] > result.metadata["max_chars"]


def test_protected_region_validation_failure_returns_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _layer(100)
    packing_input = _packing_input(
        _fragment("mk", "safe", ContextFragmentPriority.MUST_KEEP),
    )
    current = "safe-current"

    def _force_failed_validation(
        original_content: str,
        optimized_content: str,
        **kwargs: object,
    ) -> ProtectedRegionValidationResult:
        return ProtectedRegionValidationResult(
            status=ProtectedRegionValidationStatus.FAILED,
            regions_checked=1,
            regions_preserved=0,
            regions_failed=1,
            failures=("protected region missing",),
        )

    monkeypatch.setattr(
        "intergrax.runtime.token_optimization.layers.budget_aware_packing.validate_protected_regions",
        _force_failed_validation,
    )

    result = layer.optimize(
        _request(current_content=current, packing_input=packing_input),
    )

    assert result.decision is TokenOptimizationLayerDecision.FALLBACK
    assert result.fallback_used is True
    assert result.output_content == current
    assert result.metadata["fallback_reason"] == "protected_region_validation_failed"


def test_metadata_reports_budget_unit_chars() -> None:
    layer = _layer(100)
    packing_input = _packing_input(
        _fragment("mk", "alpha", ContextFragmentPriority.MUST_KEEP),
    )
    result = layer.optimize(_request(packing_input=packing_input))

    assert result.metadata["budget_unit"] == "chars"


def test_metadata_reports_max_chars_final_chars_char_budget_satisfied() -> None:
    layer = _layer(100)
    packing_input = _packing_input(
        _fragment("mk", "alpha", ContextFragmentPriority.MUST_KEEP),
    )
    result = layer.optimize(_request(packing_input=packing_input))

    assert result.metadata["max_chars"] == 100
    assert result.metadata["final_chars"] == len("alpha")
    assert result.metadata["char_budget_satisfied"] is True


def test_packing_decisions_contain_no_raw_fragment_content() -> None:
    layer = _layer(100)
    secret = "secret-token-abc123"
    packing_input = _packing_input(
        _fragment("mk", secret, ContextFragmentPriority.MUST_KEEP),
        _fragment("drop", "droppable", ContextFragmentPriority.DROPPABLE),
    )
    result = layer.optimize(_request(packing_input=packing_input))

    metadata_blob = str(result.metadata["packing_decisions"])
    assert secret not in metadata_blob
    for decision in result.metadata["packing_decisions"]:
        assert set(decision.keys()) == {
            "fragment_id",
            "priority",
            "decision",
            "original_chars",
            "output_chars",
            "reason",
        }


_TOKEN_NAMED_METADATA_FIELDS = (
    "original_tokens",
    "optimized_tokens",
    "saved_tokens",
    "total_original_tokens",
    "total_optimized_tokens",
    "total_saved_tokens",
)

_CHAR_LEVEL_METADATA_FIELDS = (
    "budget_unit",
    "max_chars",
    "input_fragment_count",
    "selected_fragment_count",
    "dropped_fragment_count",
    "compacted_fragment_count",
    "must_keep_chars",
    "final_chars",
    "char_budget_satisfied",
    "packing_decisions",
    "saved_chars",
    "dropped_chars",
    "compacted_chars",
)


def test_metadata_excludes_context_packing_receipt_and_token_named_fields() -> None:
    layer = _layer(100)
    packing_input = _packing_input(
        _fragment("mk", "alpha", ContextFragmentPriority.MUST_KEEP),
        _fragment("drop", "droppable", ContextFragmentPriority.DROPPABLE),
    )
    result = layer.optimize(_request(packing_input=packing_input))

    assert "context_packing_receipt" not in result.metadata
    metadata_blob = str(result.metadata)
    for field_name in _TOKEN_NAMED_METADATA_FIELDS:
        assert field_name not in metadata_blob
    for field_name in _CHAR_LEVEL_METADATA_FIELDS:
        assert field_name in result.metadata


def test_packing_decisions_retain_char_level_fields() -> None:
    layer = _layer(100)
    packing_input = _packing_input(
        _fragment("mk", "alpha", ContextFragmentPriority.MUST_KEEP),
    )
    result = layer.optimize(_request(packing_input=packing_input))

    for decision in result.metadata["packing_decisions"]:
        assert "original_chars" in decision
        assert "output_chars" in decision


def test_normal_packing_does_not_override_previous_changes() -> None:
    layer = _layer(100)
    packing_input = _packing_input(
        _fragment("mk", "alpha", ContextFragmentPriority.MUST_KEEP),
    )
    result = layer.optimize(_request(packing_input=packing_input))

    assert result.decision is TokenOptimizationLayerDecision.APPLY
    assert result.previous_changes_overridden is False
    assert result.overridden_layer_ids == ()
    assert result.override_reason is None
