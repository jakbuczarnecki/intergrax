# © Artur Czarnecki. All rights reserved.

"""TOKEN-8B: BuiltInTokenOptimizationLayerCatalog unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from intergrax.runtime.token_optimization.builtin_catalog import (
    BuiltInTokenOptimizationLayerCatalog,
    BuiltInTokenOptimizationLayerSelection,
    BuiltInTokenOptimizationLayerSpec,
    create_builtin_token_optimization_layer_catalog,
)
from intergrax.runtime.token_optimization.contracts import (
    ContextFragmentPriority,
    TokenOptimizationBypassReason,
    TokenOptimizationLayerDecision,
    TokenOptimizationLayerRef,
    TokenOptimizationLayerRequest,
    TokenOptimizationLayerResult,
    TokenOptimizationPipelineConfig,
    TokenOptimizationPipelineMode,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationRequest,
    TokenOptimizationSourceType,
)
from intergrax.runtime.token_optimization.layers import (
    BudgetAwareContextPackingLayer,
    BudgetAwareContextPackingLayerConfig,
    BudgetAwarePackingFragment,
    BudgetAwarePackingInput,
    ExactDeduplicationLayer,
    ExactDeduplicationLayerConfig,
    ExtractiveFilteringLayer,
    ExtractiveFilteringLayerConfig,
)
from intergrax.runtime.token_optimization.pipeline import TokenOptimizationPipelineRunner

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_EXACT_DEDUP_ID = "builtin.exact_deduplication"
_EXTRACTIVE_ID = "builtin.extractive_filtering"
_BUDGET_PACKING_ID = "builtin.budget_aware_context_packing"


def _enabled_policy(**overrides: object) -> TokenOptimizationPolicy:
    base: dict[str, object] = {
        "enabled": True,
        "profile": TokenOptimizationProfile.CONSERVATIVE,
    }
    base.update(overrides)
    return TokenOptimizationPolicy(**base)  # type: ignore[arg-type]


def _pipeline_request(
    content: str,
    *,
    source_type: TokenOptimizationSourceType,
    policy: TokenOptimizationPolicy | None = None,
    metadata: dict[str, object] | None = None,
) -> TokenOptimizationRequest:
    return TokenOptimizationRequest(
        content=content,
        source_type=source_type,
        policy=policy or _enabled_policy(),
        metadata=metadata or {},
    )


def _layer_ref(layer_id: str) -> TokenOptimizationLayerRef:
    return TokenOptimizationLayerRef(layer_id=layer_id)


def _filtering_config(**overrides: object) -> ExtractiveFilteringLayerConfig:
    defaults: dict[str, object] = {
        "min_lines_before_filtering": 10,
        "head_lines": 3,
        "tail_lines": 3,
        "max_output_chars": 4000,
    }
    defaults.update(overrides)
    return ExtractiveFilteringLayerConfig(**defaults)  # type: ignore[arg-type]


def _noisy_long_output() -> str:
    lines = [f"INFO: progress step {index}" for index in range(150)]
    lines[75] = "ERROR: module compile failed"
    lines.append("INFO: final cleanup")
    return "\n".join(lines) + "\n"


def _packing_fragment(
    fragment_id: str,
    content: str,
    priority: ContextFragmentPriority,
) -> BudgetAwarePackingFragment:
    return BudgetAwarePackingFragment(
        fragment_id=fragment_id,
        content=content,
        priority=priority,
    )


# --- Test group A: canonical discovery ---


def test_canonical_catalog_layer_ids_and_specs() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()

    assert catalog.layer_ids == (
        _EXACT_DEDUP_ID,
        _EXTRACTIVE_ID,
        _BUDGET_PACKING_ID,
    )
    assert len(catalog.specs) == 3

    by_id = {spec.layer_id: spec for spec in catalog.specs}
    assert by_id[_EXACT_DEDUP_ID].config_type is ExactDeduplicationLayerConfig
    assert by_id[_EXACT_DEDUP_ID].configuration_required is False
    assert by_id[_EXTRACTIVE_ID].config_type is ExtractiveFilteringLayerConfig
    assert by_id[_EXTRACTIVE_ID].configuration_required is False
    assert by_id[_BUDGET_PACKING_ID].config_type is BudgetAwareContextPackingLayerConfig
    assert by_id[_BUDGET_PACKING_ID].configuration_required is True

    assert catalog.get(_EXACT_DEDUP_ID) is by_id[_EXACT_DEDUP_ID]
    assert catalog.get("builtin.unknown") is None


# --- Test group B: catalog validation ---


def test_catalog_rejects_duplicate_specification_ids() -> None:
    spec = BuiltInTokenOptimizationLayerSpec(
        layer_id=_EXACT_DEDUP_ID,
        config_type=ExactDeduplicationLayerConfig,
        configuration_required=False,
        factory=lambda _config: ExactDeduplicationLayer(),
    )
    with pytest.raises(ValueError, match="duplicate layer_id specification"):
        BuiltInTokenOptimizationLayerCatalog(specs=(spec, spec))


def test_catalog_preserves_specification_input_order() -> None:
    custom_specs = (
        BuiltInTokenOptimizationLayerSpec(
            layer_id="builtin.z",
            config_type=ExactDeduplicationLayerConfig,
            configuration_required=False,
            factory=lambda _config: ExactDeduplicationLayer(),
        ),
        BuiltInTokenOptimizationLayerSpec(
            layer_id="builtin.a",
            config_type=ExtractiveFilteringLayerConfig,
            configuration_required=False,
            factory=lambda _config: ExtractiveFilteringLayer(),
        ),
    )
    catalog = BuiltInTokenOptimizationLayerCatalog(specs=custom_specs)
    assert catalog.layer_ids == ("builtin.z", "builtin.a")


def test_spec_rejects_empty_layer_id() -> None:
    with pytest.raises(ValueError, match="layer_id must be a non-empty string"):
        BuiltInTokenOptimizationLayerSpec(
            layer_id="",
            config_type=ExactDeduplicationLayerConfig,
            configuration_required=False,
            factory=lambda _config: ExactDeduplicationLayer(),
        )


def test_selection_rejects_empty_layer_id() -> None:
    with pytest.raises(ValueError, match="layer_id must be a non-empty string"):
        BuiltInTokenOptimizationLayerSelection(layer_id="")


# --- Test group C: layer construction ---


def test_exact_deduplication_default_and_explicit_config() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()

    default_layer = catalog.create(_EXACT_DEDUP_ID)
    assert isinstance(default_layer, ExactDeduplicationLayer)
    assert default_layer.descriptor.layer_id == _EXACT_DEDUP_ID
    assert default_layer.descriptor.built_in is True

    explicit_layer = catalog.create(
        _EXACT_DEDUP_ID,
        ExactDeduplicationLayerConfig(case_sensitive=False),
    )
    assert isinstance(explicit_layer, ExactDeduplicationLayer)
    assert explicit_layer.descriptor.layer_id == _EXACT_DEDUP_ID
    assert explicit_layer.descriptor.built_in is True


def test_extractive_filtering_default_and_explicit_config() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()

    default_layer = catalog.create(_EXTRACTIVE_ID)
    assert isinstance(default_layer, ExtractiveFilteringLayer)
    assert default_layer.descriptor.layer_id == _EXTRACTIVE_ID
    assert default_layer.descriptor.built_in is True

    explicit_layer = catalog.create(_EXTRACTIVE_ID, _filtering_config())
    assert isinstance(explicit_layer, ExtractiveFilteringLayer)
    assert explicit_layer.descriptor.layer_id == _EXTRACTIVE_ID
    assert explicit_layer.descriptor.built_in is True


def test_budget_aware_packing_requires_explicit_config() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()

    layer = catalog.create(
        _BUDGET_PACKING_ID,
        BudgetAwareContextPackingLayerConfig(max_chars=50),
    )
    assert isinstance(layer, BudgetAwareContextPackingLayer)
    assert layer.descriptor.layer_id == _BUDGET_PACKING_ID
    assert layer.descriptor.built_in is True

    with pytest.raises(ValueError, match="configuration is required") as exc_info:
        catalog.create(_BUDGET_PACKING_ID)
    message = str(exc_info.value)
    assert _BUDGET_PACKING_ID in message
    assert "BudgetAwareContextPackingLayerConfig" in message
    assert "max_chars" not in message


@pytest.mark.parametrize(
    ("layer_id", "wrong_config"),
    [
        (_EXACT_DEDUP_ID, ExtractiveFilteringLayerConfig()),
        (_EXTRACTIVE_ID, ExactDeduplicationLayerConfig()),
        (_BUDGET_PACKING_ID, ExactDeduplicationLayerConfig()),
    ],
)
def test_wrong_config_types_rejected(layer_id: str, wrong_config: object) -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    with pytest.raises(TypeError, match="invalid configuration type"):
        catalog.create(layer_id, wrong_config)


def test_unknown_layer_id_rejected() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    with pytest.raises(ValueError, match="unknown built-in layer_id"):
        catalog.create("builtin.unknown")


# --- Test group D: fresh instances ---


def test_create_returns_fresh_layer_instances() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    first = catalog.create(_EXACT_DEDUP_ID)
    second = catalog.create(_EXACT_DEDUP_ID)
    assert first is not second


def test_create_registry_returns_fresh_registry_and_layers() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    selection = BuiltInTokenOptimizationLayerSelection(layer_id=_EXACT_DEDUP_ID)

    registry_one = catalog.create_registry((selection,))
    registry_two = catalog.create_registry((selection,))

    assert registry_one is not registry_two
    assert registry_one.get(_EXACT_DEDUP_ID) is not registry_two.get(_EXACT_DEDUP_ID)


# --- Test group E: registry creation ---


def test_create_registry_preserves_selection_order() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    selections = (
        BuiltInTokenOptimizationLayerSelection(layer_id=_EXTRACTIVE_ID),
        BuiltInTokenOptimizationLayerSelection(layer_id=_EXACT_DEDUP_ID),
        BuiltInTokenOptimizationLayerSelection(
            layer_id=_BUDGET_PACKING_ID,
            config=BudgetAwareContextPackingLayerConfig(max_chars=50),
        ),
    )
    registry = catalog.create_registry(selections)
    assert registry.layer_ids == (
        _EXTRACTIVE_ID,
        _EXACT_DEDUP_ID,
        _BUDGET_PACKING_ID,
    )


def test_create_registry_rejects_duplicate_selections() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    selections = (
        BuiltInTokenOptimizationLayerSelection(layer_id=_EXACT_DEDUP_ID),
        BuiltInTokenOptimizationLayerSelection(layer_id=_EXACT_DEDUP_ID),
    )
    with pytest.raises(ValueError, match="duplicate layer_id in selections"):
        catalog.create_registry(selections)


def test_create_registry_rejects_unknown_selected_layer() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    with pytest.raises(ValueError, match="unknown built-in layer_id"):
        catalog.create_registry(
            (BuiltInTokenOptimizationLayerSelection(layer_id="builtin.unknown"),),
        )


def test_create_registry_rejects_wrong_config_in_selection() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    with pytest.raises(TypeError, match="invalid configuration type"):
        catalog.create_registry(
            (
                BuiltInTokenOptimizationLayerSelection(
                    layer_id=_EXACT_DEDUP_ID,
                    config=ExtractiveFilteringLayerConfig(),
                ),
            ),
        )


def test_create_registry_rejects_missing_required_config() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    with pytest.raises(ValueError, match="configuration is required"):
        catalog.create_registry(
            (BuiltInTokenOptimizationLayerSelection(layer_id=_BUDGET_PACKING_ID),),
        )


def test_create_registry_empty_selections_returns_empty_registry() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    registry = catalog.create_registry(())
    assert registry.layer_ids == ()


# --- Test group F: post-construction invariants ---


@dataclass
class _FakeDescriptor:
    layer_id: str
    built_in: bool = True


class _FakeLayer:
    def __init__(
        self,
        *,
        layer_id: str,
        built_in: bool = True,
        include_optimize: bool = True,
    ) -> None:
        self.descriptor = _FakeDescriptor(layer_id=layer_id, built_in=built_in)
        if include_optimize:
            self.optimize = self._optimize

    def _optimize(
        self,
        request: TokenOptimizationLayerRequest,
    ) -> TokenOptimizationLayerResult:
        return TokenOptimizationLayerResult(
            layer_id=self.descriptor.layer_id,
            output_content=request.current_content,
            decision=TokenOptimizationLayerDecision.BYPASS,
        )


def _catalog_with_fake_spec(
    *,
    layer_id: str,
    factory: Any,
) -> BuiltInTokenOptimizationLayerCatalog:
    return BuiltInTokenOptimizationLayerCatalog(
        specs=(
            BuiltInTokenOptimizationLayerSpec(
                layer_id=layer_id,
                config_type=ExactDeduplicationLayerConfig,
                configuration_required=False,
                factory=factory,
            ),
        ),
    )


def test_catalog_rejects_descriptor_layer_id_mismatch() -> None:
    catalog = _catalog_with_fake_spec(
        layer_id="builtin.expected",
        factory=lambda _config: _FakeLayer(layer_id="builtin.actual"),
    )
    with pytest.raises(ValueError, match="descriptor.layer_id mismatch"):
        catalog.create("builtin.expected")


def test_catalog_rejects_built_in_false() -> None:
    catalog = _catalog_with_fake_spec(
        layer_id="builtin.expected",
        factory=lambda _config: _FakeLayer(layer_id="builtin.expected", built_in=False),
    )
    with pytest.raises(ValueError, match="descriptor.built_in must be True"):
        catalog.create("builtin.expected")


def test_catalog_rejects_factory_result_without_callable_optimize() -> None:
    class _NoOptimize:
        descriptor = _FakeDescriptor(layer_id="builtin.expected", built_in=True)

    catalog = _catalog_with_fake_spec(
        layer_id="builtin.expected",
        factory=lambda _config: _NoOptimize(),
    )
    with pytest.raises(TypeError, match="without callable optimize"):
        catalog.create("builtin.expected")


# --- Pipeline proofs ---


def test_pipeline_proof_exact_deduplication() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    registry = catalog.create_registry(
        (BuiltInTokenOptimizationLayerSelection(layer_id=_EXACT_DEDUP_ID),),
    )
    runner = TokenOptimizationPipelineRunner(registry=registry)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="catalog-dedupe",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_layer_ref(_EXACT_DEDUP_ID),),
    )
    content = "alpha\nalpha\nbeta"

    result = runner.run(
        request=_pipeline_request(
            content,
            source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
            policy=_enabled_policy(),
        ),
        config=config,
    )

    assert registry.get(_EXACT_DEDUP_ID) is not None
    assert result.original_content == content
    assert result.final_content == "alpha\nbeta"
    assert result.applied_layer_ids == (_EXACT_DEDUP_ID,)
    assert result.failed_layer_ids == ()


def test_pipeline_proof_extractive_filtering_with_allow_lossy() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    registry = catalog.create_registry(
        (
            BuiltInTokenOptimizationLayerSelection(
                layer_id=_EXTRACTIVE_ID,
                config=_filtering_config(),
            ),
        ),
    )
    runner = TokenOptimizationPipelineRunner(registry=registry)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="catalog-extractive",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_layer_ref(_EXTRACTIVE_ID),),
    )
    content = _noisy_long_output()

    result = runner.run(
        request=_pipeline_request(
            content,
            source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
            policy=_enabled_policy(
                allow_lossy=True,
                profile=TokenOptimizationProfile.BALANCED,
            ),
        ),
        config=config,
    )

    assert _EXTRACTIVE_ID in result.applied_layer_ids
    assert result.final_content != result.original_content
    assert result.failed_layer_ids == ()


def test_pipeline_proof_extractive_filtering_without_allow_lossy() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    registry = catalog.create_registry(
        (
            BuiltInTokenOptimizationLayerSelection(
                layer_id=_EXTRACTIVE_ID,
                config=_filtering_config(),
            ),
        ),
    )
    runner = TokenOptimizationPipelineRunner(registry=registry)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="catalog-extractive-policy",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_layer_ref(_EXTRACTIVE_ID),),
    )
    content = _noisy_long_output()

    result = runner.run(
        request=_pipeline_request(
            content,
            source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
            policy=_enabled_policy(allow_lossy=False),
        ),
        config=config,
    )

    assert result.final_content == result.original_content
    assert result.applied_layer_ids == ()
    assert result.bypassed_layer_ids == (_EXTRACTIVE_ID,)
    assert result.layer_results[0].bypass_reason is TokenOptimizationBypassReason.POLICY_DISALLOWED


def test_pipeline_proof_budget_aware_context_packing() -> None:
    catalog = create_builtin_token_optimization_layer_catalog()
    packing_input = BudgetAwarePackingInput(
        fragments=(
            _packing_fragment("mk1", "alpha", ContextFragmentPriority.MUST_KEEP),
            _packing_fragment("drop1", "x" * 100, ContextFragmentPriority.DROPPABLE),
        ),
    )
    registry = catalog.create_registry(
        (
            BuiltInTokenOptimizationLayerSelection(
                layer_id=_BUDGET_PACKING_ID,
                config=BudgetAwareContextPackingLayerConfig(max_chars=50),
            ),
        ),
    )
    runner = TokenOptimizationPipelineRunner(registry=registry)
    config = TokenOptimizationPipelineConfig(
        pipeline_id="catalog-budget-packing",
        mode=TokenOptimizationPipelineMode.REPLACE,
        layers=(_layer_ref(_BUDGET_PACKING_ID),),
    )

    result = runner.run(
        request=_pipeline_request(
            "assembled-current",
            source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
            metadata={"packing_input": packing_input},
        ),
        config=config,
    )

    assert result.failed_layer_ids == ()
    assert result.receipt_metadata["executed_layer_ids"] == [_BUDGET_PACKING_ID]
    assert result.applied_layer_ids == (_BUDGET_PACKING_ID,)
    assert result.final_content == "alpha"
