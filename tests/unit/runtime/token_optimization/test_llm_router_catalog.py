# © Artur Czarnecki. All rights reserved.

"""Unit tests for Token Optimization LLM router catalog (TOKEN-9)."""

from __future__ import annotations

import pytest

from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationRequest,
    TokenOptimizationSourceType,
)
from intergrax.runtime.token_optimization.layers import (
    BudgetAwareContextPackingLayerConfig,
    BudgetAwarePackingInput,
    ExtractiveFilteringLayerConfig,
)
from intergrax.runtime.token_optimization.llm_router_catalog import (
    create_token_optimization_router_configuration_catalog,
    packing_input_from_request,
)
from intergrax.runtime.token_optimization.llm_router_contracts import (
    TokenOptimizationLLMRouterPolicy,
    TokenOptimizationRouterConfigurationId,
)

pytestmark = pytest.mark.unit


def test_exact_seven_configuration_ids() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    assert len(catalog.configuration_ids) == 7
    assert set(catalog.configuration_ids) == set(TokenOptimizationRouterConfigurationId)


def test_deterministic_order() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    assert catalog.configuration_ids[0] is TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION
    assert catalog.configuration_ids[-1] is TokenOptimizationRouterConfigurationId.EXTRACTIVE_THEN_EXACT


def test_fixed_layer_order_exact_then_packing() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    spec = catalog.get(TokenOptimizationRouterConfigurationId.EXACT_THEN_PACKING)
    assert spec is not None
    assert [selection.layer_id for selection in spec.selections] == [
        "builtin.exact_deduplication",
        "builtin.budget_aware_context_packing",
    ]


def test_fixed_extractive_config() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    spec = catalog.get(TokenOptimizationRouterConfigurationId.EXTRACTIVE_ONLY)
    assert spec is not None
    config = spec.selections[0].config
    assert isinstance(config, ExtractiveFilteringLayerConfig)
    assert config.min_lines_before_filtering == 10
    assert config.head_lines == 3
    assert config.tail_lines == 3
    assert config.max_output_chars == 4000


def test_fixed_packing_configs() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    packing_only = catalog.get(TokenOptimizationRouterConfigurationId.PACKING_ONLY)
    mixed = catalog.get(TokenOptimizationRouterConfigurationId.EXACT_THEN_PACKING)
    assert isinstance(packing_only.selections[0].config, BudgetAwareContextPackingLayerConfig)
    assert packing_only.selections[0].config.max_chars == 80
    assert mixed.selections[1].config.max_chars == 50


def test_no_third_party_layers() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    for configuration_id in catalog.configuration_ids:
        spec = catalog.get(configuration_id)
        assert spec is not None
        for selection in spec.selections:
            assert selection.layer_id.startswith("builtin.")


def test_source_type_matrix() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    exact = catalog.get(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    extractive = catalog.get(TokenOptimizationRouterConfigurationId.EXTRACTIVE_ONLY)
    packing = catalog.get(TokenOptimizationRouterConfigurationId.PACKING_ONLY)
    assert TokenOptimizationSourceType.RAG_CONTEXT_PACK in exact.supported_source_types
    assert TokenOptimizationSourceType.TOOL_OUTPUT in extractive.supported_source_types
    assert TokenOptimizationSourceType.RETRIEVED_EVIDENCE in packing.supported_source_types


def test_lossy_flags() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    assert catalog.get(TokenOptimizationRouterConfigurationId.EXACT_ONLY).lossy is False
    assert catalog.get(TokenOptimizationRouterConfigurationId.EXTRACTIVE_ONLY).lossy is True


def test_packing_requirements() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    assert catalog.get(TokenOptimizationRouterConfigurationId.PACKING_ONLY).requires_packing_input
    assert not catalog.get(TokenOptimizationRouterConfigurationId.EXACT_ONLY).requires_packing_input


def test_no_mutable_shared_configuration_objects() -> None:
    catalog_a = create_token_optimization_router_configuration_catalog()
    catalog_b = create_token_optimization_router_configuration_catalog()
    spec_a = catalog_a.get(TokenOptimizationRouterConfigurationId.EXTRACTIVE_ONLY)
    spec_b = catalog_b.get(TokenOptimizationRouterConfigurationId.EXTRACTIVE_ONLY)
    assert spec_a.selections[0].config is not spec_b.selections[0].config


def test_available_for_includes_no_optimization() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    request = TokenOptimizationRequest(
        content="x",
        source_type=TokenOptimizationSourceType.SYSTEM_POLICY,
        policy=TokenOptimizationPolicy(
            enabled=True,
            profile=TokenOptimizationProfile.CONSERVATIVE,
        ),
    )
    available = catalog.available_for(request, TokenOptimizationLLMRouterPolicy())
    ids = {spec.configuration_id for spec in available}
    assert TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION in ids


def test_packing_input_from_request_rejects_dict() -> None:
    request = TokenOptimizationRequest(
        content="x",
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        metadata={"packing_input": {"fragments": []}},
    )
    assert packing_input_from_request(request) is None


def test_compile_no_optimization_has_no_pipeline() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    compiled = catalog.compile(TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION)
    assert compiled.pipeline_config is None


def test_compile_exact_only_builds_replace_pipeline() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    compiled = catalog.compile(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    assert compiled.pipeline_config is not None
    assert compiled.pipeline_config.pipeline_id == "router.exact_only"
    assert len(compiled.pipeline_config.layers) == 1


def test_packing_input_typed_instance_accepted() -> None:
    packing_input = BudgetAwarePackingInput(fragments=())
    request = TokenOptimizationRequest(
        content="x",
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        metadata={"packing_input": packing_input},
    )
    assert packing_input_from_request(request) is packing_input


def _request(
    *,
    source_type: TokenOptimizationSourceType = TokenOptimizationSourceType.RAG_CONTEXT_PACK,
    allow_lossy: bool = True,
    metadata: dict[str, object] | None = None,
    protected_regions: tuple = (),
) -> TokenOptimizationRequest:
    from intergrax.runtime.token_optimization.contracts import ProtectedRegion

    return TokenOptimizationRequest(
        content="x",
        source_type=source_type,
        policy=TokenOptimizationPolicy(
            enabled=True,
            profile=TokenOptimizationProfile.CONSERVATIVE,
            allow_lossy=allow_lossy,
        ),
        protected_regions=protected_regions,
        metadata=metadata or {},
    )


def test_available_rag_without_packing_input() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    available = catalog.available_for(_request(), TokenOptimizationLLMRouterPolicy())
    ids = {spec.configuration_id for spec in available}
    assert ids == {
        TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION,
        TokenOptimizationRouterConfigurationId.EXACT_ONLY,
    }


def test_available_rag_with_typed_packing_input() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    packing_input = BudgetAwarePackingInput(fragments=())
    available = catalog.available_for(
        _request(metadata={"packing_input": packing_input}),
        TokenOptimizationLLMRouterPolicy(),
    )
    ids = {spec.configuration_id for spec in available}
    assert ids == {
        TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION,
        TokenOptimizationRouterConfigurationId.EXACT_ONLY,
        TokenOptimizationRouterConfigurationId.PACKING_ONLY,
        TokenOptimizationRouterConfigurationId.EXACT_THEN_PACKING,
    }


def test_available_tool_output_allow_lossy_true() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    available = catalog.available_for(
        _request(source_type=TokenOptimizationSourceType.TOOL_OUTPUT, allow_lossy=True),
        TokenOptimizationLLMRouterPolicy(),
    )
    ids = {spec.configuration_id for spec in available}
    assert ids == {
        TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION,
        TokenOptimizationRouterConfigurationId.EXACT_ONLY,
        TokenOptimizationRouterConfigurationId.EXTRACTIVE_ONLY,
        TokenOptimizationRouterConfigurationId.EXACT_THEN_EXTRACTIVE,
        TokenOptimizationRouterConfigurationId.EXTRACTIVE_THEN_EXACT,
    }


def test_available_tool_output_allow_lossy_false() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    available = catalog.available_for(
        _request(source_type=TokenOptimizationSourceType.TOOL_OUTPUT, allow_lossy=False),
        TokenOptimizationLLMRouterPolicy(),
    )
    ids = {spec.configuration_id for spec in available}
    assert ids == {
        TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION,
        TokenOptimizationRouterConfigurationId.EXACT_ONLY,
    }


def test_available_protected_tool_output_excludes_lossy() -> None:
    from intergrax.runtime.token_optimization.contracts import (
        ProtectedRegion,
        ProtectedRegionKind,
    )

    catalog = create_token_optimization_router_configuration_catalog()
    protected = ProtectedRegion(kind=ProtectedRegionKind.IDENTIFIER, value="SECRET")
    available = catalog.available_for(
        _request(
            source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
            allow_lossy=True,
            protected_regions=(protected,),
        ),
        TokenOptimizationLLMRouterPolicy(),
    )
    ids = {spec.configuration_id for spec in available}
    assert TokenOptimizationRouterConfigurationId.EXTRACTIVE_ONLY not in ids
    assert TokenOptimizationRouterConfigurationId.EXACT_ONLY in ids


def test_available_unsupported_source_type_only_no_optimization() -> None:
    catalog = create_token_optimization_router_configuration_catalog()
    available = catalog.available_for(
        _request(source_type=TokenOptimizationSourceType.SYSTEM_POLICY),
        TokenOptimizationLLMRouterPolicy(),
    )
    ids = {spec.configuration_id for spec in available}
    assert ids == {TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION}
