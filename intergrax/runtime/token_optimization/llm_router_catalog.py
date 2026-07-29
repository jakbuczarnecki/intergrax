# © Artur Czarnecki. All rights reserved.

"""Approved deterministic configuration catalog for the Token Optimization LLM router (TOKEN-9)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.token_optimization.builtin_catalog import (
    BuiltInTokenOptimizationLayerSelection,
    create_builtin_token_optimization_layer_catalog,
)
from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationLayerRef,
    TokenOptimizationPipelineConfig,
    TokenOptimizationPipelineMode,
    TokenOptimizationRequest,
    TokenOptimizationSourceType,
)
from intergrax.runtime.token_optimization.layers import (
    BudgetAwareContextPackingLayerConfig,
    BudgetAwarePackingInput,
    ExtractiveFilteringLayerConfig,
)
from intergrax.runtime.token_optimization.llm_router_contracts import (
    TokenOptimizationLLMRouterPolicy,
    TokenOptimizationRouterConfigurationId,
)

_EXACT_DEDUP_ID = "builtin.exact_deduplication"
_EXTRACTIVE_ID = "builtin.extractive_filtering"
_BUDGET_PACKING_ID = "builtin.budget_aware_context_packing"

_PACKING_MAX_CHARS = 80
_MIXED_PACKING_MAX_CHARS = 50


def _extractive_config() -> ExtractiveFilteringLayerConfig:
    return ExtractiveFilteringLayerConfig(
        min_lines_before_filtering=10,
        head_lines=3,
        tail_lines=3,
        max_output_chars=4000,
    )


def _packing_config(max_chars: int) -> BudgetAwareContextPackingLayerConfig:
    return BudgetAwareContextPackingLayerConfig(max_chars=max_chars)

_EXACT_SOURCE_TYPES = frozenset(
    {
        TokenOptimizationSourceType.PROMPT,
        TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        TokenOptimizationSourceType.RETRIEVED_EVIDENCE,
        TokenOptimizationSourceType.CONVERSATION_HISTORY,
        TokenOptimizationSourceType.TOOL_OUTPUT,
    }
)

_EXTRACTIVE_SOURCE_TYPES = frozenset(
    {
        TokenOptimizationSourceType.TOOL_OUTPUT,
        TokenOptimizationSourceType.TERMINAL_OUTPUT,
        TokenOptimizationSourceType.LOG_OUTPUT,
    }
)

_PACKING_SOURCE_TYPES = frozenset(
    {
        TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        TokenOptimizationSourceType.RETRIEVED_EVIDENCE,
    }
)


def _layer_ref(layer_id: str) -> TokenOptimizationLayerRef:
    return TokenOptimizationLayerRef(layer_id=layer_id)


@dataclass(frozen=True, slots=True)
class TokenOptimizationRouterConfigurationSpec:
    configuration_id: TokenOptimizationRouterConfigurationId
    description: str
    supported_source_types: frozenset[TokenOptimizationSourceType]
    lossy: bool
    requires_packing_input: bool
    selections: tuple[BuiltInTokenOptimizationLayerSelection, ...]
    layer_refs: tuple[TokenOptimizationLayerRef, ...]

    def __post_init__(self) -> None:
        selection_ids = [selection.layer_id for selection in self.selections]
        ref_ids = [layer_ref.layer_id for layer_ref in self.layer_refs]
        if selection_ids != ref_ids:
            raise ValueError("selection order must equal layer-ref order")
        if len(selection_ids) != len(set(selection_ids)):
            raise ValueError("duplicate layer IDs are not allowed")


@dataclass(frozen=True, slots=True)
class TokenOptimizationRouterCompiledConfiguration:
    configuration_id: TokenOptimizationRouterConfigurationId
    selections: tuple[BuiltInTokenOptimizationLayerSelection, ...]
    layer_refs: tuple[TokenOptimizationLayerRef, ...]
    pipeline_config: TokenOptimizationPipelineConfig | None


class TokenOptimizationRouterConfigurationCatalog:
    """Closed catalog mapping LLM-selected configuration IDs to engine objects."""

    def __init__(
        self,
        *,
        specs: tuple[TokenOptimizationRouterConfigurationSpec, ...],
    ) -> None:
        lookup: dict[TokenOptimizationRouterConfigurationId, TokenOptimizationRouterConfigurationSpec] = {}
        ordered: list[TokenOptimizationRouterConfigurationId] = []
        for spec in specs:
            if spec.configuration_id in lookup:
                raise ValueError(f"duplicate configuration_id: {spec.configuration_id}")
            lookup[spec.configuration_id] = spec
            ordered.append(spec.configuration_id)
        self._specs = specs
        self._lookup = lookup
        self._configuration_ids = tuple(ordered)

    @property
    def configuration_ids(self) -> tuple[TokenOptimizationRouterConfigurationId, ...]:
        return self._configuration_ids

    def get(
        self,
        configuration_id: TokenOptimizationRouterConfigurationId,
    ) -> TokenOptimizationRouterConfigurationSpec | None:
        return self._lookup.get(configuration_id)

    def available_for(
        self,
        request: TokenOptimizationRequest,
        router_policy: TokenOptimizationLLMRouterPolicy,
    ) -> tuple[TokenOptimizationRouterConfigurationSpec, ...]:
        has_packing_input = packing_input_from_request(request) is not None
        protected_lossy_blocked = (
            bool(request.protected_regions)
            and router_policy.require_review_for_protected_lossy_content
        )
        available: list[TokenOptimizationRouterConfigurationSpec] = []
        for spec in self._specs:
            if spec.configuration_id is TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION:
                available.append(spec)
                continue
            if request.source_type not in spec.supported_source_types:
                continue
            if spec.lossy and not request.policy.allow_lossy:
                continue
            if spec.requires_packing_input and not has_packing_input:
                continue
            if spec.lossy and protected_lossy_blocked:
                continue
            available.append(spec)
        return tuple(available)

    def compile(
        self,
        configuration_id: TokenOptimizationRouterConfigurationId,
    ) -> TokenOptimizationRouterCompiledConfiguration:
        spec = self._lookup.get(configuration_id)
        if spec is None:
            raise ValueError(f"unknown configuration_id: {configuration_id}")

        builtin_catalog = create_builtin_token_optimization_layer_catalog()
        for selection in spec.selections:
            if builtin_catalog.get(selection.layer_id) is None:
                raise ValueError(f"unknown built-in layer: {selection.layer_id}")

        if configuration_id is TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION:
            return TokenOptimizationRouterCompiledConfiguration(
                configuration_id=configuration_id,
                selections=(),
                layer_refs=(),
                pipeline_config=None,
            )

        pipeline_config = TokenOptimizationPipelineConfig(
            pipeline_id=f"router.{configuration_id.value}",
            mode=TokenOptimizationPipelineMode.REPLACE,
            layers=spec.layer_refs,
        )
        return TokenOptimizationRouterCompiledConfiguration(
            configuration_id=configuration_id,
            selections=spec.selections,
            layer_refs=spec.layer_refs,
            pipeline_config=pipeline_config,
        )


def _packing_packing_only() -> TokenOptimizationRouterConfigurationSpec:
    selections = (
        BuiltInTokenOptimizationLayerSelection(
            layer_id=_BUDGET_PACKING_ID,
            config=_packing_config(_PACKING_MAX_CHARS),
        ),
    )
    layer_refs = (_layer_ref(_BUDGET_PACKING_ID),)
    return TokenOptimizationRouterConfigurationSpec(
        configuration_id=TokenOptimizationRouterConfigurationId.PACKING_ONLY,
        description="Budget-aware context packing for RAG evidence",
        supported_source_types=_PACKING_SOURCE_TYPES,
        lossy=False,
        requires_packing_input=True,
        selections=selections,
        layer_refs=layer_refs,
    )


def create_token_optimization_router_configuration_catalog() -> TokenOptimizationRouterConfigurationCatalog:
    """Return the canonical approved router configuration catalog."""
    all_source_types = frozenset(TokenOptimizationSourceType)
    _ = all_source_types

    specs: tuple[TokenOptimizationRouterConfigurationSpec, ...] = (
        TokenOptimizationRouterConfigurationSpec(
            configuration_id=TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION,
            description="Skip optimization; content is already acceptable",
            supported_source_types=frozenset(TokenOptimizationSourceType),
            lossy=False,
            requires_packing_input=False,
            selections=(),
            layer_refs=(),
        ),
        TokenOptimizationRouterConfigurationSpec(
            configuration_id=TokenOptimizationRouterConfigurationId.EXACT_ONLY,
            description="Lossless exact line deduplication",
            supported_source_types=_EXACT_SOURCE_TYPES,
            lossy=False,
            requires_packing_input=False,
            selections=(BuiltInTokenOptimizationLayerSelection(layer_id=_EXACT_DEDUP_ID),),
            layer_refs=(_layer_ref(_EXACT_DEDUP_ID),),
        ),
        TokenOptimizationRouterConfigurationSpec(
            configuration_id=TokenOptimizationRouterConfigurationId.EXTRACTIVE_ONLY,
            description="Lossy extractive filtering for noisy tool output",
            supported_source_types=_EXTRACTIVE_SOURCE_TYPES,
            lossy=True,
            requires_packing_input=False,
            selections=(
        BuiltInTokenOptimizationLayerSelection(
            layer_id=_EXTRACTIVE_ID,
            config=_extractive_config(),
        ),
            ),
            layer_refs=(_layer_ref(_EXTRACTIVE_ID),),
        ),
        _packing_packing_only(),
        TokenOptimizationRouterConfigurationSpec(
            configuration_id=TokenOptimizationRouterConfigurationId.EXACT_THEN_PACKING,
            description="Exact deduplication followed by budget-aware packing",
            supported_source_types=_PACKING_SOURCE_TYPES,
            lossy=False,
            requires_packing_input=True,
            selections=(
                BuiltInTokenOptimizationLayerSelection(layer_id=_EXACT_DEDUP_ID),
                BuiltInTokenOptimizationLayerSelection(
                    layer_id=_BUDGET_PACKING_ID,
                    config=_packing_config(_MIXED_PACKING_MAX_CHARS),
                ),
            ),
            layer_refs=(
                _layer_ref(_EXACT_DEDUP_ID),
                _layer_ref(_BUDGET_PACKING_ID),
            ),
        ),
        TokenOptimizationRouterConfigurationSpec(
            configuration_id=TokenOptimizationRouterConfigurationId.EXACT_THEN_EXTRACTIVE,
            description="Exact deduplication then extractive filtering",
            supported_source_types=frozenset({TokenOptimizationSourceType.TOOL_OUTPUT}),
            lossy=True,
            requires_packing_input=False,
            selections=(
                BuiltInTokenOptimizationLayerSelection(layer_id=_EXACT_DEDUP_ID),
        BuiltInTokenOptimizationLayerSelection(
            layer_id=_EXTRACTIVE_ID,
            config=_extractive_config(),
        ),
            ),
            layer_refs=(
                _layer_ref(_EXACT_DEDUP_ID),
                _layer_ref(_EXTRACTIVE_ID),
            ),
        ),
        TokenOptimizationRouterConfigurationSpec(
            configuration_id=TokenOptimizationRouterConfigurationId.EXTRACTIVE_THEN_EXACT,
            description="Extractive filtering then exact deduplication",
            supported_source_types=frozenset({TokenOptimizationSourceType.TOOL_OUTPUT}),
            lossy=True,
            requires_packing_input=False,
            selections=(
        BuiltInTokenOptimizationLayerSelection(
            layer_id=_EXTRACTIVE_ID,
            config=_extractive_config(),
        ),
                BuiltInTokenOptimizationLayerSelection(layer_id=_EXACT_DEDUP_ID),
            ),
            layer_refs=(
                _layer_ref(_EXTRACTIVE_ID),
                _layer_ref(_EXACT_DEDUP_ID),
            ),
        ),
    )
    return TokenOptimizationRouterConfigurationCatalog(specs=specs)


def packing_input_from_request(request: TokenOptimizationRequest) -> BudgetAwarePackingInput | None:
    """Extract typed packing input from request metadata."""
    raw = request.metadata.get("packing_input")
    if raw is None:
        return None
    if isinstance(raw, BudgetAwarePackingInput):
        return raw
    return None
