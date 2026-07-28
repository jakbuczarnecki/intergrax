# © Artur Czarnecki. All rights reserved.

"""Explicit built-in layer catalog for Token Optimization pipeline composition (TOKEN-8B)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.runtime.token_optimization.contracts import TokenOptimizationLayer
from intergrax.runtime.token_optimization.layers import (
    BudgetAwareContextPackingLayer,
    BudgetAwareContextPackingLayerConfig,
    ExactDeduplicationLayer,
    ExactDeduplicationLayerConfig,
    ExtractiveFilteringLayer,
    ExtractiveFilteringLayerConfig,
)
from intergrax.runtime.token_optimization.registry import TokenOptimizationLayerRegistry
from intergrax.utils import attribute_access


@dataclass(frozen=True, slots=True)
class BuiltInTokenOptimizationLayerSpec:
    """Immutable specification for constructing one built-in optimization layer."""

    layer_id: str
    config_type: type[object]
    configuration_required: bool
    factory: Callable[[object | None], TokenOptimizationLayer]

    def __post_init__(self) -> None:
        if not self.layer_id or not self.layer_id.strip():
            raise ValueError("layer_id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class BuiltInTokenOptimizationLayerSelection:
    """Construction-only selection of a built-in layer and optional typed config."""

    layer_id: str
    config: object | None = None

    def __post_init__(self) -> None:
        if not self.layer_id or not self.layer_id.strip():
            raise ValueError("layer_id must be a non-empty string")


class BuiltInTokenOptimizationLayerCatalog:
    """Deterministic catalog of known built-in optimization layer specifications."""

    def __init__(
        self,
        *,
        specs: tuple[BuiltInTokenOptimizationLayerSpec, ...],
    ) -> None:
        lookup: dict[str, BuiltInTokenOptimizationLayerSpec] = {}
        ordered_ids: list[str] = []
        for spec in specs:
            if spec.layer_id in lookup:
                raise ValueError(f"duplicate layer_id specification: {spec.layer_id!r}")
            lookup[spec.layer_id] = spec
            ordered_ids.append(spec.layer_id)
        self._specs = specs
        self._lookup = lookup
        self._layer_ids = tuple(ordered_ids)

    @property
    def layer_ids(self) -> tuple[str, ...]:
        return self._layer_ids

    @property
    def specs(self) -> tuple[BuiltInTokenOptimizationLayerSpec, ...]:
        return self._specs

    def get(
        self,
        layer_id: str,
    ) -> BuiltInTokenOptimizationLayerSpec | None:
        return self._lookup.get(layer_id)

    def create(
        self,
        layer_id: str,
        config: object | None = None,
    ) -> TokenOptimizationLayer:
        spec = self._lookup.get(layer_id)
        if spec is None:
            raise ValueError(f"unknown built-in layer_id: {layer_id!r}")

        if config is None:
            if spec.configuration_required:
                raise ValueError(
                    f"configuration is required for layer_id {layer_id!r}; "
                    f"expected config type {spec.config_type.__name__}"
                )
        elif not isinstance(config, spec.config_type):
            raise TypeError(
                f"invalid configuration type for layer_id {layer_id!r}: "
                f"expected {spec.config_type.__name__}, got {type(config).__name__}"
            )

        layer = spec.factory(config)
        return _validate_constructed_layer(layer, spec)

    def create_registry(
        self,
        selections: tuple[BuiltInTokenOptimizationLayerSelection, ...],
    ) -> TokenOptimizationLayerRegistry:
        seen_layer_ids: set[str] = set()
        for selection in selections:
            if selection.layer_id in seen_layer_ids:
                raise ValueError(
                    f"duplicate layer_id in selections: {selection.layer_id!r}"
                )
            seen_layer_ids.add(selection.layer_id)

        registry = TokenOptimizationLayerRegistry()
        for selection in selections:
            registry.register(
                self.create(selection.layer_id, selection.config),
            )
        return registry


def create_builtin_token_optimization_layer_catalog() -> BuiltInTokenOptimizationLayerCatalog:
    """Return the canonical Intergrax built-in layer catalog in deterministic order."""
    return BuiltInTokenOptimizationLayerCatalog(specs=_CANONICAL_BUILTIN_LAYER_SPECS)


def _exact_deduplication_factory(config: object | None) -> TokenOptimizationLayer:
    if config is None:
        return ExactDeduplicationLayer()
    return ExactDeduplicationLayer(config=config)  # type: ignore[arg-type]


def _extractive_filtering_factory(config: object | None) -> TokenOptimizationLayer:
    if config is None:
        return ExtractiveFilteringLayer()
    return ExtractiveFilteringLayer(config=config)  # type: ignore[arg-type]


def _budget_aware_context_packing_factory(config: object | None) -> TokenOptimizationLayer:
    return BudgetAwareContextPackingLayer(config=config)  # type: ignore[arg-type]


_CANONICAL_BUILTIN_LAYER_SPECS: tuple[BuiltInTokenOptimizationLayerSpec, ...] = (
    BuiltInTokenOptimizationLayerSpec(
        layer_id="builtin.exact_deduplication",
        config_type=ExactDeduplicationLayerConfig,
        configuration_required=False,
        factory=_exact_deduplication_factory,
    ),
    BuiltInTokenOptimizationLayerSpec(
        layer_id="builtin.extractive_filtering",
        config_type=ExtractiveFilteringLayerConfig,
        configuration_required=False,
        factory=_extractive_filtering_factory,
    ),
    BuiltInTokenOptimizationLayerSpec(
        layer_id="builtin.budget_aware_context_packing",
        config_type=BudgetAwareContextPackingLayerConfig,
        configuration_required=True,
        factory=_budget_aware_context_packing_factory,
    ),
)


def _validate_constructed_layer(
    layer: object,
    spec: BuiltInTokenOptimizationLayerSpec,
) -> TokenOptimizationLayer:
    descriptor = attribute_access.optional(layer, "descriptor", None)
    if descriptor is None:
        raise TypeError(
            f"factory for layer_id {spec.layer_id!r} returned object without descriptor"
        )

    if not attribute_access.is_callable_attr(layer, "optimize"):
        raise TypeError(
            f"factory for layer_id {spec.layer_id!r} returned object without callable optimize"
        )

    descriptor_layer_id = attribute_access.optional(descriptor, "layer_id", None)
    if descriptor_layer_id is None:
        raise TypeError(
            f"factory for layer_id {spec.layer_id!r} returned descriptor without layer_id"
        )

    descriptor_built_in = attribute_access.optional(descriptor, "built_in", None)
    if descriptor_built_in is None:
        raise TypeError(
            f"factory for layer_id {spec.layer_id!r} returned descriptor without built_in"
        )

    if descriptor_layer_id != spec.layer_id:
        raise ValueError(
            f"factory invariant failed for layer_id {spec.layer_id!r}: "
            f"descriptor.layer_id mismatch (expected {spec.layer_id!r}, "
            f"got {descriptor_layer_id!r})"
        )

    if descriptor_built_in is not True:
        raise ValueError(
            f"factory invariant failed for layer_id {spec.layer_id!r}: "
            f"descriptor.built_in must be True"
        )

    return layer  # type: ignore[return-value]
