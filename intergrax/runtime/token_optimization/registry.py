# © Artur Czarnecki. All rights reserved.

"""Layer registry for Token Optimization pipeline composition (TOKEN-8A)."""

from __future__ import annotations

from collections.abc import Iterable

from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationLayer,
    TokenOptimizationLayerDescriptor,
    TokenOptimizationLayerRef,
)


class TokenOptimizationLayerRegistry:
    """Explicit, helper-only registry of optimization layers keyed by layer_id."""

    def __init__(
        self,
        layers: Iterable[TokenOptimizationLayer] = (),
    ) -> None:
        self._layers: dict[str, TokenOptimizationLayer] = {}
        self._registration_order: list[str] = []
        for layer in layers:
            self.register(layer)

    def register(
        self,
        layer: TokenOptimizationLayer,
    ) -> None:
        descriptor = layer.descriptor
        layer_id = descriptor.layer_id
        if layer_id in self._layers:
            raise ValueError(f"duplicate layer_id: {layer_id!r}")
        self._layers[layer_id] = layer
        self._registration_order.append(layer_id)

    def get(
        self,
        layer_id: str,
    ) -> TokenOptimizationLayer | None:
        return self._layers.get(layer_id)

    def resolve(
        self,
        layer_ref: TokenOptimizationLayerRef,
    ) -> TokenOptimizationLayer | None:
        layer = self._layers.get(layer_ref.layer_id)
        if layer is None:
            return None
        descriptor = layer.descriptor
        if layer_ref.plugin_id is not None and descriptor.plugin_id != layer_ref.plugin_id:
            return None
        if layer_ref.version is not None and descriptor.version != layer_ref.version:
            return None
        return layer

    @property
    def layer_ids(self) -> tuple[str, ...]:
        return tuple(self._registration_order)

    @property
    def descriptors(
        self,
    ) -> tuple[TokenOptimizationLayerDescriptor, ...]:
        return tuple(self._layers[layer_id].descriptor for layer_id in self._registration_order)
