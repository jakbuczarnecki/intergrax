# © Artur Czarnecki. All rights reserved.

"""TOKEN-8A: TokenOptimizationLayerRegistry unit tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.token_optimization.contracts import (
    StrategySafetyClass,
    TokenOptimizationLayerDecision,
    TokenOptimizationLayerDescriptor,
    TokenOptimizationLayerRef,
    TokenOptimizationLayerRequest,
    TokenOptimizationLayerResult,
    TokenOptimizationMechanism,
    TokenOptimizationStrategyKind,
    TokenOptimizationStrategyRef,
)
from intergrax.runtime.token_optimization.registry import TokenOptimizationLayerRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_BUILTIN_STRATEGY = TokenOptimizationStrategyRef(
    strategy_id="builtin.fake",
    mechanism=TokenOptimizationMechanism.DEDUPLICATION,
    kind=TokenOptimizationStrategyKind.DEDUPLICATION,
    safety_class=StrategySafetyClass.LOSSLESS,
    version="1",
)

_PLUGIN_STRATEGY = TokenOptimizationStrategyRef(
    strategy_id="plugin.custom",
    mechanism=TokenOptimizationMechanism.TOOL_OUTPUT_COMPACTION,
    kind=TokenOptimizationStrategyKind.EXTRACTIVE_FILTERING,
    safety_class=StrategySafetyClass.LOSSY,
    version="2",
    plugin_id="acme.optimizer",
)


class _FakeLayer:
    def __init__(
        self,
        *,
        layer_id: str,
        plugin_id: str | None = None,
        version: str = "1",
        built_in: bool = False,
    ) -> None:
        self._descriptor = TokenOptimizationLayerDescriptor(
            layer_id=layer_id,
            name=f"Fake {layer_id}",
            version=version,
            strategy=_PLUGIN_STRATEGY if plugin_id else _BUILTIN_STRATEGY,
            plugin_id=plugin_id,
            built_in=built_in,
            safety_class=StrategySafetyClass.LOSSLESS,
        )

    @property
    def descriptor(self) -> TokenOptimizationLayerDescriptor:
        return self._descriptor

    def optimize(
        self,
        request: TokenOptimizationLayerRequest,
    ) -> TokenOptimizationLayerResult:
        return TokenOptimizationLayerResult(
            layer_id=self._descriptor.layer_id,
            output_content=request.current_content,
            decision=TokenOptimizationLayerDecision.BYPASS,
        )


def test_register_and_get_layer() -> None:
    layer = _FakeLayer(layer_id="builtin.alpha")
    registry = TokenOptimizationLayerRegistry()
    registry.register(layer)

    assert registry.get("builtin.alpha") is layer
    assert registry.get("missing") is None


def test_registration_order_preserved() -> None:
    first = _FakeLayer(layer_id="builtin.first")
    second = _FakeLayer(layer_id="builtin.second")
    registry = TokenOptimizationLayerRegistry(layers=(first, second))

    assert registry.layer_ids == ("builtin.first", "builtin.second")
    assert [descriptor.layer_id for descriptor in registry.descriptors] == [
        "builtin.first",
        "builtin.second",
    ]


def test_duplicate_layer_id_rejected() -> None:
    registry = TokenOptimizationLayerRegistry()
    registry.register(_FakeLayer(layer_id="builtin.dup"))

    with pytest.raises(ValueError, match="duplicate layer_id"):
        registry.register(_FakeLayer(layer_id="builtin.dup"))


def test_resolve_matches_layer_id() -> None:
    layer = _FakeLayer(layer_id="builtin.alpha", version="3")
    registry = TokenOptimizationLayerRegistry(layers=(layer,))

    ref = TokenOptimizationLayerRef(layer_id="builtin.alpha", version="3")
    assert registry.resolve(ref) is layer


def test_resolve_plugin_id_mismatch_returns_none() -> None:
    layer = _FakeLayer(layer_id="plugin.layer", plugin_id="acme.optimizer")
    registry = TokenOptimizationLayerRegistry(layers=(layer,))

    ref = TokenOptimizationLayerRef(layer_id="plugin.layer", plugin_id="other.plugin")
    assert registry.resolve(ref) is None


def test_resolve_version_mismatch_returns_none() -> None:
    layer = _FakeLayer(layer_id="builtin.alpha", version="1")
    registry = TokenOptimizationLayerRegistry(layers=(layer,))

    ref = TokenOptimizationLayerRef(layer_id="builtin.alpha", version="9")
    assert registry.resolve(ref) is None


def test_builtin_and_plugin_descriptors_register() -> None:
    builtin = _FakeLayer(layer_id="builtin.catalog", built_in=True)
    plugin = _FakeLayer(layer_id="plugin.custom", plugin_id="acme.optimizer", built_in=False)
    registry = TokenOptimizationLayerRegistry(layers=(builtin, plugin))

    builtin_descriptor = registry.get("builtin.catalog").descriptor  # type: ignore[union-attr]
    plugin_descriptor = registry.get("plugin.custom").descriptor  # type: ignore[union-attr]

    assert builtin_descriptor.built_in is True
    assert builtin_descriptor.plugin_id is None
    assert plugin_descriptor.built_in is False
    assert plugin_descriptor.plugin_id == "acme.optimizer"


def test_registry_does_not_auto_import_builtin_catalog() -> None:
    registry = TokenOptimizationLayerRegistry()

    assert registry.layer_ids == ()
    assert registry.descriptors == ()
    assert registry.get("builtin.exact_deduplication") is None
