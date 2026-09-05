# © Artur Czarnecki. All rights reserved.

"""Canonical profile resolution layer identity (P1.1)."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class ProfileLayer(StrEnum):
    """Canonical resolution order — lower layers are applied first."""

    PLATFORM = "platform"
    PRODUCT = "product"
    APPLICATION = "application"
    AGENT = "agent"
    RUN = "run"
    EXECUTION = "execution"


class ProfileLayerOrder(IntEnum):
  """Deterministic ordering for layer inputs independent of caller list order."""

  PLATFORM = 0
  PRODUCT = 1
  APPLICATION = 2
  AGENT = 3
  RUN = 4
  EXECUTION = 5


CANONICAL_LAYER_ORDER: tuple[ProfileLayer, ...] = (
    ProfileLayer.PLATFORM,
    ProfileLayer.PRODUCT,
    ProfileLayer.APPLICATION,
    ProfileLayer.AGENT,
    ProfileLayer.RUN,
    ProfileLayer.EXECUTION,
)

_LAYER_ORDER: dict[ProfileLayer, ProfileLayerOrder] = {
    ProfileLayer.PLATFORM: ProfileLayerOrder.PLATFORM,
    ProfileLayer.PRODUCT: ProfileLayerOrder.PRODUCT,
    ProfileLayer.APPLICATION: ProfileLayerOrder.APPLICATION,
    ProfileLayer.AGENT: ProfileLayerOrder.AGENT,
    ProfileLayer.RUN: ProfileLayerOrder.RUN,
    ProfileLayer.EXECUTION: ProfileLayerOrder.EXECUTION,
}


def profile_layer_sort_key(layer: ProfileLayer) -> int:
    """Return stable sort key for canonical layer ordering."""
    return int(_LAYER_ORDER[layer])
