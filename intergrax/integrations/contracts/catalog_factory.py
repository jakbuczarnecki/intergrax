# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical typing for integration catalog factory materialization."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from intergrax.runtime.integrations.contracts import PlatformIntegrationContract

IntegrationFactoryConfigValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | Mapping[str, "IntegrationFactoryConfigValue"]
    | Sequence["IntegrationFactoryConfigValue"]
)

IntegrationFactory: TypeAlias = Callable[..., "PlatformIntegrationContract"]

__all__ = ["IntegrationFactory", "IntegrationFactoryConfigValue"]
