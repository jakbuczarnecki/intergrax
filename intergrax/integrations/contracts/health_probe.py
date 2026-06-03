# © Artur Czarnecki. All rights reserved.

"""Optional health probe protocol for integration backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.base import HealthStatus


@runtime_checkable
class IntegrationHealthProbe(Protocol):
    """Backends that expose an explicit ``health()`` probe."""

    def health(self) -> HealthStatus | bool: ...
