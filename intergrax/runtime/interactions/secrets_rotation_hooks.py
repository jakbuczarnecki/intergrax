# © Artur Czarnecki. All rights reserved.

"""Secrets rotation hooks via IntegrationProfile (IDEAL-4.4)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


RotationHook = Callable[[str], None]


@dataclass
class SecretsRotationRegistry:
    _hooks: dict[str, RotationHook] = field(default_factory=dict)

    def register(self, integration_slug: str, hook: RotationHook) -> None:
        self._hooks[integration_slug] = hook

    def rotate(self, integration_slug: str, *, reason: str = "") -> None:
        hook = self._hooks.get(integration_slug)
        if hook is None:
            raise KeyError(f"no rotation hook for integration: {integration_slug}")
        hook(reason or "scheduled_rotation")
