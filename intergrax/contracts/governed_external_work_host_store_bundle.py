# © Artur Czarnecki. All rights reserved.

"""Platform port for injected GCA host store bundle (strict external-work composition)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GovernedExternalWorkHostStoreBundlePort(Protocol):
    """Process-scoped GCA execution/receipt/bundle/continuation stores for strict wiring."""

    @property
    def is_durable(self) -> bool:
        """True only when all member stores are production-durable (not self-certifying)."""


__all__ = ["GovernedExternalWorkHostStoreBundlePort"]
