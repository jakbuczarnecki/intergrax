# © Artur Czarnecki. All rights reserved.

"""Capability health projection failures (P1.5)."""

from __future__ import annotations


class CapabilityHealthProviderConflictError(RuntimeError):
    """Duplicate provider routing identity — fail closed before projection."""

    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id
        super().__init__(
            f"duplicate capability health provider id {provider_id!r}",
        )
