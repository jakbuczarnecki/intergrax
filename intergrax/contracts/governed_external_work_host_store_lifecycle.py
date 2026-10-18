# © Artur Czarnecki. All rights reserved.

"""Process-lifetime ownership port for injected GCA host store bundles."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.governed_external_work_host_store_bundle import (
    GovernedExternalWorkHostStoreBundlePort,
)


@runtime_checkable
class OwnedGovernedExternalWorkHostStoreBundle(Protocol):
    """Durable GCA host bundle opened for one production process lifetime."""

    @property
    def host_store_bundle(self) -> GovernedExternalWorkHostStoreBundlePort:
        """Bundle exposed to strict composition and orchestrator wiring."""

    def close(self) -> None:
        """Release process-scoped persistence resources (e.g. SQLite connections)."""


__all__ = ["OwnedGovernedExternalWorkHostStoreBundle"]
