# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Exact runtime/package factory resolution for AP-10 registry projection.

Production projection must resolve ``AgentFactory`` from immutable
``(package_digest, factory_reference)`` authority bound to a ``RuntimeRevision``.
Host builders maps are not production authority.

``VENV_BUNDLE`` production loading is implemented by
:class:`~intergrax.applications._shared.venv_bundle_runtime_agent_factory_resolver.VenvBundleRuntimeAgentFactoryResolver`.
``OCI_IMAGE`` and ``SANDBOX_SIDECAR`` remain deferred.
"""

from __future__ import annotations

import threading
from typing import Final, Protocol

from intergrax.agent_distribution._digest import normalize_package_digest
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.runtime_revision import RuntimeRevision
from intergrax.applications.contracts.factory import AgentFactory

# Aggregate deferral remains True while OCI_IMAGE / SANDBOX_SIDECAR lack loaders.
# VENV_BUNDLE production resolver: ``venv_bundle_runtime_agent_factory_resolver``.
PRODUCTION_RUNTIME_FACTORY_ADAPTER_DEFERRED: Final = True

_FactoryKey = tuple[str, str | None, str | None]


class RuntimeAgentFactoryResolutionError(ValueError):
    """Exact runtime factory could not be resolved from immutable package authority."""


class RuntimeAgentFactoryResolver(Protocol):
    """Resolve an agent factory from immutable runtime/package authority."""

    def resolve_factory(
        self,
        *,
        runtime_revision: RuntimeRevision,
        package_digest: str,
        factory_reference: AgentBindingFactoryReference,
    ) -> AgentFactory:
        """Return the factory bound to ``package_digest`` + ``factory_reference``."""


def _factory_key(
    package_digest: str,
    factory_reference: AgentBindingFactoryReference,
) -> _FactoryKey:
    return (
        normalize_package_digest(package_digest),
        factory_reference.builder_key,
        factory_reference.factory_path,
    )


class InMemoryRuntimeAgentFactoryResolver:
    """Instance-scoped test/dev resolver keyed by package digest + factory reference."""

    def __init__(self) -> None:
        self._factories: dict[_FactoryKey, AgentFactory] = {}
        self._lock = threading.Lock()

    def register(
        self,
        *,
        package_digest: str,
        factory_reference: AgentBindingFactoryReference,
        factory: AgentFactory,
    ) -> None:
        """Bind one exact ``(package_digest, factory_reference)`` to a factory."""
        key = _factory_key(package_digest, factory_reference)
        with self._lock:
            existing = self._factories.get(key)
            if existing is not None and existing is not factory:
                raise RuntimeAgentFactoryResolutionError(
                    "conflicting factory registration for "
                    f"package_digest={key[0]!r} factory_reference={factory_reference}"
                )
            self._factories[key] = factory

    def resolve_factory(
        self,
        *,
        runtime_revision: RuntimeRevision,
        package_digest: str,
        factory_reference: AgentBindingFactoryReference,
    ) -> AgentFactory:
        digest = normalize_package_digest(package_digest)
        trusted = frozenset(runtime_revision.installed_agent_package_digests)
        if digest not in trusted:
            raise RuntimeAgentFactoryResolutionError(
                f"package_digest {digest!r} is not part of runtime revision "
                f"{runtime_revision.runtime_revision_id!r}"
            )
        key = _factory_key(digest, factory_reference)
        with self._lock:
            factory = self._factories.get(key)
        if factory is None:
            raise RuntimeAgentFactoryResolutionError(
                "cannot resolve factory for "
                f"package_digest={digest!r} factory_reference={factory_reference}"
            )
        return factory


__all__ = [
    "InMemoryRuntimeAgentFactoryResolver",
    "PRODUCTION_RUNTIME_FACTORY_ADAPTER_DEFERRED",
    "RuntimeAgentFactoryResolutionError",
    "RuntimeAgentFactoryResolver",
]
