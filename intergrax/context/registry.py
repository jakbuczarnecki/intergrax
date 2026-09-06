# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Context plugin catalog registry (Phase CE-1.4, P1.9 hardening)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator

from intergrax.context.contracts import ContextProviderDescriptor
from intergrax.context.errors import ContextProviderRegistrationError
from intergrax.context.provider_descriptor import resolve_provider_descriptor
from intergrax.context.protocols import (
    ContextBudgetAllocator,
    ContextFormatter,
    ContextRanker,
    ContextSourceProvider,
    ContextValidator,
)


class UnknownContextPluginError(KeyError):
    """Raised when a context plugin id is not registered."""


ContextPluginRegisterFn = Callable[["ContextPluginRegistry"], None]


@dataclass
class _RegisteredProvider:
    descriptor: ContextProviderDescriptor
    provider: ContextSourceProvider


@dataclass
class ContextPluginRegistry:
    """Mutable registry of context providers and optional pipeline overrides."""

    _providers: dict[str, _RegisteredProvider] = field(default_factory=dict)
    _ranker: ContextRanker | None = None
    _allocator: ContextBudgetAllocator | None = None
    _formatter: ContextFormatter | None = None
    _validator: ContextValidator | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def add_provider(
        self,
        provider: ContextSourceProvider,
        *,
        override: bool = False,
        origin: str | None = None,
    ) -> None:
        descriptor = resolve_provider_descriptor(provider)
        if origin is not None and origin.strip():
            descriptor = ContextProviderDescriptor(
                provider_id=descriptor.provider_id,
                provider_version=descriptor.provider_version,
                supported_sources=descriptor.supported_sources,
                origin=origin.strip(),
            )
        with self._lock:
            existing = self._providers.get(descriptor.provider_id)
            if existing is not None and not override:
                raise ContextProviderRegistrationError(
                    f"Context provider '{descriptor.provider_id}' is already registered",
                )
            if (
                existing is not None
                and existing.descriptor == descriptor
                and existing.provider is not provider
                and not override
            ):
                raise ContextProviderRegistrationError(
                    f"Context provider '{descriptor.provider_id}' already registered "
                    f"with same descriptor but different object",
                )
            self._providers[descriptor.provider_id] = _RegisteredProvider(
                descriptor=descriptor,
                provider=provider,
            )

    def remove_provider(self, provider_id: str) -> None:
        normalized = provider_id.strip().lower()
        with self._lock:
            self._providers.pop(normalized, None)

    def get_provider(self, provider_id: str) -> ContextSourceProvider:
        normalized = provider_id.strip().lower()
        with self._lock:
            try:
                return self._providers[normalized].provider
            except KeyError as exc:
                raise UnknownContextPluginError(normalized) from exc

    def get_provider_descriptor(self, provider_id: str) -> ContextProviderDescriptor:
        normalized = provider_id.strip().lower()
        with self._lock:
            try:
                return self._providers[normalized].descriptor
            except KeyError as exc:
                raise UnknownContextPluginError(normalized) from exc

    def list_provider_descriptors(self) -> tuple[ContextProviderDescriptor, ...]:
        with self._lock:
            return tuple(
                item.descriptor
                for item in sorted(self._providers.values(), key=lambda entry: entry.descriptor.provider_id)
            )

    def list_providers(self) -> tuple[ContextSourceProvider, ...]:
        with self._lock:
            return tuple(
                item.provider
                for item in sorted(self._providers.values(), key=lambda entry: entry.descriptor.provider_id)
            )

    def snapshot_providers(self) -> tuple[tuple[ContextProviderDescriptor, ContextSourceProvider], ...]:
        """Immutable view of active providers for one assembly — registration order independent."""
        with self._lock:
            return tuple(
                (item.descriptor, item.provider)
                for item in sorted(self._providers.values(), key=lambda entry: entry.descriptor.provider_id)
            )

    def set_ranker(self, ranker: ContextRanker | None) -> None:
        self._ranker = ranker

    def set_allocator(self, allocator: ContextBudgetAllocator | None) -> None:
        self._allocator = allocator

    def set_formatter(self, formatter: ContextFormatter | None) -> None:
        self._formatter = formatter

    def set_validator(self, validator: ContextValidator | None) -> None:
        self._validator = validator

    @property
    def ranker(self) -> ContextRanker | None:
        return self._ranker

    @property
    def allocator(self) -> ContextBudgetAllocator | None:
        return self._allocator

    @property
    def formatter(self) -> ContextFormatter | None:
        return self._formatter

    @property
    def validator(self) -> ContextValidator | None:
        return self._validator


@dataclass(frozen=True)
class ContextPluginEntry:
    plugin_id: str
    version: str
    description: str
    register: ContextPluginRegisterFn

    def register_into(self, registry: ContextPluginRegistry) -> None:
        self.register(registry)


_CATALOG: dict[str, ContextPluginEntry] = {}


def register_context_plugin_entry(entry: ContextPluginEntry, *, override: bool = False) -> None:
    plugin_id = entry.plugin_id.strip().lower()
    if plugin_id in _CATALOG and not override:
        raise ValueError(f"Context plugin '{plugin_id}' is already registered")
    _CATALOG[plugin_id] = ContextPluginEntry(
        plugin_id=plugin_id,
        version=entry.version,
        description=entry.description,
        register=entry.register,
    )


def unregister_context_plugin(plugin_id: str) -> None:
    _CATALOG.pop(plugin_id.strip().lower(), None)


def clear_context_plugin_catalog() -> None:
    _CATALOG.clear()


def get_context_plugin(plugin_id: str) -> ContextPluginEntry:
    normalized = plugin_id.strip().lower()
    try:
        return _CATALOG[normalized]
    except KeyError as exc:
        raise UnknownContextPluginError(normalized) from exc


def iter_context_plugins() -> Iterator[ContextPluginEntry]:
    yield from _CATALOG.values()


def list_context_plugin_ids() -> list[str]:
    return sorted(_CATALOG)
