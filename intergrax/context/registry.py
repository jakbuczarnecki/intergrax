# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Context plugin catalog registry (Phase CE-1.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

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
class ContextPluginRegistry:
    """Mutable registry of context providers and optional pipeline overrides."""

    _providers: dict[str, ContextSourceProvider] = field(default_factory=dict)
    _ranker: ContextRanker | None = None
    _allocator: ContextBudgetAllocator | None = None
    _formatter: ContextFormatter | None = None
    _validator: ContextValidator | None = None

    def add_provider(self, provider: ContextSourceProvider, *, override: bool = False) -> None:
        provider_id = provider.provider_id.strip()
        if not provider_id:
            raise ValueError("provider_id must be non-empty")
        if provider_id in self._providers and not override:
            raise ValueError(f"Context provider '{provider_id}' is already registered")
        self._providers[provider_id] = provider

    def remove_provider(self, provider_id: str) -> None:
        self._providers.pop(provider_id.strip(), None)

    def list_providers(self) -> tuple[ContextSourceProvider, ...]:
        return tuple(self._providers.values())

    def get_provider(self, provider_id: str) -> ContextSourceProvider:
        normalized = provider_id.strip()
        try:
            return self._providers[normalized]
        except KeyError as exc:
            raise UnknownContextPluginError(normalized) from exc

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
