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
from intergrax.context.budget.compaction import ContextCompactionStrategy, NoOpContextCompactionStrategy
from intergrax.context.budget.token_counter import CharEstimateContextTokenCounter, ContextTokenCounter
from intergrax.context.protocols import (
    ContextBudgetAllocator,
    ContextConflictResolver,
    ContextFormatter,
    ContextRanker,
    ContextScoreNormalizer,
    ContextSemanticDeduper,
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
    _score_normalizer: ContextScoreNormalizer | None = None
    _semantic_deduper: ContextSemanticDeduper | None = None
    _conflict_resolver: ContextConflictResolver | None = None
    _formatter: ContextFormatter | None = None
    _validator: ContextValidator | None = None
    _token_counter: ContextTokenCounter | None = None
    _compaction_strategy: ContextCompactionStrategy | None = None
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
                trusted_authority_class=descriptor.trusted_authority_class,
                allowed_authority_classes=descriptor.allowed_authority_classes,
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
        with self._lock:
            self._ranker = ranker

    def set_allocator(self, allocator: ContextBudgetAllocator | None) -> None:
        with self._lock:
            self._allocator = allocator

    def set_score_normalizer(self, normalizer: ContextScoreNormalizer | None) -> None:
        with self._lock:
            self._score_normalizer = normalizer

    def set_semantic_deduper(self, deduper: ContextSemanticDeduper | None) -> None:
        with self._lock:
            self._semantic_deduper = deduper

    def set_conflict_resolver(self, resolver: ContextConflictResolver | None) -> None:
        with self._lock:
            self._conflict_resolver = resolver

    def set_formatter(self, formatter: ContextFormatter | None) -> None:
        with self._lock:
            self._formatter = formatter

    def set_validator(self, validator: ContextValidator | None) -> None:
        with self._lock:
            self._validator = validator

    def set_token_counter(self, counter: ContextTokenCounter | None) -> None:
        with self._lock:
            self._token_counter = counter

    def set_compaction_strategy(self, strategy: ContextCompactionStrategy | None) -> None:
        with self._lock:
            self._compaction_strategy = strategy

    @property
    def ranker(self) -> ContextRanker | None:
        with self._lock:
            return self._ranker

    @property
    def allocator(self) -> ContextBudgetAllocator | None:
        with self._lock:
            return self._allocator

    @property
    def score_normalizer(self) -> ContextScoreNormalizer | None:
        with self._lock:
            return self._score_normalizer

    @property
    def semantic_deduper(self) -> ContextSemanticDeduper | None:
        with self._lock:
            return self._semantic_deduper

    @property
    def conflict_resolver(self) -> ContextConflictResolver | None:
        with self._lock:
            return self._conflict_resolver

    @property
    def formatter(self) -> ContextFormatter | None:
        with self._lock:
            return self._formatter

    @property
    def validator(self) -> ContextValidator | None:
        with self._lock:
            return self._validator

    @property
    def token_counter(self) -> ContextTokenCounter | None:
        with self._lock:
            return self._token_counter

    @property
    def compaction_strategy(self) -> ContextCompactionStrategy:
        with self._lock:
            if self._compaction_strategy is None:
                return NoOpContextCompactionStrategy()
            return self._compaction_strategy

    def resolved_token_counter(self) -> ContextTokenCounter:
        with self._lock:
            return self._token_counter or CharEstimateContextTokenCounter()


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
