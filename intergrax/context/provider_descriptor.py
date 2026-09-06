# © Artur Czarnecki. All rights reserved.

"""Context provider descriptor helpers (P1.9)."""

from __future__ import annotations

import hashlib

from intergrax.context.contracts import ContextFragmentSource, ContextProviderDescriptor
from intergrax.context.errors import ContextProviderRegistrationError
from intergrax.context.protocols import ContextSourceProvider


def normalize_provider_id(provider_id: str) -> str:
    normalized = provider_id.strip().lower()
    if not normalized:
        raise ContextProviderRegistrationError("provider_id must be non-empty")
    return normalized


def build_provider_descriptor(
    provider_id: str,
    *,
    provider_version: str,
    supported_sources: frozenset[ContextFragmentSource],
    origin: str = "builtin",
) -> ContextProviderDescriptor:
    return ContextProviderDescriptor(
        provider_id=normalize_provider_id(provider_id),
        provider_version=provider_version,
        supported_sources=supported_sources,
        origin=origin,
    )


def resolve_provider_descriptor(provider: ContextSourceProvider) -> ContextProviderDescriptor:
    descriptor = provider.descriptor
    normalized_id = normalize_provider_id(provider.provider_id)
    if descriptor.provider_id != normalized_id:
        raise ContextProviderRegistrationError(
            f"provider descriptor id {descriptor.provider_id!r} "
            f"does not match provider_id {normalized_id!r}",
        )
    if descriptor.supported_sources != provider.supported_sources:
        raise ContextProviderRegistrationError(
            f"provider descriptor supported_sources mismatch for {normalized_id}",
        )
    return descriptor


def compute_provider_set_fingerprint(
    descriptors: tuple[ContextProviderDescriptor, ...],
) -> str:
    parts: list[str] = []
    for descriptor in sorted(descriptors, key=lambda item: item.provider_id):
        sources = ",".join(sorted(source.value for source in descriptor.supported_sources))
        parts.append(
            f"{descriptor.provider_id}@{descriptor.provider_version}|{sources}|{descriptor.origin}",
        )
    payload = ";".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
