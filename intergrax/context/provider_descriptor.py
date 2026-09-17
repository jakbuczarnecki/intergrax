# © Artur Czarnecki. All rights reserved.

"""Context provider descriptor helpers (P1.9)."""

from __future__ import annotations

import hashlib

from intergrax.context.contracts import (
    ContextAuthorityClass,
    ContextFragmentSource,
    ContextProviderDescriptor,
)
from intergrax.context.errors import ContextProviderRegistrationError
from intergrax.context.protocols import ContextSourceProvider
from intergrax.context.trusted_provider_bindings import trusted_authority_for_provider_id


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
    trusted_authority_class: ContextAuthorityClass | None = None,
    allowed_authority_classes: frozenset[ContextAuthorityClass] | None = None,
) -> ContextProviderDescriptor:
    normalized_id = normalize_provider_id(provider_id)
    trusted = trusted_authority_class
    if trusted is None and origin == "builtin":
        trusted = trusted_authority_for_provider_id(normalized_id)
    allowed = allowed_authority_classes
    if allowed is None and trusted is not None:
        allowed = frozenset({ContextAuthorityClass.UNASSIGNED, trusted})
    if allowed is None:
        allowed = frozenset({ContextAuthorityClass.UNASSIGNED})
    return ContextProviderDescriptor(
        provider_id=normalized_id,
        provider_version=provider_version,
        supported_sources=supported_sources,
        origin=origin,
        trusted_authority_class=trusted,
        allowed_authority_classes=allowed,
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
            f"{descriptor.provider_id}@{descriptor.provider_version}|{sources}|{descriptor.origin}|"
            f"{descriptor.trusted_authority_class.value if descriptor.trusted_authority_class else ''}",
        )
    payload = ";".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
