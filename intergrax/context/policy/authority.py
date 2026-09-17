# © Artur Czarnecki. All rights reserved.

"""Trusted authority validation for context fragments (MEM-XINT-5-R)."""

from __future__ import annotations

from intergrax.context.contracts import (
    ContextAuthorityClass,
    ContextFragment,
    ContextProviderDescriptor,
    PROVIDER_FORBIDDEN_AUTHORITY_CLASSES,
    replace_context_fragment,
)
from intergrax.context.errors import ContextProviderContractViolationError

_AUTHORITY_RANK: dict[ContextAuthorityClass, int] = {
    ContextAuthorityClass.SYSTEM_CONTEXT: 100,
    ContextAuthorityClass.CANONICAL_MEMORY: 90,
    ContextAuthorityClass.DERIVED_MEMORY: 70,
    ContextAuthorityClass.RAG_EVIDENCE: 60,
    ContextAuthorityClass.TOOL_OBSERVATION: 55,
    ContextAuthorityClass.SESSION_EPISODIC: 50,
    ContextAuthorityClass.UNASSIGNED: 0,
}


def authority_rank(authority: ContextAuthorityClass) -> int:
    return _AUTHORITY_RANK.get(authority, 0)


def allowed_authority_classes(descriptor: ContextProviderDescriptor) -> frozenset[ContextAuthorityClass]:
    return descriptor.allowed_authority_classes


def enforce_provider_authority(
    fragment: ContextFragment,
    *,
    descriptor: ContextProviderDescriptor | None = None,
) -> ContextFragment:
    if descriptor is None:
        if fragment.authority_class in PROVIDER_FORBIDDEN_AUTHORITY_CLASSES:
            raise ValueError("provider self-elevated authority class")
        return fragment

    permitted = allowed_authority_classes(descriptor)
    declared = fragment.authority_class
    if declared not in permitted:
        raise ContextProviderContractViolationError(
            descriptor=descriptor,
            reason_code="provider.contract_violation",
            detail="authority class not permitted for provider descriptor",
        )
    if declared is ContextAuthorityClass.UNASSIGNED and descriptor.trusted_authority_class is not None:
        return replace_context_fragment(
            fragment,
            authority_class=descriptor.trusted_authority_class,
        )
    return fragment


def filter_fragments_by_authority_contract(
    fragments: list[ContextFragment],
    *,
    descriptors_by_id: dict[str, ContextProviderDescriptor],
) -> tuple[list[ContextFragment], list[tuple[ContextFragment, str]]]:
    kept: list[ContextFragment] = []
    excluded: list[tuple[ContextFragment, str]] = []
    for fragment in fragments:
        provenance = fragment.provider_provenance
        provider_id = provenance.provider_id if provenance is not None else ""
        descriptor = descriptors_by_id.get(provider_id)
        if descriptor is None:
            if fragment.authority_class in PROVIDER_FORBIDDEN_AUTHORITY_CLASSES:
                excluded.append((fragment, "authority.contract_violation"))
                continue
            kept.append(fragment)
            continue
        permitted = allowed_authority_classes(descriptor)
        if fragment.authority_class not in permitted:
            excluded.append((fragment, "authority.contract_violation"))
            continue
        kept.append(fragment)
    return kept, excluded
