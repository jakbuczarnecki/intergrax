# © Artur Czarnecki. All rights reserved.

"""Execution authority policy contract and platform default (UE-8P2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.delegation_authority import (
    EffectiveDelegationAuthority,
    ParentExecutionAuthority,
    effective_delegation_to_parent_authority,
    mint_effective_delegation_authority,
)


@dataclass(frozen=True, slots=True)
class ChildAuthorityContext:
    """Inputs for resolving child execution authority under an active parent."""

    parent_authority: ParentExecutionAuthority
    requested_permission_scopes: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class ChildAuthorityResolution:
    """Resolved child authority and optional delegation evidence."""

    authority: ParentExecutionAuthority
    effective_delegation: EffectiveDelegationAuthority | None


@runtime_checkable
class ExecutionAuthorityPolicy(Protocol):
    """Pluggable strategy for narrowing child execution authority."""

    def resolve_child_authority(
        self,
        context: ChildAuthorityContext,
    ) -> ChildAuthorityResolution:
        """Resolve authority for a child execution under ``context.parent_authority``."""


class DefaultStrictAuthorityPolicy:
    """
    Platform default: child authority cannot exceed immediate parent authority (UE-8A).

    ``requested_permission_scopes is None`` inherits parent unchanged with no delegation
    evidence. A tuple (including empty) triggers explicit delegation narrowing.
    """

    def resolve_child_authority(
        self,
        context: ChildAuthorityContext,
    ) -> ChildAuthorityResolution:
        if context.requested_permission_scopes is None:
            return ChildAuthorityResolution(
                authority=context.parent_authority,
                effective_delegation=None,
            )

        effective = mint_effective_delegation_authority(
            parent=context.parent_authority,
            requested_permission_scopes=context.requested_permission_scopes,
        )
        return ChildAuthorityResolution(
            authority=effective_delegation_to_parent_authority(effective),
            effective_delegation=effective,
        )
