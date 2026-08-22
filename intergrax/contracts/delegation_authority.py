# © Artur Czarnecki. All rights reserved.

"""Typed delegation execution authority (IDT-FIX-B)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DelegationAuthorityError(ValueError):
    """Raised when delegated authority cannot be narrowed fail-closed."""


class ParentExecutionAuthority(BaseModel):
    """Trusted parent execution permission authority at the Nexus graph boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    permission_scopes: tuple[str, ...] = Field(default_factory=tuple)
    unrestricted: bool = False

    @classmethod
    def unknown(cls) -> ParentExecutionAuthority:
        return cls(permission_scopes=(), unrestricted=False)

    @classmethod
    def unrestricted_root(cls) -> ParentExecutionAuthority:
        return cls(permission_scopes=(), unrestricted=True)

    @classmethod
    def scoped(cls, scopes: tuple[str, ...]) -> ParentExecutionAuthority:
        return cls(permission_scopes=tuple(scopes), unrestricted=False)

    @property
    def is_unknown(self) -> bool:
        return not self.unrestricted and not self.permission_scopes


class EffectiveDelegationAuthority(BaseModel):
    """Effective authority granted to a delegated child after parent narrowing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_permission_scopes: tuple[str, ...] = Field(default_factory=tuple)
    parent_effective_scopes: tuple[str, ...] = Field(default_factory=tuple)
    parent_unrestricted: bool = False
    effective_permission_scopes: tuple[str, ...] = Field(default_factory=tuple)


EXECUTION_AUTHORITY_UNRESTRICTED_METADATA_KEY = "execution_authority_unrestricted"
EXECUTION_PERMISSION_SCOPES_METADATA_KEY = "execution_permission_scopes"
EFFECTIVE_DELEGATION_AUTHORITY_NODE_KEY = "effective_delegation_authority"
EFFECTIVE_PERMISSION_SCOPES_METADATA_KEY = "effective_permission_scopes"
REQUESTED_PERMISSION_SCOPES_METADATA_KEY = "requested_permission_scopes"


def resolve_root_parent_execution_authority(
    execution_authority: ParentExecutionAuthority | None,
) -> ParentExecutionAuthority:
    """Resolve trusted root authority from typed task/runtime field only."""
    if execution_authority is None:
        return ParentExecutionAuthority.unknown()
    return execution_authority


def validate_execution_authority_metadata_assertions(
    task_metadata: Mapping[str, Any],
    trusted: ParentExecutionAuthority,
) -> str | None:
    """Fail-closed when legacy metadata asserts authority conflicting with typed root."""
    unrestricted_asserted = (
        task_metadata.get(EXECUTION_AUTHORITY_UNRESTRICTED_METADATA_KEY) is True
    )
    raw_scopes = task_metadata.get(EXECUTION_PERMISSION_SCOPES_METADATA_KEY)
    asserted_scopes: tuple[str, ...] = ()
    if isinstance(raw_scopes, (list, tuple)):
        asserted_scopes = tuple(str(scope) for scope in raw_scopes if str(scope))

    if unrestricted_asserted and not trusted.unrestricted:
        return (
            "execution_authority metadata asserts unrestricted "
            "conflicting with typed root authority"
        )

    if asserted_scopes and not trusted.is_unknown and not trusted.unrestricted:
        trusted_set = set(trusted.permission_scopes)
        asserted_set = set(asserted_scopes)
        if asserted_set != trusted_set:
            return (
                "execution_authority metadata scopes "
                f"{sorted(asserted_set)!r} conflict with typed root "
                f"{sorted(trusted_set)!r}"
            )
    return None


def validate_effective_delegation_metadata_assertions(
    request_metadata: Mapping[str, Any],
    trusted: EffectiveDelegationAuthority,
) -> str | None:
    """Fail-closed when request metadata asserts effective scopes conflicting with typed authority."""
    raw = request_metadata.get(EFFECTIVE_PERMISSION_SCOPES_METADATA_KEY)
    if not isinstance(raw, (list, tuple)):
        return None
    asserted = tuple(str(scope) for scope in raw if str(scope))
    if not asserted:
        return None
    if asserted != trusted.effective_permission_scopes:
        return (
            "effective_permission_scopes metadata "
            f"{list(asserted)!r} conflicts with typed effective delegation authority "
            f"{list(trusted.effective_permission_scopes)!r}"
        )
    return None


def effective_delegation_to_parent_authority(
    effective: EffectiveDelegationAuthority,
) -> ParentExecutionAuthority:
    """Project child effective authority to the parent carrier for nested delegation."""
    if effective.parent_unrestricted:
        return ParentExecutionAuthority.scoped(effective.effective_permission_scopes)
    return ParentExecutionAuthority.scoped(effective.effective_permission_scopes)


def resolve_parent_execution_authority_for_node(
    graph: Any,
    node: Any,
    *,
    root_authority: ParentExecutionAuthority,
) -> ParentExecutionAuthority:
    """Resolve immediate parent authority from dependency effective authority chain."""
    if not node.depends_on:
        return root_authority

    resolved: list[ParentExecutionAuthority] = []
    for dep_id in node.depends_on:
        dep = graph.node_by_id(dep_id)
        stored = dep.metadata.get(EFFECTIVE_DELEGATION_AUTHORITY_NODE_KEY)
        if isinstance(stored, EffectiveDelegationAuthority):
            resolved.append(effective_delegation_to_parent_authority(stored))
        else:
            resolved.append(
                resolve_parent_execution_authority_for_node(
                    graph,
                    dep,
                    root_authority=root_authority,
                )
            )
    return _intersect_parent_authorities(resolved)


def mint_effective_delegation_authority(
    *,
    parent: ParentExecutionAuthority,
    requested_permission_scopes: tuple[str, ...],
) -> EffectiveDelegationAuthority:
    """Mint typed child authority: effective = parent ∩ requested (fail-closed on overreach)."""
    requested = tuple(requested_permission_scopes or ())

    if parent.is_unknown:
        if requested:
            raise DelegationAuthorityError(
                "delegation permission_scopes exceed unknown parent authority"
            )
        return EffectiveDelegationAuthority(
            requested_permission_scopes=requested,
            parent_effective_scopes=(),
            parent_unrestricted=False,
            effective_permission_scopes=(),
        )

    if parent.unrestricted:
        return EffectiveDelegationAuthority(
            requested_permission_scopes=requested,
            parent_effective_scopes=(),
            parent_unrestricted=True,
            effective_permission_scopes=requested,
        )

    parent_scopes = parent.permission_scopes
    if not requested:
        return EffectiveDelegationAuthority(
            requested_permission_scopes=requested,
            parent_effective_scopes=parent_scopes,
            parent_unrestricted=False,
            effective_permission_scopes=(),
        )

    over_broad = tuple(scope for scope in requested if scope not in parent_scopes)
    if over_broad:
        raise DelegationAuthorityError(
            f"delegation permission_scopes exceed parent authority: {over_broad!r}"
        )

    return EffectiveDelegationAuthority(
        requested_permission_scopes=requested,
        parent_effective_scopes=parent_scopes,
        parent_unrestricted=False,
        effective_permission_scopes=requested,
    )


def _intersect_parent_authorities(
    authorities: list[ParentExecutionAuthority],
) -> ParentExecutionAuthority:
    if not authorities:
        return ParentExecutionAuthority.unknown()
    if any(authority.unrestricted for authority in authorities):
        scoped = [authority for authority in authorities if not authority.unrestricted]
        if not scoped:
            return ParentExecutionAuthority.unrestricted_root()
        return _intersect_parent_authorities(scoped)
    if any(authority.is_unknown for authority in authorities):
        return ParentExecutionAuthority.unknown()
    common = set(authorities[0].permission_scopes)
    for authority in authorities[1:]:
        common &= set(authority.permission_scopes)
    return ParentExecutionAuthority.scoped(tuple(sorted(common)))
