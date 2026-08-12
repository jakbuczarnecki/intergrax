# © Artur Czarnecki. All rights reserved.

"""Human-facing provider and connection resolution for conversational journeys."""

from __future__ import annotations

from dataclasses import dataclass
import re

from local_workspace_application.workspaces.tenant_connection_conversation_models import (
    TenantConnectionPlanningConnectionV1,
    TenantConnectionPlanningProviderV1,
)


class TenantConnectionConversationResolutionError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class TenantConnectionProviderResolution:
    provider_id: str | None
    ambiguous: bool


@dataclass(frozen=True, slots=True)
class TenantConnectionReferenceResolution:
    connection_ref: str | None
    ambiguous: bool


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_reference(value: str) -> str:
    return _NORMALIZE_RE.sub(" ", value.casefold()).strip()


_PROVIDER_ALIASES: dict[str, tuple[str, ...]] = {
    "google": ("google_workspace",),
    "google workspace": ("google_workspace",),
    "google drive": ("google_workspace",),
    "workspace google": ("google_workspace",),
    "microsoft": ("ms365_graph",),
    "microsoft 365": ("ms365_graph",),
    "m365": ("ms365_graph",),
    "ms365": ("ms365_graph",),
    "office 365": ("ms365_graph",),
    "slack": ("slack",),
}


def _provider_match_keys(provider: TenantConnectionPlanningProviderV1) -> set[str]:
    keys = {
        _normalize_reference(provider.provider_id),
        _normalize_reference(provider.safe_display_name),
    }
    provider_id = provider.provider_id.casefold()
    for alias, provider_ids in _PROVIDER_ALIASES.items():
        if provider_id in provider_ids:
            keys.add(_normalize_reference(alias))
    return {item for item in keys if item}


def _connection_match_keys(connection: TenantConnectionPlanningConnectionV1) -> set[str]:
    keys = {
        _normalize_reference(connection.connection_ref),
        _normalize_reference(connection.safe_display_name),
        _normalize_reference(connection.provider_id),
    }
    provider_id = connection.provider_id.casefold()
    for alias, provider_ids in _PROVIDER_ALIASES.items():
        if provider_id in provider_ids:
            keys.add(_normalize_reference(alias))
    return {item for item in keys if item}


def resolve_provider_reference(
    providers: tuple[TenantConnectionPlanningProviderV1, ...],
    *,
    provider_reference: str,
) -> TenantConnectionProviderResolution:
    needle = _normalize_reference(provider_reference)
    if not needle:
        return TenantConnectionProviderResolution(provider_id=None, ambiguous=False)

    alias_ids = _PROVIDER_ALIASES.get(needle)
    matches: list[str] = []
    for provider in providers:
        if alias_ids is not None:
            if provider.provider_id in alias_ids:
                matches.append(provider.provider_id)
            continue
        if needle in _provider_match_keys(provider):
            matches.append(provider.provider_id)

    unique = tuple(dict.fromkeys(matches))
    if not unique:
        return TenantConnectionProviderResolution(provider_id=None, ambiguous=False)
    if len(unique) > 1:
        return TenantConnectionProviderResolution(provider_id=None, ambiguous=True)
    return TenantConnectionProviderResolution(provider_id=unique[0], ambiguous=False)


def resolve_connection_reference(
    connections: tuple[TenantConnectionPlanningConnectionV1, ...],
    *,
    connection_reference: str,
    provider_id: str | None = None,
) -> TenantConnectionReferenceResolution:
    needle = _normalize_reference(connection_reference)
    if not needle:
        return TenantConnectionReferenceResolution(connection_ref=None, ambiguous=False)

    scoped = connections
    if provider_id is not None:
        scoped = tuple(item for item in connections if item.provider_id == provider_id)

    alias_ids = _PROVIDER_ALIASES.get(needle)
    matches: list[str] = []
    for connection in scoped:
        if alias_ids is not None:
            if connection.provider_id in alias_ids:
                matches.append(connection.connection_ref)
            continue
        if needle in _connection_match_keys(connection):
            matches.append(connection.connection_ref)

    unique = tuple(dict.fromkeys(matches))
    if not unique:
        return TenantConnectionReferenceResolution(connection_ref=None, ambiguous=False)
    if len(unique) > 1:
        return TenantConnectionReferenceResolution(connection_ref=None, ambiguous=True)
    return TenantConnectionReferenceResolution(connection_ref=unique[0], ambiguous=False)


__all__ = [
    "TenantConnectionConversationResolutionError",
    "TenantConnectionProviderResolution",
    "TenantConnectionReferenceResolution",
    "resolve_connection_reference",
    "resolve_provider_reference",
]
