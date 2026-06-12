# © Artur Czarnecki. All rights reserved.

"""Capability alias registry, resolution, and manifest validation (APP-EVOL-3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from intergrax.applications.contracts.capability_alias import (
    CAPABILITY_ALIAS_REDIRECT_KEY,
    CapabilityAlias,
    CapabilityGovernanceProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import ApplicationManifest


@dataclass(frozen=True, slots=True)
class CapabilityAliasRegistry:
    """In-memory alias lookup table for one environment."""

    aliases: dict[str, CapabilityAlias]
    minimum_alias_window_days: int = 14


@dataclass(frozen=True, slots=True)
class CapabilityAliasResolution:
    """Outcome of resolving a task capability through the alias registry."""

    requested: str
    resolved: str
    redirected: bool
    blocked: bool
    reason: str | None = None


def build_capability_alias_registry(
    profile: CapabilityGovernanceProfile,
) -> CapabilityAliasRegistry:
    """Build alias lookup from environment capability governance profile."""
    aliases: dict[str, CapabilityAlias] = {}
    for entry in profile.aliases:
        aliases[entry.alias.strip()] = entry
    return CapabilityAliasRegistry(
        aliases=aliases,
        minimum_alias_window_days=profile.minimum_alias_window_days,
    )


def _parse_instant(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def resolve_capability_alias(
    capability: str,
    registry: CapabilityAliasRegistry,
    *,
    now: datetime,
    strict: bool,
) -> CapabilityAliasResolution:
    """Resolve legacy capability tokens to canonical routing values."""
    requested = capability.strip()
    if not requested:
        return CapabilityAliasResolution(
            requested=requested,
            resolved=requested,
            redirected=False,
            blocked=False,
        )

    entry = registry.aliases.get(requested)
    if entry is None:
        return CapabilityAliasResolution(
            requested=requested,
            resolved=requested,
            redirected=False,
            blocked=False,
        )

    instant = now.astimezone(UTC) if now.tzinfo is not None else now.replace(tzinfo=UTC)

    if entry.effective_from:
        effective_from = _parse_instant(entry.effective_from)
        if instant < effective_from:
            return CapabilityAliasResolution(
                requested=requested,
                resolved=requested,
                redirected=False,
                blocked=False,
            )

    if entry.sunset_at:
        sunset_at = _parse_instant(entry.sunset_at)
        if instant >= sunset_at:
            if strict:
                return CapabilityAliasResolution(
                    requested=requested,
                    resolved=requested,
                    redirected=False,
                    blocked=True,
                    reason=(
                        f"capability alias {requested!r} sunset at {entry.sunset_at}; "
                        "use canonical capability in STRICT mode"
                    ),
                )
            return CapabilityAliasResolution(
                requested=requested,
                resolved=requested,
                redirected=False,
                blocked=False,
                reason=f"capability alias {requested!r} past sunset — redirect disabled",
            )

    canonical = entry.canonical.strip()
    return CapabilityAliasResolution(
        requested=requested,
        resolved=canonical,
        redirected=canonical != requested,
        blocked=False,
    )


def capability_alias_redirect_payload(
    resolution: CapabilityAliasResolution,
) -> dict[str, str | bool]:
    """Serialize redirect audit metadata for task intake."""
    return {
        "requested_capability": resolution.requested,
        "canonical_capability": resolution.resolved,
        "redirected": resolution.redirected,
    }


def validate_capability_alias_entry(
    entry: CapabilityAlias,
    *,
    minimum_window_days: int,
) -> list[str]:
    """Validate one alias row including minimum migration window."""
    violations: list[str] = []
    if entry.effective_from and entry.sunset_at:
        try:
            start = _parse_instant(entry.effective_from)
            end = _parse_instant(entry.sunset_at)
        except ValueError as exc:
            violations.append(f"{entry.alias}: invalid effective_from/sunset_at — {exc}")
            return violations
        if end <= start:
            violations.append(f"{entry.alias}: sunset_at must be after effective_from")
        elif end - start < timedelta(days=minimum_window_days):
            violations.append(
                f"{entry.alias}: alias window shorter than {minimum_window_days} days",
            )
    return violations


def validate_capability_governance_profile(
    profile: CapabilityGovernanceProfile,
) -> list[str]:
    """Validate alias registry shape for CI."""
    violations: list[str] = []
    seen: set[str] = set()
    for entry in profile.aliases:
        violations.extend(
            validate_capability_alias_entry(
                entry,
                minimum_window_days=profile.minimum_alias_window_days,
            ),
        )
        alias = entry.alias.strip()
        if alias in seen:
            violations.append(f"duplicate capability alias: {alias!r}")
        seen.add(alias)
    return violations


def check_manifest_lists_canonical_capabilities(
    package: str,
    manifest: ApplicationManifest,
    registry: CapabilityAliasRegistry,
) -> list[str]:
    """Manifest roster must declare canonical capabilities, not legacy aliases."""
    violations: list[str] = []
    for binding in manifest.enabled_agents():
        for capability in binding.capabilities:
            cap = capability.strip()
            if cap in registry.aliases:
                violations.append(
                    f"{package}: AgentBinding lists alias {cap!r} — use canonical "
                    f"{registry.aliases[cap].canonical!r}",
                )
    return violations


def check_environment_capability_aliases(
    package: str,
    manifest: ApplicationManifest,
    profile: CapabilityGovernanceProfile,
) -> list[str]:
    """Run all capability alias checks for one host."""
    violations = validate_capability_governance_profile(profile)
    registry = build_capability_alias_registry(profile)
    violations.extend(check_manifest_lists_canonical_capabilities(package, manifest, registry))
    return violations


def strict_mode_for_environment(execution_mode: ExecutionMode) -> bool:
    """Whether alias sunset should block intake."""
    return execution_mode == ExecutionMode.STRICT


__all__ = [
    "CAPABILITY_ALIAS_REDIRECT_KEY",
    "CapabilityAliasRegistry",
    "CapabilityAliasResolution",
    "build_capability_alias_registry",
    "capability_alias_redirect_payload",
    "check_environment_capability_aliases",
    "resolve_capability_alias",
    "strict_mode_for_environment",
    "validate_capability_governance_profile",
]
