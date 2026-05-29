# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integration health probes (Phase M.1)."""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from intergrax.integrations.contracts.base import HealthStatus, IntegrationEntry
from intergrax.integrations.registry.profile import IntegrationProfile

HealthProbe = Callable[[object], HealthStatus]


def health_check(instance: object, *, slug: str) -> HealthStatus:
    probe = getattr(instance, "health", None)
    if callable(probe):
        result = probe()
        if isinstance(result, HealthStatus):
            return result
        return HealthStatus(slug=slug, healthy=bool(result))
    return HealthStatus(slug=slug, healthy=True, detail="no probe")


def health_check_entry(entry: IntegrationEntry, instance: object) -> HealthStatus:
    return health_check(instance, slug=entry.slug)


def health_check_all(
    profile: IntegrationProfile,
    *,
    categories: Optional[Iterable[str]] = None,
) -> list[HealthStatus]:
    """Run optional health probes for all slugs selected by ``profile``."""
    from intergrax.integrations.contracts.base import PROFILE_FIELD_BY_CATEGORY
    from intergrax.integrations.registry.factory import resolve_from_profile

    selected_categories = list(categories) if categories is not None else list(PROFILE_FIELD_BY_CATEGORY)
    results: list[HealthStatus] = []
    seen: set[str] = set()

    for category in selected_categories:
        slug = profile.slug_for_category(category)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        try:
            instance = resolve_from_profile(profile, category)
        except Exception as exc:  # noqa: BLE001 — aggregate startup probe failures
            results.append(HealthStatus(slug=slug, healthy=False, detail=str(exc)))
            continue
        results.append(health_check(instance, slug=slug))

    return results
