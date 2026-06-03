# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integration health probes (Phase M.1, W-OPS.2 circuit breaker)."""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from intergrax.integrations._shared.circuit_breaker_registry import get_breaker_for_slug
from intergrax.integrations.contracts.base import (
    HealthStatus,
    IntegrationDependencyError,
    IntegrationEntry,
)
from intergrax.integrations.contracts.health_probe import IntegrationHealthProbe
from intergrax.integrations.registry.profile import IntegrationProfile

HealthProbe = Callable[[object], HealthStatus]


def health_check(instance: object, *, slug: str) -> HealthStatus:
    if isinstance(instance, IntegrationHealthProbe):
        result = instance.health()
        if isinstance(result, HealthStatus):
            return result
        return HealthStatus(slug=slug, healthy=bool(result))
    return HealthStatus(slug=slug, healthy=True, detail="no probe")


def health_check_entry(entry: IntegrationEntry, instance: object) -> HealthStatus:
    return health_check(instance, slug=entry.slug)


def _resolve_with_breaker(profile: IntegrationProfile, category: str, slug: str) -> object:
    from intergrax.integrations.registry.factory import resolve_from_profile

    breaker = get_breaker_for_slug(slug)
    return breaker.call(lambda: resolve_from_profile(profile, category))


def health_check_all(
    profile: IntegrationProfile,
    *,
    categories: Optional[Iterable[str]] = None,
    use_circuit_breaker: bool = True,
) -> list[HealthStatus]:
    """Run optional health probes for all slugs selected by ``profile``."""
    from intergrax.integrations.contracts.base import PROFILE_FIELD_BY_CATEGORY

    selected_categories = list(categories) if categories is not None else list(PROFILE_FIELD_BY_CATEGORY)
    results: list[HealthStatus] = []
    seen: set[str] = set()

    for category in selected_categories:
        slug = profile.slug_for_category(category)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        try:
            if use_circuit_breaker:
                instance = _resolve_with_breaker(profile, category, slug)
            else:
                from intergrax.integrations.registry.factory import resolve_from_profile

                instance = resolve_from_profile(profile, category)
        except IntegrationDependencyError as exc:
            results.append(HealthStatus(slug=slug, healthy=False, detail=str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 — aggregate startup probe failures
            results.append(HealthStatus(slug=slug, healthy=False, detail=str(exc)))
            continue
        results.append(health_check(instance, slug=slug))

    return results


def health_check_catalog_slugs(
    slugs: Iterable[str],
    *,
    use_circuit_breaker: bool = True,
) -> list[HealthStatus]:
    """Probe integrations by catalog slug (independent of ``IntegrationProfile`` bindings)."""
    from intergrax.integrations.registry.catalog import get_entry
    from intergrax.integrations.registry.factory import resolve

    results: list[HealthStatus] = []
    for raw_slug in slugs:
        slug = str(raw_slug).strip().lower()
        if not slug:
            continue
        entry = get_entry(slug)
        category = entry.categories[0]
        try:
            if use_circuit_breaker:
                breaker = get_breaker_for_slug(slug)

                def _resolve() -> object:
                    return resolve(category, slug=slug)

                instance = breaker.call(_resolve)
            else:
                instance = resolve(category, slug=slug)
        except IntegrationDependencyError as exc:
            results.append(HealthStatus(slug=slug, healthy=False, detail=str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 — aggregate startup probe failures
            results.append(HealthStatus(slug=slug, healthy=False, detail=str(exc)))
            continue
        results.append(health_check(instance, slug=slug))

    return results
