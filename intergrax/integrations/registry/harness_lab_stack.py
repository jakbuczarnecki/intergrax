# © Artur Czarnecki. All rights reserved.

"""Canonical lab harness integration stack (Phase S-Ops.1)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from intergrax.integrations.contracts.base import IntegrationMetadata, IntegrationStatus
from intergrax.integrations.registry.catalog import metadata_for_slug
from intergrax.integrations.registry.slugs import IntegrationSlug

HARNESS_LAB_STABLE_SLUGS: frozenset[str] = frozenset(
    {
        IntegrationSlug.SQLITE.value,
        IntegrationSlug.POSTGRESQL.value,
        IntegrationSlug.REDIS.value,
        IntegrationSlug.QDRANT.value,
        IntegrationSlug.SLACK.value,
        IntegrationSlug.SENTRY.value,
        IntegrationSlug.OTEL.value,
        IntegrationSlug.LAB_JSON.value,
        IntegrationSlug.LOG.value,
    }
)


class HarnessLabStackValidationError(ValueError):
    """Raised when a harness stack slug is missing or not marked stable."""


def harness_lab_stack_metadata() -> tuple[IntegrationMetadata, ...]:
    """Return catalog metadata rows for the lab harness stable stack (registration required)."""
    return tuple(metadata_for_slug(slug) for slug in sorted(HARNESS_LAB_STABLE_SLUGS))


def validate_harness_lab_stable_stack(
    *,
    catalog: Mapping[str, IntegrationMetadata] | None = None,
) -> None:
    """
    Assert every harness lab slug is registered with ``IntegrationStatus.STABLE``.

    Pass an explicit ``catalog`` map (slug → metadata) for unit tests without full bootstrap.
    """
    if catalog is None:
        for meta in harness_lab_stack_metadata():
            if meta.status is not IntegrationStatus.STABLE:
                raise HarnessLabStackValidationError(
                    f"Integration '{meta.slug}' must be stable for harness lab stack (got {meta.status.value})."
                )
        return

    for slug in HARNESS_LAB_STABLE_SLUGS:
        meta = catalog.get(slug)
        if meta is None:
            raise HarnessLabStackValidationError(f"Harness stack slug '{slug}' is missing from catalog.")
        if meta.status is not IntegrationStatus.STABLE:
            raise HarnessLabStackValidationError(
                f"Integration '{slug}' must be stable for harness lab stack (got {meta.status.value})."
            )


def list_harness_lab_stable_slugs() -> Sequence[str]:
    """Sorted slug list for documentation and operator runbooks."""
    return tuple(sorted(HARNESS_LAB_STABLE_SLUGS))
