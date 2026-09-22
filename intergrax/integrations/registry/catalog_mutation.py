# © Artur Czarnecki. All rights reserved.

"""Atomic catalog replacement and isolated preset materialization (domain primitive)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from intergrax.integrations.contracts.base import IntegrationEntry
from intergrax.integrations.registry.bootstrap import IntegrationPreset
from intergrax.integrations.registry.catalog_revision import (
    CatalogRevision,
    project_target_revision,
)

__all__ = [
    "CatalogReplaceOutcome",
    "CatalogReplaceResult",
    "build_catalog_entries_for_preset",
    "current_catalog_revision",
    "replace_catalog_if_revision",
    "reset_catalog_revision_tracking_for_tests",
]


class CatalogReplaceOutcome(StrEnum):
    COMMITTED = "committed"
    NO_CHANGE = "no_change"
    REVISION_CONFLICT = "revision_conflict"


@dataclass(frozen=True, slots=True)
class CatalogReplaceResult:
    outcome: CatalogReplaceOutcome
    before_revision: CatalogRevision
    after_revision: CatalogRevision


def current_catalog_revision() -> CatalogRevision:
    from intergrax.integrations.registry.catalog import (
        catalog_state_lock,
        read_catalog_revision_under_lock,
    )

    with catalog_state_lock():
        return read_catalog_revision_under_lock()


def build_catalog_entries_for_preset(
    preset: IntegrationPreset,
) -> dict[str, IntegrationEntry]:
    """Materialize preset registrations into an isolated dict without mutating live catalog."""
    from intergrax.integrations.registry.bootstrap_core import register_core_integrations
    from intergrax.integrations.registry.catalog import isolated_catalog_registration

    sink: dict[str, IntegrationEntry] = {}
    with isolated_catalog_registration(sink):
        register_core_integrations(override=True)
        if preset == "full":
            from intergrax.integrations.registry.bootstrap_extended import (
                register_extended_integrations,
            )

            register_extended_integrations(override=True)
    return dict(sink)


def replace_catalog_if_revision(
    *,
    expected_revision: CatalogRevision,
    candidate_entries: Mapping[str, IntegrationEntry],
) -> CatalogReplaceResult:
    """CAS commit — generation bumps only on material digest change."""
    from intergrax.integrations.registry.catalog import (
        _atomic_replace_catalog_entries,
        _assign_catalog_generation,
        catalog_state_lock,
        read_catalog_revision_under_lock,
    )

    with catalog_state_lock():
        before = read_catalog_revision_under_lock()
        if before != expected_revision:
            return CatalogReplaceResult(
                outcome=CatalogReplaceOutcome.REVISION_CONFLICT,
                before_revision=before,
                after_revision=before,
            )
        target = project_target_revision(before, candidate_entries)
        if target == before:
            return CatalogReplaceResult(
                outcome=CatalogReplaceOutcome.NO_CHANGE,
                before_revision=before,
                after_revision=before,
            )
        _atomic_replace_catalog_entries(dict(candidate_entries))
        _assign_catalog_generation(target.generation)
        return CatalogReplaceResult(
            outcome=CatalogReplaceOutcome.COMMITTED,
            before_revision=before,
            after_revision=target,
        )


def reset_catalog_revision_tracking_for_tests() -> None:
    """Test helper — reset generation counter after isolated catalog clears."""
    from intergrax.integrations.registry.catalog import (
        _assign_catalog_generation,
        catalog_state_lock,
    )

    with catalog_state_lock():
        _assign_catalog_generation(0)
