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
    compute_catalog_state_digest,
    project_target_revision,
)

_CATALOG_GENERATION = 0


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
        _catalog_entries_for_revision,
        catalog_state_lock,
    )

    with catalog_state_lock():
        return _read_revision_under_lock()


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
        catalog_state_lock,
    )

    with catalog_state_lock():
        before = _read_revision_under_lock()
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
        global _CATALOG_GENERATION
        _CATALOG_GENERATION = target.generation
        return CatalogReplaceResult(
            outcome=CatalogReplaceOutcome.COMMITTED,
            before_revision=before,
            after_revision=target,
        )


def reset_catalog_revision_tracking_for_tests() -> None:
    """Test helper — reset generation counter after ``clear_catalog()``."""
    global _CATALOG_GENERATION
    from intergrax.integrations.registry.catalog import catalog_state_lock

    with catalog_state_lock():
        _CATALOG_GENERATION = 0


def _read_revision_under_lock() -> CatalogRevision:
    from intergrax.integrations.registry.catalog import _catalog_entries_for_revision

    entries = _catalog_entries_for_revision()
    digest = compute_catalog_state_digest(entries)
    generation = _sync_generation_for_entries(entries)
    return CatalogRevision(generation=generation, state_digest=digest)


def _sync_generation_for_entries(entries: Mapping[str, IntegrationEntry]) -> int:
    global _CATALOG_GENERATION
    if not entries:
        _CATALOG_GENERATION = 0
        return 0
    if _CATALOG_GENERATION == 0:
        _CATALOG_GENERATION = 1
    return _CATALOG_GENERATION
